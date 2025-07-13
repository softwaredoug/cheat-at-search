import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from cheat_at_search.data_dir import ensure_data_subdir
from cheat_at_search.embedder import TextEmbedder, SentenceTransformerEmbedder
from cheat_at_search.strategy.embedding import EmbeddingSearch
import os
import pandas as pd
import logging
from cheat_at_search.search import run_strategy, ndcgs
from cheat_at_search.logger import log_to_stdout

from torch.utils.data import Dataset


enrich_output_dir = ensure_data_subdir('enrich_output')

enriched_products = pd.read_csv(os.path.join(enrich_output_dir, 'enriched_products.csv'))
enriched_products['product_description'].fillna('', inplace=True)
enriched_products['product_name'].fillna('', inplace=True)
enriched_queries = pd.read_csv(os.path.join(enrich_output_dir, 'query_attributes.csv'))


class WANDSDataset(Dataset):
    """
    PyTorch Dataset for the WANDS product search dataset.
    Produces (query, product_name, product_description, grade).
    """

    def __init__(self):
        data_path = os.path.join(enrich_output_dir, 'labeled_query_products.csv')
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        query = row['query']
        product_name = row['product_name'] if not isinstance(row['product_name'], float) else ''
        product_description = row['product_description'] if not isinstance(row['product_description'], float) else ''
        grade = int(row['grade'])
        return query, product_name, product_description, grade


def run_baseline_strategy(products, queries):
    """
    Run the baseline embedding strategy.
    """
    embedder = SentenceTransformerEmbedder(products, model_name="sentence-transformers/all-MiniLM-L6-v2")
    graded = run_strategy(EmbeddingSearch(products, embedder))
    avg_ndcg = ndcgs(graded).mean()
    print(f"Baseline NDCG (miniLM): {avg_ndcg:.4f}")


class TwoTowerModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", embedding_dim=768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)

        self.embedding_dim = embedding_dim

        # Optional: Projection layers to reduce embedding size
        self.doc_proj = nn.Linear(embedding_dim, embedding_dim)
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)

        # Document feature projections
        self.product_name_proj = nn.Linear(embedding_dim, embedding_dim)
        self.product_description_proj = nn.Linear(embedding_dim, embedding_dim)

    def encode_text(self, encoded):
        output = self.text_encoder(encoded['input_ids'], attention_mask=encoded['attention_mask'])
        # Use CLS token representation
        return output.last_hidden_state[:, 0, :]

    def forward(self, query_tokens, product_token_features):
        query_emb = self.encode_text(query_tokens)
        query_emb = self.query_proj(query_emb)

        # Product name and description
        doc_features = []
        name_embedding = self.encode_text(product_token_features['product_name'])
        name_embedding = self.product_name_proj(name_embedding)
        doc_features.append(name_embedding)

        description_embedding = self.encode_text(product_token_features['product_description'])
        description_embedding = self.product_description_proj(description_embedding)
        doc_features.append(description_embedding)

        # Concatenate product name and description embeddings
        doc_emb = torch.stack(doc_features, dim=0).mean(dim=0)
        doc_emb = self.doc_proj(doc_emb)

        # Normalize embeddings (optional, helps with cosine similarity)
        query_emb = nn.functional.normalize(query_emb, dim=1)
        doc_emb = nn.functional.normalize(doc_emb, dim=1)

        return query_emb, doc_emb

    def embed_query(self, query_text):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        tokens = self.tokenizer(query_text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            emb = self.encode_text(tokens)
            emb = self.query_proj(emb)
            emb = nn.functional.normalize(emb, dim=1)
        return emb

    def embed_document(self, product):
        product_name = product['product_name']
        product_description = product['product_description']
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            name_tokens = self.tokenizer(product_name, return_tensors="pt", truncation=True, padding=True).to(device)
            desc_tokens = self.tokenizer(product_description, return_tensors="pt", truncation=True, padding=True).to(device)

            name_embedding = self.encode_text(name_tokens)
            name_embedding = self.product_name_proj(name_embedding)

            description_embedding = self.encode_text(desc_tokens)
            description_embedding = self.product_description_proj(description_embedding)

            doc_emb = torch.stack([name_embedding, description_embedding], dim=0).mean(dim=0)
            doc_emb = self.doc_proj(doc_emb)
            doc_emb = nn.functional.normalize(doc_emb, dim=1)
        return doc_emb

    def embed_batch(self, product_batch):
        """Embed a single batch."""
        product_names = product_batch['product_name']
        product_descriptions = product_batch['product_description']
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            name_tokens = self.tokenizer(list(product_names), return_tensors="pt", truncation=True, padding=True).to(device)
            desc_tokens = self.tokenizer(list(product_descriptions), return_tensors="pt", truncation=True, padding=True).to(device)

            name_embedding = self.encode_text(name_tokens)
            name_embedding = self.product_name_proj(name_embedding)

            description_embedding = self.encode_text(desc_tokens)
            description_embedding = self.product_description_proj(description_embedding)

            doc_emb = torch.stack([name_embedding, description_embedding], dim=0).mean(dim=0)
            doc_emb = self.doc_proj(doc_emb)
            doc_emb = nn.functional.normalize(doc_emb, dim=1)
        return doc_emb


class TwoTowerEmbedder(TextEmbedder):
    def __init__(self, model, queries: pd.DataFrame, products: pd.DataFrame):
        self.queries = queries
        self.products = products
        self.model = model

    def query(self, keywords: str) -> torch.Tensor:
        """Embed the query tower."""
        queries = self.model.embed_query(keywords).cpu().numpy()
        return queries[0]

    def document(self, batch_size=32) -> torch.Tensor:
        """Embed the document tower."""
        embeddings = []
        for i in tqdm(range(0, len(self.products), batch_size), desc="Embedding documents"):
            batch = self.products.iloc[i:i + batch_size]
            emb = self.model.embed_batch(batch)
            embeddings.append(emb)
        concatted = torch.cat(embeddings)
        # To CPU and numpy
        return concatted.cpu().numpy()


def contrastive_loss(query_emb, doc_emb, temperature=0.05):
    scores = torch.matmul(query_emb, doc_emb.t()) / temperature
    labels = torch.arange(scores.size(0)).to(scores.device)
    return nn.CrossEntropyLoss()(scores, labels)


def run_strategy_for_epoch(products, queries, epoch):
    """Load model at epoch and run strategy."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load two_tower_model at the specified epoch
    model = TwoTowerModel().to(device)
    model_path = f"data/two_tower/two_tower_epoch_{epoch}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")

    # Run the strategy with the loaded model
    embedder = TwoTowerEmbedder(model, queries, products)
    embedder.query("foo")
    strategy = EmbeddingSearch(products, embedder)
    graded_results = run_strategy(strategy)
    avg_ndcg = ndcgs(graded_results).mean()
    print(f"Average NDCG after epoch {epoch}: {avg_ndcg:.4f}")


def train(start_epoch=0, epochs=3):

    # cuda then mps then cpu
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Initialize dataset and dataloader
    dataset = WANDSDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model and optimizer
    model = TwoTowerModel().to(device)
    if start_epoch > 0:
        model = TwoTowerModel().to(device)

        model_path = f"data/two_tower/two_tower_epoch_{start_epoch}.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            queries = list(batch[0])
            product_names = list(batch[1])
            product_descriptions = list(batch[2])

            query_tokens = tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(device)
            product_name_tokens = tokenizer(product_names, padding=True, truncation=True, return_tensors="pt").to(device)
            product_description_tokens = tokenizer(product_descriptions, padding=True, truncation=True, return_tensors="pt").to(device)
            product_text_tokens = {
                'product_name': product_name_tokens,
                'product_description': product_description_tokens
            }

            optimizer.zero_grad()
            query_emb, doc_emb = model(query_tokens, product_text_tokens)
            loss = contrastive_loss(query_emb, doc_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Cleanup
            del query_tokens, product_text_tokens, query_emb, doc_emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Checkpoint
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f"two_tower_epoch_{epoch + 1}.pth")
            print(f"Checkpoint saved for epoch {epoch + 1}")

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


if __name__ == "__main__":
    log_to_stdout(level=logging.INFO)
    # run_baseline_strategy(enriched_products, enriched_queries)
    # run_strategy_for_epoch(enriched_products, enriched_queries, 1)
    # run_strategy_for_epoch(enriched_products, enriched_queries, 2)
    # run_strategy_for_epoch(enriched_products, enriched_queries, 3)
    # run_strategy_for_epoch(enriched_products, enriched_queries, 4)
    # run_strategy_for_epoch(enriched_products, enriched_queries, 5)
    # run_strategy_for_epoch(enriched_products, enriched_queries, 6)
    start_epoch = 26
    run_strategy_for_epoch(enriched_products, enriched_queries, 1)
    run_strategy_for_epoch(enriched_products, enriched_queries, start_epoch // 2)
    run_strategy_for_epoch(enriched_products, enriched_queries, start_epoch)
    train(epochs=50, start_epoch=start_epoch)
