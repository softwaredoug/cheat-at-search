This repo holds a set of helpers for running search experiments. Particularly any that incorporate LLMs or an Agent.

## Datasets

This repo holds search datasets. Each contained in its own module (ie esci_data, tmdb_data, etc).

The datasets in this repo honor a contract. You can import a corpus and judgments from them, ie

```python
from esci_data import corpus, judgments
```

Many get lazy-loaded / cached. IE, you'll notice code in the data modules that looks like:

```
def __getattr__(name):
    """Load dataset lazily."""
    ds = None
    if name in globals():
        return globals()[name]
    if name == "judgments" or name == "queries":
        ds = _judgments()
        globals()["judgments"] = ds
        queries = ds[['query', 'query_id']].drop_duplicates().reset_index(drop=True)
        globals()["queries"] = queries
        return globals()[name]
```

A query dataframe is guaranteed to have the following columns:

- query - A string representing the query
- query_id - A unique identifier for the query

The judgment dataframe is guaranteed to have the following columns:

- query - A string representing the query
- query_id - A unique identifier for the query
- doc_id - A unique identifier for the document
- grade - A relevance grade for the query-document pair. Higher is more relevant. The exact meaning of the grades is dataset-specific, but they are typically integers where 0 means not relevant and higher numbers indicate increasing relevance.


The corpus dataframe is guaranteed to have the following columns:

- doc_id - A unique identifier for the document
- title - The title of the document
- description - A description of the document. This is often a longer text field that provides more information about the document's content. It may be used for search and relevance judgments.


## Search Strategy

Core to this repo is the idea of a "SearchStrategy" that implements a search experiment.

A SearchStrategy works with a pandas datafrome, the corpus or index (the 'corpus' from a dataset above). The `search` takes a query and k. It returns the top k results for that query.

Specifically it returns two parallel lists or arrays:

- indices - The indices (iloc) of the top k results in the corpus dataframe
- scores - The relevance scores for those top k results. Higher is more relevant. The exact meaning of the scores is strategy-specific, but they are typically floats where higher numbers indicate increasing relevance.

A user of SearchStrategy that want to evaluate it call `search_all`. Usually this gets called via `run_strategy` (more on that in a bit)

### How its used

Most users of SearchStrategy inherit from it and implement `search`. This returns top k indices (ie iloc) of relevant results

In some cases `search_batch` would be implemented. This is a batch version of search that takes in a list of queries and returns a list of lists of indices and scores. That's useful for things like embedding-based search where you can batch the embedding calls.

### Caching

You'll notice that by default, search_all *caches*. That's useful to save time when repeating runs or re-analyzing. To be cacheable, a SearchStrategy must implement `cache_key`, which returns a string that uniquely identifies the strategy and its parameters. The cache is stored in a directory called `cache` in the root of the repo. Each cache file is named after the cache key with a .pkl extension.


## Root data directory

You'll notice that the repo allows mounting of a data directory (see data_dir.py). That's used throughout the repo for caching of different types of information.

By default, the data directory uses the platform's per-user cache location for `cheat-at-search`. This is `~/Library/Caches/cheat-at-search` on macOS, `~/.cache/cheat-at-search` on Linux, and the equivalent local application cache directory on Windows. This gives separate repositories and projects a shared location for large datasets and derived indexes.

An environment variable, CHEAT_AT_SEARCH_DATA_PATH, can be used as the highest priority override.

If `mount` gets called, it changes the data directory and preserves the legacy mount behavior. Usually this would be done before other processing. It can mount manual paths or, in a colab environment, attempt to mount google drive. An explicit mount therefore takes precedence over the default and environment-selected paths.

Changing the default does not migrate existing repository-local data. Set `CHEAT_AT_SEARCH_DATA_PATH` to an existing data directory when reusing a previously downloaded dataset.

## SearchArray

The lexical search in this library uses [SearchArray](https://github.com/softwaredoug/searcharray).

This library is a pandas extension array for performing BM25 search. The usual workflow is to:

```
from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenezer
from cheat_at_search.wands_data import corpus
import pandas as pd

# Indexing...
corpus['title_snowball'] = SearchArray.index(corpus['title'],
                                             tokenizer=snowball_tokenezer)

# BM25 score for every document for a given query
query = "red shoes"
tokenized_query = snowball_tokenezer(query)
scores = np.zeros(len(corpus))
for term in tokenized_query:
    scores += corpus['title_snowball'].array.score(term)
```


## Creation of lexical index up front

Most strategies, on most datasets, use some combination of lexical and/or embedding search.

In these cases, it's useful to precompute the lexical index for the most common tokenizer (snowball).

### Always index lexical after downloading

Most of the time, its useful to have a title_snowball, and description_snowball column to search.

As part of the data sest contract, lets always index these after downloading. This way, we can be sure that they're always there for any strategy that wants to use them.


## Embeddings

We also need to have an embedding registry for default embedding use cases.

This is intended to be built on demand. But cached to be retrieved later if the same embeddings are needed for this  dataset.

All embeddings here should be loaded and used via SentenceTransformers library. All embeddings use npy files and numpy dot product, etc as similarity. No vector database / index.


### Cache + Building

An embedding is requested using a corpus and different parameters.

- The dataset (esci, tmdb, etc)
- A hugging face path to a model. For example, "sentence-transformers/all-MiniLM-L6-v2"
- A text creation function (passage_fn). This is a function that takes in the corpus dataframe row, returns a string - that's the text to be embedded. For example, it might concatenate the title and description, or just use the description.

The cache key then is used to create an md5 signature of the dataset name, model name and passage_fn. This is used to check if the embeddings already exist in the cache. If they do, they're loaded. If not, they're built and saved to the cache for next time.

Here's the function for building / indexing the embeddings in chunks

def load_or_create_embeddings(
    corpus,
    passage_fn: Callable[[pd.DataFrame], pd.Series] = default_passage_fn,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    show_progress: bool = True,
) -> tuple[Iterable[np.ndarray], object | None]:


Other parameters names here:

- device - The device to run the embedding model on. For example, "mps", "cuda" or "cpu". If None, try "mps" then "cuda" then "cpu". Warn on cpu.
- show_progress - Whether to show a progress bar when building the embeddings (ie tqdm). This can be helpful for large corpora.

The function returns an iterator of individual embedding vectors and the model
used to create them. The iterator loads one cached chunk at a time rather than
assembling the full embedding matrix in memory.


### The default passage function

The default passage function puts title on the first line. Then description starting on the 3rd line. That's what's embedded.


### Chunking of the cache

The cache gets built in chunks iof 10K vectors at a time. Batched up via SentenceTransformers. This is to avoid memory issues when building the cache. And to allow resuming of the cache build if it gets interrupted. If the process gets interrupted, the already built chunks should be loaded and the build should resume from the next chunk that needs to be built.

Cache lives under a dedicated location under the data directory. The cache key is built from the dataset name, model name and a hash of the passage_fn (since that can change the embeddings significantly). Each chunk is stored as a separate npy file with a consistent naming convention.

Chunks are tracked via a manifest.

  - embeddings_<signature>.manifest.json contains:
    - signature, model, dim, count, chunk_size, num_chunks, completed_chunks.

Rebuild if:

   - If manifest signature/model mismatch or invalid JSON → rebuild.
   - If chunk file shape is incompatible → rebuild that chunk and update manifest.


## LLM Query / Doc Enrichment

This repo has utilities for using an LLM provider to enrich queries, documents, etc.

For the enrich/ module read enrich_prd.md here
