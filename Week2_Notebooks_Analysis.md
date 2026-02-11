# Week 2 Notebooks Analysis: Cheating at Search with LLMs

## Table of Contents

- [Phase 1 - High-Level Overview (Search Functionality POV)](#phase-1---high-level-overview)
- [Phase 2 - Architecture Analysis](#phase-2---architecture-analysis)
- [Phase 3 - Detailed Component Analysis](#phase-3---detailed-component-analysis)
- [Phase 4 - Python Pattern Explanations](#phase-4---python-pattern-explanations)

---

## Phase 1 - High-Level Overview

### What These Notebooks Do

The Week 2 notebooks are part of Doug Turnbull's "Cheat at Search with Agents" course. They explore how LLMs (Large Language Models) can improve e-commerce product search by enhancing query understanding -- specifically through **synonym expansion** and **query categorization**.

All three notebooks work with the same dataset and shared library (`cheat_at_search`), but each tests a different strategy for improving search relevance:

| Notebook | Strategy | Core Question |
|----------|----------|---------------|
| **1 - Synonyms from LLMs** | Expand queries with LLM-generated synonyms | "Can LLM synonyms improve a BM25 search?" |
| **2 - Perfect Categorization** | Classify queries into product categories using ground truth | "What is the theoretical maximum if we perfectly classify queries?" |
| **2a - Query Categories List** | Classify queries into multiple category paths using an LLM | "Can an LLM classify queries into a list of product taxonomy paths?" |

### The Search Problem Being Solved

E-commerce search suffers from the **vocabulary mismatch problem**: a user searching for "suede couch" won't find products listed as "microfiber sofa" because the words are different even though the intent is the same.

These notebooks test two approaches to bridging that gap:

1. **Synonym expansion** -- Ask an LLM to generate alternative words/phrases for the user's query, then search for those too.
2. **Query categorization** -- Ask an LLM to classify the query into a product taxonomy (e.g., "suede couch" maps to "Furniture / Living Room Furniture / Sofas"), then boost products in that category.

### The Dataset: WANDS

All notebooks use the [Wayfair Annotated Dataset (WANDS)](https://github.com/wayfair/WANDS), which contains:

- **~43,000 products** -- furniture and home goods with names, descriptions, categories, features, ratings
- **480 search queries** -- real e-commerce queries like "salon chair", "turquoise pillows", "wood bar stools"
- **Relevance judgments** -- human-labeled relevance grades (0 = not relevant, 1 = somewhat relevant, 2 = relevant) linking queries to products

### How Search Quality Is Measured

Search quality is evaluated using **NDCG (Normalized Discounted Cumulative Gain)**, a standard information retrieval metric. NDCG rewards search results that place the most relevant items at the top of the results list. A score of 1.0 means perfect ranking; lower scores mean relevant items appear too far down.

The notebooks compare each strategy against a **BM25 baseline** -- a well-known text matching algorithm that scores documents by term frequency and inverse document frequency.

---

## Phase 2 - Architecture Analysis

### Overall Code Organization

The notebooks rely on a shared Python library `cheat_at_search` installed from GitHub. The library provides the infrastructure; the notebooks provide the experimental strategies.

```
cheat_at_search (library)
├── data_dir.py        -- Data mounting and API key management
├── wands_data.py      -- WANDS dataset loading (products, judgments, queries)
├── search.py          -- Evaluation framework (run_strategy, ndcgs, ndcg_delta, vs_ideal, graded_bm25)
├── enrich.py          -- AutoEnricher: wraps LLM calls with structured output
├── tokenizers.py      -- snowball_tokenizer for stemming
└── strategy/
    └── strategy.py    -- SearchStrategy base class

External dependencies:
├── searcharray        -- SearchArray: in-memory lexical search engine
├── openai             -- OpenAI API client
├── pydantic           -- Structured data models (used for LLM structured output)
└── pandas / numpy     -- Data manipulation and scoring
```

### Architectural Pattern: Strategy Pattern

The notebooks follow a **Strategy pattern**. Each notebook defines a concrete `SearchStrategy` subclass that encapsulates a different search approach:

```
SearchStrategy (base class)
├── SynonymSearch      (Notebook 1) -- BM25 + LLM synonym boosting
├── CategorySearch     (Notebook 2) -- BM25 + ground-truth category boosting
└── CategorySearch     (Notebook 2a) -- BM25 + LLM-predicted category list boosting
```

Each strategy implements a `search(query, k=10)` method that returns the top-k document indices and their scores. The framework then evaluates all strategies using the same `run_strategy` / NDCG pipeline.

### Data Flow

Here is the end-to-end data flow that all three notebooks share:

```
1. SETUP
   ┌─────────────┐    ┌──────────────────┐
   │  WANDS Data  │───>│  products (DF)   │  ~43K products with text fields
   │  (CSV files) │    │  judgments (DF)   │  query-product relevance grades
   └─────────────┘    └──────────────────┘

2. INDEXING (in SearchStrategy.__init__)
   ┌──────────────────┐    ┌──────────────────────┐
   │  products["name"] │───>│  SearchArray.index()  │──> product_name_snowball
   │  products["desc"] │───>│  snowball_tokenizer   │──> product_description_snowball
   └──────────────────┘    └──────────────────────┘

3. QUERY-TIME ENRICHMENT (varies by notebook)
   ┌─────────┐    ┌──────────────┐    ┌──────────────────┐
   │  Query   │───>│  LLM / GT    │───>│  Structured Data │
   │  string  │    │  Enrichment  │    │  (Pydantic model)│
   └─────────┘    └──────────────┘    └──────────────────┘

4. SEARCH (in SearchStrategy.search)
   ┌──────────────────┐    ┌─────────────┐    ┌───────────┐
   │  BM25 base score │ +  │  Enrichment │ =  │  Final    │
   │  (name + desc)   │    │  boost      │    │  ranking  │
   └──────────────────┘    └─────────────┘    └───────────┘

5. EVALUATION
   ┌───────────┐    ┌─────────────┐    ┌──────────────┐
   │  Rankings  │───>│ run_strategy │───>│  graded DF   │──> ndcgs() ──> mean NDCG
   │  per query │    │ + judgments  │    │  with NDCG   │──> ndcg_delta() vs baseline
   └───────────┘    └─────────────┘    └──────────────┘
```

### How the Three Notebooks Differ

The notebooks share steps 1, 2, 4 (base scoring), and 5. They differ in step 3 (enrichment) and how that enrichment is incorporated into the score:

**Notebook 1 (Synonyms):** The LLM generates synonym mappings. Synonym tokens get additional BM25 scoring against the same fields.

**Notebook 2 (Perfect Categorization):** Ground truth category labels are looked up directly (no LLM). Products matching the predicted category get a constant score boost (e.g., +100 for category, +50 for subcategory).

**Notebook 2a (Category List):** An LLM predicts a list of fully-qualified category paths. Products matching any predicted category get a constant score boost. Evaluation uses Jaccard similarity and recall instead of just precision.

---

## Phase 3 - Detailed Component Analysis

### 3.1 Library Components

#### `mount()` -- Data Directory Setup

```python
from cheat_at_search.data_dir import mount
mount(use_gdrive=True)
```

Configures where data files (WANDS dataset, API keys) are stored. In Google Colab, this can mount Google Drive for persistence across sessions. Locally, you can specify a `manual_path`.

#### `products` and `judgments` -- WANDS Data

```python
from cheat_at_search.wands_data import products, judgments
```

- `products` is a pandas DataFrame with ~43,000 rows and columns like `product_name`, `product_description`, `category hierarchy`, `product_class`, `average_rating`, plus pre-computed stemmed fields (`product_name_snowball`, `product_description_snowball`).
- `judgments` maps (query, product_id) pairs to relevance grades (0, 1, or 2).

#### `snowball_tokenizer` -- Text Tokenization and Stemming

```python
from cheat_at_search.tokenizers import snowball_tokenizer
snowball_tokenizer("fancy furniture")
# Returns: ['fanci', 'furnitur']
```

Takes a text string and returns a list of stemmed tokens. Uses the [Snowball stemmer](https://snowballstem.org/) which reduces words to their root form. This means "furniture", "furnishing", and "furnishings" all become the same stem, improving recall.

#### `SearchArray` -- In-Memory Lexical Search

```python
from searcharray import SearchArray

# Create a searchable index from a pandas Series of text
products['product_name_snowball'] = SearchArray.index(
    products['product_name'],
    snowball_tokenizer
)

# Score documents for a query term
scores = products['product_name_snowball'].array.score("furnitur")
```

SearchArray is a numpy-backed lexical search library by the course author. It creates an inverted index (a mapping from term to documents containing that term) and supports BM25 scoring. Think of it as a tiny, in-process Elasticsearch.

Key method: `.score(token_or_tokens)` returns a numpy array of BM25 scores, one per document. For a multi-token query, it computes a phrase-aware score.

#### `SearchStrategy` -- Base Class for Strategies

```python
from cheat_at_search.strategy.strategy import SearchStrategy

class MyStrategy(SearchStrategy):
    def __init__(self, products):
        super().__init__(products)
        # Build your index here

    def search(self, query, k=10):
        # Return (top_k_indices, scores)
        pass
```

The base class that all search strategies inherit from. Subclasses must implement `search(query, k)` which returns a tuple of (numpy array of document indices, numpy array of scores).

#### `AutoEnricher` -- LLM Wrapper with Structured Output

```python
from cheat_at_search.enrich import AutoEnricher

enricher = AutoEnricher(
    model="openai/gpt-4o",
    system_prompt="You are a helpful AI assistant...",
    response_model=QueryWithSynonyms  # A Pydantic model
)

result = enricher.enrich("Generate synonyms for: suede couch")
# result is an instance of QueryWithSynonyms
```

`AutoEnricher` wraps LLM API calls and enforces **structured output** -- meaning the LLM must return data conforming to a Pydantic model schema. This replaces "please return JSON" prompting with a guaranteed, parseable response.

The three parameters:
- `model` -- which LLM to use (e.g., `"openai/gpt-4o"`, `"openai/gpt-5-nano"`)
- `system_prompt` -- primes the LLM for its task
- `response_model` -- the Pydantic class defining the expected output shape

#### Evaluation Functions

```python
from cheat_at_search.search import run_strategy, graded_bm25, ndcgs, ndcg_delta, vs_ideal
```

| Function | Purpose | Example |
|----------|---------|---------|
| `run_strategy(strategy, judgments)` | Runs every WANDS query through the strategy, grades results | `graded_syns = run_strategy(syns, judgments)` |
| `graded_bm25` | Pre-computed BM25 baseline results | Used as the comparison benchmark |
| `ndcgs(graded_df)` | Extracts per-query NDCG scores | `ndcgs(graded_bm25).mean()` gives overall NDCG |
| `ndcg_delta(a, b)` | Shows which queries improved/degraded between strategies | `ndcg_delta(graded_syns, graded_bm25)` |
| `vs_ideal(graded_df, judgments, products)` | Compares results against the ideal ordering | Useful for per-query debugging |

### 3.2 Notebook 1 -- Synonym Generation

#### Pydantic Models

The notebook defines three nested models to structure the LLM's synonym output:

```python
class Query(BaseModel):
    keywords: str = Field(..., description="The original search query keywords")

class SynonymMapping(BaseModel):
    phrase: str = Field(..., description="The original phrase from the query")
    synonyms: List[str] = Field(..., description="List of synonyms or equivalent phrases")

class QueryWithSynonyms(Query):
    synonyms: List[SynonymMapping] = Field(
        ..., description="Mapping of phrases to equivalent phrases or synonyms"
    )
```

When the LLM processes the query "suede couch", it returns something like:

```python
QueryWithSynonyms(
    keywords='suede couch',
    synonyms=[
        SynonymMapping(phrase='suede', synonyms=['microfiber', 'faux suede', 'soft fabric']),
        SynonymMapping(phrase='couch', synonyms=['sofa', 'settee', 'loveseat', 'divan'])
    ]
)
```

This is powerful because the LLM understands that "suede" in a furniture context relates to "microfiber" and "faux suede" -- domain knowledge that a simple synonym dictionary wouldn't have.

#### Inspecting the JSON Schema

Right after defining the Pydantic models, the notebook calls `model_json_schema()` to show what the LLM actually "sees":

```python
QueryWithSynonyms.model_json_schema()
```

This produces:

```python
{
  '$defs': {
    'SynonymMapping': {
      'description': 'Model for mapping phrases in the query to equivalent phrases or synonyms.',
      'properties': {
        'phrase': {'description': 'The original phrase from the query', 'type': 'string'},
        'synonyms': {'description': 'List of synonyms or equivalent phrases', 'type': 'array',
                     'items': {'type': 'string'}}
      },
      'required': ['phrase', 'synonyms'],
      'type': 'object'
    }
  },
  'description': 'Extended model for search queries that includes synonyms for keywords.\nInherits from the base Query model.',
  'properties': {
    'keywords': {'description': 'The original search query keywords sent in as input', 'type': 'string'},
    'synonyms': {'description': 'Mapping of phrases in the query to equivalent phrases or synonyms',
                 'items': {'$ref': '#/$defs/SynonymMapping'}, 'type': 'array'}
  },
  'required': ['keywords', 'synonyms'],
  'type': 'object'
}
```

This is important to understand: **the Pydantic model becomes a JSON Schema that is sent to the LLM alongside your prompt.** The LLM reads the schema -- including the `description` strings you wrote on each field -- and is constrained to produce output that conforms to it. The field descriptions act as instructions to the LLM. For example, when it sees `'description': 'The original phrase from the query'`, it knows the `phrase` field should contain a substring from the user's input, not an invented word.

#### Direct Enrichment -- Raw OpenAI API Call

Before introducing the `AutoEnricher` wrapper, the notebook shows the **raw OpenAI API call** so you can see what's happening under the hood:

```python
from cheat_at_search.data_dir import key_for_provider
from openai import OpenAI

openai_key = key_for_provider("openai")

client = OpenAI(api_key=openai_key)

prompts = []
prompts.append({"role": "system",
                "content": "You are a search query synonym generator for furniture e-commerce"})
prompts.append({"role": "user",
                "content": "Please generate synonyms for query: suede couch"})

response = client.responses.parse(
    model="gpt-4o",
    input=prompts,
    text_format=QueryWithSynonyms
)

response.output_parsed
```

This returns:

```python
QueryWithSynonyms(
    keywords='suede couch',
    synonyms=[
        SynonymMapping(phrase='suede', synonyms=['microfiber', 'faux suede', 'suede-like', 'soft fabric']),
        SynonymMapping(phrase='couch', synonyms=['sofa', 'settee', 'loveseat', 'divan'])
    ]
)
```

**Breaking this down step by step:**

1. **`key_for_provider("openai")`** -- Retrieves the OpenAI API key from the data directory (previously saved to Google Drive or entered interactively). This keeps secrets out of the notebook code.

2. **`OpenAI(api_key=openai_key)`** -- Creates an OpenAI client. This is the official `openai` Python package.

3. **Building the `prompts` list** -- OpenAI's chat API expects a list of message dicts, each with a `role` and `content`:
   - `"system"` messages set the LLM's persona and behavioral context. Here, we tell it to act as a "search query synonym generator for furniture e-commerce" -- this primes it to generate furniture-relevant synonyms, not general-purpose ones.
   - `"user"` messages contain the actual request. This is the prompt the LLM responds to.

4. **`client.responses.parse(..., text_format=QueryWithSynonyms)`** -- This is OpenAI's **structured output** API. The `text_format` parameter accepts a Pydantic class. OpenAI sends the JSON schema to the model and guarantees the response will parse into that class. No regex parsing, no "please return valid JSON" begging.

5. **`response.output_parsed`** -- The parsed result, already an instance of `QueryWithSynonyms`. You can immediately access `response.output_parsed.synonyms[0].phrase` etc.

This cell demonstrates the concept. The next cells wrap this pattern into a reusable component.

#### Synonym Generation Code -- The AutoEnricher Wrapper

The notebook then introduces `AutoEnricher`, which wraps the raw API call pattern shown above into a reusable, configurable object:

```python
syn_enricher = AutoEnricher(
    model="openai/gpt-5-nano",
    system_prompt="You are a helpful AI assistant extracting synonyms from queries.",
    response_model=QueryWithSynonyms
)
```

**What each parameter does:**

- **`model="openai/gpt-5-nano"`** -- Specifies which LLM to call. The `"openai/"` prefix tells AutoEnricher to use the OpenAI provider. `"gpt-5-nano"` is a smaller, cheaper, faster model. The notebook deliberately picks a cheap model here because synonym generation will be called once per query (480 times during evaluation), and each call costs money. Using a larger model like `gpt-4o` would be more accurate but much more expensive.

- **`system_prompt="You are a helpful AI assistant extracting synonyms from queries."`** -- This is the system message that gets sent with every request. Compare this to the raw API call above where we manually built `{"role": "system", "content": "..."}`. AutoEnricher handles that plumbing for you.

- **`response_model=QueryWithSynonyms`** -- The Pydantic model that defines the output structure. AutoEnricher passes this to the API as the `text_format` (or equivalent depending on the provider), ensuring the LLM returns a valid `QueryWithSynonyms` instance.

**The prompt template function:**

```python
def get_prompt(query: str):
    prompt = f"""
        Extract synonyms from the following query that will help us find relevant products for the query.

        {query}
    """
    return prompt

print(get_prompt("rack glass"))
# Output:
#     Extract synonyms from the following query that will help us find relevant
#     products for the query.
#
#     rack glass
```

This is a simple f-string template. The key design decision is in the wording: "that will help us find relevant **products** for the query." This steers the LLM toward e-commerce-relevant synonyms. Without this guidance, the LLM might generate general English synonyms that don't help with product search (e.g., "glass" -> "spectacles" instead of "glass" -> "stemware" or "glassware").

The notebook prints the prompt to show exactly what gets sent to the LLM -- a useful debugging technique.

**The complete enrichment pipeline:**

```python
def query_to_syn(query: str):
    return syn_enricher.enrich(get_prompt(query))

query_to_syn("foldout blue ugly love seat")
```

Returns:

```python
QueryWithSynonyms(
    keywords='foldout blue ugly love seat',
    synonyms=[
        SynonymMapping(phrase='foldout',    synonyms=['pull-out', 'sofa bed', 'futon bed', 'convertible sofa']),
        SynonymMapping(phrase='blue',       synonyms=['navy blue', 'azure', 'cobalt', 'blue color', 'blue shade', 'blue upholstery']),
        SynonymMapping(phrase='ugly',       synonyms=['unattractive', 'unsightly', 'plain', 'hideous', 'unappealing', 'dull']),
        SynonymMapping(phrase='love seat',  synonyms=['loveseat', 'two-seater sofa', 'smaller sofa', 'duo-seater'])
    ]
)
```

**What `query_to_syn` does, step by step:**

1. Takes the raw query string `"foldout blue ugly love seat"`
2. Wraps it in the prompt template via `get_prompt()` -- adding the instruction context
3. Sends the prompt + system message + JSON schema to the LLM via `syn_enricher.enrich()`
4. The LLM analyzes the query, splits it into meaningful phrases, and generates synonyms for each
5. Returns a validated `QueryWithSynonyms` Pydantic object

**Observations about the LLM's output:**

- The LLM intelligently groups "love seat" as a single phrase (two words that form one concept), rather than treating "love" and "seat" separately.
- It understands "foldout" in a furniture context means a convertible/sleeper style.
- It generates both single words ("azure") and multi-word phrases ("two-seater sofa").
- Some synonyms are arguably not helpful for search -- "unattractive" as a synonym for "ugly" is unlikely to match any product description. This is one of the weaknesses of naive synonym expansion.

This `query_to_syn` function is what gets passed into the `SynonymSearch` strategy as its `synonym_generator` parameter. Every time `SynonymSearch.search()` is called for a query, it calls `query_to_syn` to get synonyms, then uses them to boost BM25 scores.

#### SynonymSearch Strategy

```python
class SynonymSearch(SearchStrategy):
    def __init__(self, products, synonym_generator,
                 name_boost=9.3, description_boost=4.1):
        super().__init__(products)
        self.index = products
        self.index['product_name_snowball'] = SearchArray.index(
            products['product_name'], snowball_tokenizer)
        self.index['product_description_snowball'] = SearchArray.index(
            products['product_description'], snowball_tokenizer)
        self.query_to_syn = synonym_generator

    def search(self, query, k=10):
        tokenized = snowball_tokenizer(query)
        bm25_scores = np.zeros(len(self.index))

        # Step 1: Base BM25 scoring on original query tokens
        for token in tokenized:
            bm25_scores += self.name_boost * self.index['product_name_snowball'].array.score(token)
            bm25_scores += self.description_boost * self.index['product_description_snowball'].array.score(token)

        # Step 2: Generate synonyms via LLM
        synonyms = self.query_to_syn(query)

        # Step 3: Add BM25 scores for synonym phrases
        for mapping in synonyms.synonyms:
            for phrase in mapping.synonyms:
                tokenized = snowball_tokenizer(phrase)
                bm25_scores += self.index['product_name_snowball'].array.score(tokenized)
                bm25_scores += self.index['product_description_snowball'].array.score(tokenized)

        # Step 4: Return top-k
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores
```

**How it works step by step for query "suede couch":**

1. Tokenize "suede couch" with Snowball stemmer: `["sued", "couch"]`
2. Score every product on BM25 match against "sued" and "couch" in name (weighted 9.3x) and description (weighted 4.1x)
3. Call LLM to get synonyms: "microfiber", "faux suede", "sofa", "settee", etc.
4. For each synonym phrase, also compute BM25 scores and add them
5. Products matching both original terms AND synonyms get higher scores
6. Return top 10 by total score

**Key observation:** The synonym scores are added on top of the base BM25 scores with equal weight (1.0x). The name and description boosts (9.3x and 4.1x) only apply to the original query terms. This means synonyms contribute less to the total score than the original query.

#### Running the Strategy and Getting Graded Results

Once the `SynonymSearch` strategy is built, the notebook runs it against every query in the WANDS dataset:

```python
# for each query
#   results = syns.search(query)
#   -- Give each result a 'grade'
#   --- Compute DCG
graded_syns = run_strategy(syns, judgments)
graded_syns
```

`run_strategy` does the following behind the scenes:
1. Iterates over all 480 WANDS queries
2. For each query, calls `syns.search(query)` to get the top 10 results
3. Looks up each result in `judgments` to find its relevance grade (0, 1, or 2)
4. Computes DCG (Discounted Cumulative Gain) and NDCG for each query
5. Returns a single DataFrame (`graded_syns`) with 4,800 rows (480 queries x 10 results)

The returned DataFrame has columns including `product_name`, `score`, `query`, `rank`, `grade`, `dcg`, and `ndcg`. Each row is one search result with its relevance grade attached.

#### Analyzing Results: Mean NDCG Comparison

```python
ndcgs(graded_bm25).mean(), ndcgs(graded_syns).mean()
# Output: (0.5411, 0.5507)
```

This compares the average NDCG across all 480 queries for the two strategies:

| Strategy | Mean NDCG |
|----------|-----------|
| BM25 baseline | 0.5411 |
| BM25 + LLM synonyms | 0.5507 |

The synonym strategy improves NDCG by about **+0.01** (roughly a 1.8% relative improvement). This is a modest gain -- the synonyms help a little on average, but the improvement is not dramatic.

`ndcgs()` extracts the per-query NDCG values from a graded DataFrame (since every row for a given query has the same NDCG, it effectively deduplicates to one NDCG per query). Calling `.mean()` gives the single-number summary.

#### Win / Loss Analysis with `ndcg_delta`

```python
ndcg_delta(graded_syns, graded_bm25)
```

This shows the **per-query NDCG difference** (synonyms minus BM25), sorted from biggest win to biggest loss:

| Query | NDCG Delta |
|-------|------------|
| midcentury tv unit | +0.666 |
| cover set for outdoor furniture | +0.453 |
| 7qt slow cooker | +0.446 |
| bathroom vanity knobs | +0.410 |
| desk for kids | +0.316 |
| ... | ... |
| closet storage with zipper | -0.132 |
| adjustable height artist stool | -0.152 |
| bathroom single faucet | -0.182 |
| king size bed | -0.288 |
| capricorn chest | -0.293 |

**What this reveals:**

- There are **some massive wins** -- "midcentury tv unit" improved by +0.67 NDCG, meaning synonyms like "mid-century modern" or "tv stand" helped find products that pure keyword matching missed.
- There are also **significant losses** -- "capricorn chest" dropped by -0.29. The LLM likely generated irrelevant synonyms that polluted the results.
- The table shows 180 rows (queries where NDCG changed). The remaining 300 queries had identical NDCG with both strategies, meaning synonyms neither helped nor hurt.
- The **high variance** (big wins AND big losses) makes this a risky change in production. A search team would want consistent improvement, not a coin flip.

#### Examining a Single Query: "seat cushions desk"

The notebook picks a specific query to examine side-by-side:

```python
QUERY = "seat cushions desk"
```

**BM25 baseline results:**

```python
graded_bm25[graded_bm25['query'] == QUERY][['rank', 'product_name', 'product_description', 'grade']]
```

| Rank | Product Name | Grade |
|------|-------------|-------|
| 1 | ergonomic memory foam seat cushion | 2 (relevant) |
| 2 | chiavari seat cushion | 1 (partial) |
| 3 | deluxe seat cushion | 1 (partial) |
| 4 | deep outdoor seat cushion | 1 (partial) |
| 5 | outdoor seat cushion | 1 (partial) |
| 6 | indoor seat cushion | 1 (partial) |
| 7 | indoor/outdoor seat cushion | 1 (partial) |
| 8 | outdoor seat/back cushion | 1 (partial) |
| 9 | outdoor sunbrella seat cushion | 0 (not relevant) |
| 10 | gel seat cushion | 2 (relevant) |

BM25 finds products matching "seat" and "cushion" but ignores the "desk" intent. Most results are outdoor/indoor general cushions (grade 1), not the desk/office-specific cushions the user wants (grade 2). Only rank 1 and rank 10 are fully relevant.

**Synonym strategy results:**

```python
graded_syns[graded_syns['query'] == QUERY][['rank', 'product_name', 'product_description', 'grade']]
```

The synonym results show a similar mix. The LLM generated:

```python
query_to_syn(QUERY)
# QueryWithSynonyms(
#     keywords='seat cushions desk',
#     synonyms=[
#         SynonymMapping(
#             phrase='seat cushions desk',
#             synonyms=['seat cushions', 'desk cushions', 'chair cushions',
#                       'cushions for seat', 'office chair cushions',
#                       'desk chair cushions', 'seat pads', 'desk pad cushions']
#         )
#     ]
# )
```

The synonyms include "office chair cushions" and "desk chair cushions" which are good -- they capture the desk/office intent. However, they also include generic terms like "seat cushions" and "cushions for seat" that don't add specificity. The net result for this query: the synonyms help surface some desk-specific cushions but also bring in more general matches.

#### Comparing Against the Ideal with `vs_ideal`

```python
against_ideal = vs_ideal(graded_syns, judgments, products)
against_ideal[against_ideal['query'] == QUERY]
```

This produces a side-by-side comparison for each rank position:

| Rank | Ideal Product (title_ideal) | Actual Product (title_actual) | Grade Actual |
|------|-----------------------------|-------------------------------|-------------|
| 1 | amamedic mesh seat cushion | ergonomic memory foam seat cushion | 2 |
| 2 | yeslife seat cushion | deep outdoor seat cushion | 1 |
| 3 | office chair seat cushion | outdoor seat cushion | 1 |
| 4 | pressure relief non-slip orthopedic seat cushion | indoor seat cushion | 1 |
| 5 | ergonomic gel seat cushion | chiavari seat cushion | 1 |
| ... | ... | ... | ... |

The `title_ideal` column shows what a perfect ranking would have returned. The `title_actual` column shows what our synonym strategy returned. You can see the ideal ranking is dominated by ergonomic/office cushions (grade 2), while the actual ranking has many generic seat cushions (grade 1).

The NDCG for this query is **0.615** -- meaning the synonym strategy retrieved about 61.5% of the ideal ranking quality. There is room for improvement, which motivates the category-based approaches explored in Notebooks 2 and 2a.

**Key takeaway from the evaluation:** Synonym expansion provides modest overall improvement (+1.8% mean NDCG) but with high variance. Some queries benefit dramatically (synonyms fill vocabulary gaps), while others degrade (irrelevant synonyms pollute results). This motivates exploring query categorization as a more structured, less noisy approach to improving search.

### 3.3 Notebook 2 -- Perfect Categorization (Cheating with Ground Truth)

#### Building the Ground Truth

The notebook constructs a "cheating" classifier that always returns the correct answer. It does this by analyzing the labeled data:

```python
def get_top_category(column, no_fit_label, cutoff=0.8):
    # 1. Get only the most relevant products (grade == 2) for each query
    top_products = labeled_query_products[labeled_query_products['grade'] == 2]

    # 2. Count how often each category appears for each query
    categories_per_query_ideal = top_products.groupby('query')[column].value_counts()

    # 3. Convert to proportions (what % of relevant products are in each category?)
    top_cat_proportion = ...  # percentage calculation

    # 4. Keep only categories that represent >80% of relevant products
    top_cat_proportion = top_cat_proportion[top_cat_proportion['count'] > cutoff]

    # 5. Label queries without a dominant category as "No Category Fits"
    ...
    return ground_truth_cat
```

**Example:** For the query "turquoise pillows", if 96% of the relevant products are in "Decorative Pillows & Blankets", then that becomes the ground truth subcategory.

For the query "zen", relevant products might span "Decor & Pillows" (40%), "Home Improvement" (30%), and "Outdoor" (30%) -- no single category exceeds 80%, so it gets "No Category Fits".

#### The Cheating Classifier

```python
def categorized(query):
    category = "No Category Fits"
    sub_category = "No SubCategory Fits"
    if query in ground_truth_cat['query'].values:
        cat_at_query = ground_truth_cat[ground_truth_cat['query'] == query]['category']
        category = cat_at_query.values[0].strip()
    if query in ground_truth_sub_cat['query'].values:
        sub_cat_at_query = ground_truth_sub_cat[ground_truth_sub_cat['query'] == query]['sub_category']
        sub_category = sub_cat_at_query.values[0].strip()
    return QueryCategory(keywords=query, category=category, sub_category=sub_category)
```

This simply looks up the answer from the ground truth. It is "cheating" because in production you would not have these labels -- you would need an LLM or classifier to predict them. The point is to establish **the theoretical upper bound**: if categorization were perfect, how much would it help?

#### Pydantic Models with Literal Types

```python
Categories = Literal['Furniture', 'Home Improvement', 'Decor & Pillows', 'Outdoor',
                     'Storage & Organization', 'Lighting', 'Rugs', 'Bed & Bath',
                     'Kitchen & Tabletop', 'Baby & Kids', ..., 'No Category Fits']

SubCategories = Literal['Bedroom Furniture', 'Small Kitchen Appliances',
                        'Living Room Furniture', ..., 'No SubCategory Fits']

class QueryCategory(Query):
    category: Categories = Field(description="Category of the product")
    sub_category: SubCategories = Field(description="Sub-category of the product")
```

The `Literal` type constrains possible values. When used with `AutoEnricher`, this forces the LLM to pick from the predefined list rather than inventing categories. This is important because the category must match values that exist in the product catalog.

#### CategorySearch Strategy

```python
class CategorySearch(SearchStrategy):
    def __init__(self, products, query_to_cat,
                 name_boost=9.3, description_boost=4.1,
                 category_boost=100, sub_category_boost=50):
        super().__init__(products)
        # Index text fields (same as synonym search)
        self.index['product_name_snowball'] = SearchArray.index(...)
        self.index['product_description_snowball'] = SearchArray.index(...)
        # ALSO index category fields
        self.index['category_snowball'] = SearchArray.index(
            products['category'], snowball_tokenizer)
        self.index['subcategory_snowball'] = SearchArray.index(
            products['subcategory'], snowball_tokenizer)

    def search(self, query, k=10):
        bm25_scores = np.zeros(len(self.index))
        structured = self.query_to_cat(query)  # Get category prediction
        tokenized = snowball_tokenizer(query)

        # Base BM25 scoring (same as before)
        for token in tokenized:
            bm25_scores += self.index['product_name_snowball'].array.score(token) * self.name_boost
            bm25_scores += self.index['product_description_snowball'].array.score(token) * self.description_boost

        # Category boosting: add a CONSTANT score to all products in the predicted subcategory
        if structured.sub_category != "No SubCategory Fits":
            tokenized_subcategory = snowball_tokenizer(structured.sub_category)
            subcategory_match = self.index['subcategory_snowball'].array.score(tokenized_subcategory) > 0
            bm25_scores[subcategory_match] += self.sub_category_boost  # +50

        # Same for top-level category
        if structured.category != "No Category Fits":
            tokenized_category = snowball_tokenizer(structured.category)
            category_match = self.index['category_snowball'].array.score(tokenized_category) > 0
            bm25_scores[category_match] += self.category_boost  # +100

        top_k = np.argsort(-bm25_scores)[:k]
        return top_k, bm25_scores[top_k]
```

**How the category boost works for query "turquoise pillows":**

1. Base BM25: products with "turquois" and "pillow" in their name/description get scored
2. Ground truth says category = "Decor & Pillows", subcategory = "Decorative Pillows & Blankets"
3. Every product in "Decorative Pillows & Blankets" gets +50 points
4. Every product in "Decor & Pillows" gets +100 points
5. A pillow product in the right category might score 80 (BM25) + 100 (category) + 50 (subcategory) = 230
6. A random product that happens to mention "turquoise" in its description might score 15 (BM25) + 0 = 15

The constant boost effectively filters results to the correct department, dramatically improving relevance.

#### Precision Evaluation

```python
def prec_cat(ground_truth, column, no_fit_label, categorized, N=500):
    hits = []
    misses = []
    for _, row in ground_truth.sample(frac=1).iterrows():
        query = row['query']
        expected_category = row[column]
        cat = categorized(query)
        pred = get_pred(cat, column)
        if pred == expected_category.strip():
            hits.append(...)
        else:
            misses.append(...)
    return len(hits) / (len(hits) + len(misses))
```

This function measures how often the classifier correctly predicts the ground truth category. When "cheating" (using ground truth directly), precision is 100%. This establishes the ceiling for what an LLM classifier could achieve.

### 3.4 Notebook 2a -- Query Categories as a List

#### Fully Qualified Classifications

The key innovation in Notebook 2a is that instead of predicting a single category, the LLM predicts a **list of fully-qualified category paths**:

```python
FullyQualifiedClassifications = Literal[
    'Furniture / Bedroom Furniture / Beds & Headboards / Beds',
    'Furniture / Living Room Furniture / Chairs & Seating / Accent Chairs',
    'Rugs / Area Rugs',
    'Furniture / Office Furniture / Desks',
    ...  # ~200+ paths
    'No Classification Fits'
]
```

This is much more specific than just "Furniture" or "Living Room Furniture". It goes all the way down the taxonomy tree.

#### QueryClassification Model

```python
class QueryClassification(Query):
    classifications: list[FullyQualifiedClassifications] = Field(
        description="A possible classification for the product."
    )

    @property
    def categories(self):
        return set([c.split(" / ")[0] for c in self.classifications])

    @property
    def sub_categories(self):
        return set([c.split(" / ")[1] for c in self.classifications if len(c.split(" / ")) > 1])
```

The `categories` and `sub_categories` properties extract top-level and second-level categories from the full paths by splitting on " / ". For example, from "Furniture / Living Room Furniture / Sofas", it extracts category = "Furniture" and sub_category = "Living Room Furniture".

#### LLM Classification

```python
enricher = AutoEnricher(
    model="openai/gpt-4o",
    system_prompt="You are a helpful furniture shopping agent...",
    response_model=QueryClassification
)

def fully_classified(query):
    prompt = f"""
    Your task is to search with a structured query against a furniture product catalog.
    Here is the users request: {query}
    Return the best classifications for this user's query.
    Try to pick as diverse a set of possible to ensure the customer finds what they need.
    Return an empty list if no classification fits, or its too ambiguous.
    """
    classification = enricher.enrich(prompt)
    return classification
```

**Example:** `fully_classified("sofa loveseat")` returns:

```python
QueryClassification(
    keywords='sofa loveseat',
    classifications=[
        'Furniture / Living Room Furniture / Sofas',
        'Reception Area / Reception Seating / Reception Sofas & Loveseats',
        'Reception Area / Reception Seating / Office Sofas & Loveseats'
    ]
)
```

The prompt instructs the LLM to pick **diverse** classifications. This means it may suggest "Furniture" AND "Reception Area" categories, covering both residential and commercial furniture.

#### List-Based Ground Truth

Notebook 2a also builds a different ground truth with a lower threshold:

```python
# Previous: >80% of relevant products must be in this category (strict, single)
ground_truth_cat = get_top_categories('category', 'No Category Fits', cutoff=0.8)

# New: >5% of relevant products is enough to include (relaxed, multi-label)
ground_truth_cat_list = get_top_categories('category', 'No Category Fits', cutoff=0.05)

# Group into lists per query
ground_truth_cat_list = ground_truth_cat_list.groupby('query').agg({'category': list})
```

**Example:** The query "zen" might have:
- Previous ground truth: "No Category Fits" (no category > 80%)
- New ground truth: `['Decor & Pillows', 'Home Improvement', 'Outdoor']` (each > 5%)

#### Jaccard Similarity Evaluation

Since predictions are now lists (not single values), the notebook uses different metrics:

```python
def jaccard_sim(ground_truth, column, no_fit_label, classifier_fn):
    jaccard_sum = 0
    recall_sum = 0
    for _, row in ground_truth.iterrows():
        query = row['query']
        expected_category = set(row[column])
        preds = set(get_preds(classifier_fn(query), column))

        # Jaccard: |intersection| / |union|
        jaccard = len(preds.intersection(expected_category)) / len(preds.union(expected_category))

        # Recall: |intersection| / |expected|
        recall = len(preds.intersection(expected_category)) / len(expected_category)
    ...
```

**Jaccard similarity** measures how much two sets overlap. If the LLM predicts `{'Furniture', 'Outdoor'}` and the ground truth is `{'Furniture'}`, the Jaccard is 1/2 = 0.5 (they share 1 element, but the union has 2).

**Recall** measures what fraction of the ground truth categories were found. In the same example: 1/1 = 1.0 (the ground truth category "Furniture" was found).

---

## Phase 4 - Python Pattern Explanations

### 4.1 Pydantic BaseModel and Field

**What it is:** Pydantic is a data validation library. `BaseModel` creates classes with automatic validation, serialization, and schema generation.

**Why it's used here:** OpenAI's structured output API accepts a Pydantic model as a schema. The LLM is then forced to return JSON conforming to that schema -- no more parsing fragile text output.

```python
from pydantic import BaseModel, Field
from typing import List

class SynonymMapping(BaseModel):
    phrase: str = Field(..., description="The original phrase from the query")
    synonyms: List[str] = Field(..., description="List of synonyms")

# Usage
mapping = SynonymMapping(phrase="couch", synonyms=["sofa", "settee"])
print(mapping.phrase)         # "couch"
print(mapping.model_dump())   # {'phrase': 'couch', 'synonyms': ['sofa', 'settee']}
```

The `Field(...)` function:
- `...` (Ellipsis) means the field is **required** (no default value)
- `description=` provides documentation that the LLM reads to understand what to generate

The `model_json_schema()` method generates a JSON Schema that gets sent to the OpenAI API:

```python
QueryWithSynonyms.model_json_schema()
# Returns a dict describing the expected JSON structure
```

### 4.2 Class Inheritance

**What it is:** Creating a class that inherits attributes and methods from a parent class.

**Why it's used here:** Both `QueryWithSynonyms` and `QueryCategory` extend `Query` to avoid repeating the `keywords` field:

```python
class Query(BaseModel):
    keywords: str = Field(..., description="The original search query keywords")

class QueryWithSynonyms(Query):
    # Inherits 'keywords' from Query, adds 'synonyms'
    synonyms: List[SynonymMapping] = Field(...)

class QueryCategory(Query):
    # Inherits 'keywords' from Query, adds 'category' and 'sub_category'
    category: Categories = Field(...)
    sub_category: SubCategories = Field(...)
```

`SearchStrategy` is also used as a base class:

```python
class SynonymSearch(SearchStrategy):
    def __init__(self, products, synonym_generator, ...):
        super().__init__(products)  # Call parent's __init__
        # ... add our own setup
```

`super().__init__(products)` calls the parent class's `__init__` method, ensuring any base setup is done before the subclass adds its own.

### 4.3 typing.Literal -- Constrained String Types

**What it is:** `Literal` restricts a type to specific allowed values, like an enum but for type hints.

**Why it's used here:** To constrain LLM outputs to valid categories from the product catalog.

```python
from typing import Literal

Categories = Literal[
    'Furniture',
    'Home Improvement',
    'Decor & Pillows',
    ...
    'No Category Fits'
]

class QueryCategory(Query):
    category: Categories = Field(description="Category of the product")
```

When this schema is sent to the LLM, the LLM can ONLY return one of the listed strings. If the product catalog has 21 categories, the LLM must pick from exactly those 21. This prevents hallucinated categories like "Electronics" that don't exist in the catalog.

**Accessing Literal values at runtime:**

```python
from typing import get_args

classifications_list = get_args(FullyQualifiedClassifications)
# Returns a tuple of all ~200 string values in the Literal
```

`get_args()` extracts the arguments from a generic type at runtime, useful for building sets of known values.

### 4.4 @property Decorator

**What it is:** Makes a method accessible like an attribute (without parentheses).

**Why it's used here:** To derive computed values from stored data:

```python
class QueryCategory(Query):
    category: Categories
    sub_category: SubCategories

    @property
    def classification(self) -> str:
        return f"{self.category} / {self.sub_category}"

# Usage:
cat = QueryCategory(keywords="pillows", category="Decor & Pillows",
                     sub_category="Decorative Pillows & Blankets")
print(cat.classification)  # "Decor & Pillows / Decorative Pillows & Blankets"
# Note: no parentheses -- it looks like accessing an attribute, not calling a method
```

In Notebook 2a, properties extract category levels from full paths:

```python
class QueryClassification(Query):
    classifications: list[FullyQualifiedClassifications]

    @property
    def categories(self):
        return set([c.split(" / ")[0] for c in self.classifications])

    @property
    def sub_categories(self):
        return set([c.split(" / ")[1] for c in self.classifications if len(c.split(" / ")) > 1])
```

### 4.5 List Comprehensions and Set Comprehensions

**What they are:** Compact syntax for creating lists/sets from iterables with optional filtering.

```python
# List comprehension: [expression for item in iterable if condition]
categories = [c.split(" / ")[0] for c in self.classifications]
# Equivalent to:
# categories = []
# for c in self.classifications:
#     categories.append(c.split(" / ")[0])

# Set comprehension: same but with {} and produces unique values
known_categories = set([c.split(" / ")[0].strip() for c in classifications_list])

# With filtering:
sub_categories = set([
    c.split(" / ")[1]
    for c in self.classifications
    if len(c.split(" / ")) > 1  # Only include paths with at least 2 levels
])
```

The `if len(c.split(" / ")) > 1` guard prevents an IndexError when a classification string has no subcategory (e.g., "No Classification Fits" has no "/" separator).

### 4.6 Lambda Functions

**What they are:** Anonymous, one-line functions.

**Why they're used here:** For quick transformations in pandas operations:

```python
# Split category hierarchy and extract the first element
products['category'] = cat_split.apply(
    lambda x: x[0].strip() if len(x) > 0 else ""
)

# Filter out empty/no-fit categories from lists
ground_truth_cat_list['category'] = ground_truth_cat_list['category'].apply(
    lambda x: [y.strip() for y in x if y and y != 'No Category Fits']
)
```

The `lambda x:` creates an unnamed function that takes `x` and returns the expression. The `.apply()` method runs this function on every element in the pandas Series.

### 4.7 NumPy Array Operations

**What they are:** Operations on entire arrays at once (vectorized), avoiding slow Python loops.

**Why they're used here:** For efficient scoring of 43,000 products simultaneously:

```python
bm25_scores = np.zeros(len(self.index))  # Array of 43,000 zeros

# Add BM25 scores for ALL products at once (no loop over products)
bm25_scores += self.name_boost * self.index['product_name_snowball'].array.score(token)

# Boolean masking: get a True/False array for all products
subcategory_match = self.index['subcategory_snowball'].array.score(tokenized_subcategory) > 0

# Apply constant boost only where mask is True
bm25_scores[subcategory_match] += self.sub_category_boost

# Sort all 43,000 scores in descending order and take top 10
top_k = np.argsort(-bm25_scores)[:k]
```

Key concepts:
- `np.zeros(n)` creates an array of n zeros
- `array + scalar` adds the scalar to every element
- `array > 0` creates a boolean mask (True where condition holds)
- `array[bool_mask]` selects only elements where mask is True
- `np.argsort(-array)[:k]` returns indices of the k largest values (negate for descending)

### 4.8 f-strings (Formatted String Literals)

**What they are:** Strings prefixed with `f` that allow embedding expressions inside `{}`.

**Why they're used here:** For building prompts dynamically:

```python
def get_prompt(query: str):
    prompt = f"""
        Extract synonyms from the following query that will help us
        find relevant products for the query.

        {query}
    """
    return prompt

# get_prompt("suede couch") produces:
# "Extract synonyms from the following query...
#  suede couch"
```

The triple-quoted f-string (`f"""..."""`) allows multi-line strings with embedded variables. This is the standard way to build LLM prompts that include user input.

### 4.9 Pandas GroupBy and Aggregation

**What it is:** Split-apply-combine operations on DataFrames.

**Why it's used here:** For computing category proportions from labeled data:

```python
# Count how many products of each category exist for each query
categories_per_query = top_products.groupby('query')['category'].value_counts()

# Get proportion of each category per query
top_cat_proportion = (
    categories_per_query.groupby(['query', 'category']).sum()
    / categories_per_query.groupby('query').sum()
)

# Aggregate categories into lists per query
ground_truth_cat_list = ground_truth_cat_list.groupby('query').agg({'category': list})
```

The `.groupby('query')` splits the DataFrame by unique query values. Then `.value_counts()` counts occurrences within each group. Dividing group-level counts by query-level totals gives proportions.

The `.agg({'category': list})` aggregation collects all category values for each query into a Python list, transforming multiple rows into a single row with a list column.

### 4.10 Set Operations for Evaluation

**What they are:** Mathematical set operations (intersection, union) on Python sets.

```python
preds = {'Furniture', 'Outdoor'}
expected = {'Furniture', 'Home Improvement'}

# Intersection: elements in BOTH sets
preds.intersection(expected)  # {'Furniture'}

# Union: elements in EITHER set
preds.union(expected)  # {'Furniture', 'Outdoor', 'Home Improvement'}

# Jaccard similarity
jaccard = len(preds.intersection(expected)) / len(preds.union(expected))
# = 1 / 3 = 0.333

# Recall
recall = len(preds.intersection(expected)) / len(expected)
# = 1 / 2 = 0.5
```

These operations are used in Notebook 2a to evaluate multi-label classification. Sets are ideal here because the order of categories doesn't matter -- we only care about which categories are present.

---

## Summary: What Did We Learn?

The three notebooks teach a progression of search improvement techniques:

1. **Synonyms (Notebook 1):** The most intuitive approach -- if users say "couch" but products say "sofa", add synonyms. However, this can introduce noise (false matches) and the improvement over BM25 may be modest.

2. **Perfect Categorization (Notebook 2):** Establishes the ceiling. If we could perfectly predict that "turquoise pillows" should show results from "Decorative Pillows & Blankets", search quality jumps dramatically. This motivates investing in query classification.

3. **LLM Classification with Lists (Notebook 2a):** The practical approach. An LLM predicts multiple possible category paths. Some queries ("sofa loveseat") map cleanly to one category; others ("zen") could span many. The list-based approach handles ambiguity gracefully.

The key takeaway: **query understanding (knowing WHAT the user wants) can be more impactful than query expansion (adding more words to search for)**. A correct category prediction acts like a strong filter, while synonyms merely add more potential matches.
