# CollabNBs Notebook Analysis

A comprehensive educational guide to the `cheat-at-search` notebooks and library: what they do, how they work, and the Python patterns they use.

---

## Phase 1 -- High-Level Overview (Search Functionality Perspective)

### What Problem Does This Solve?

These notebooks teach how to combine **traditional keyword search** with **Large Language Models (LLMs)** to build a product search assistant. The core idea: instead of relying solely on a search engine *or* solely on an LLM, you make them collaborate.

The specific domain is **e-commerce furniture search** using the Wayfair WANDS (Wayfair ANnotated DataSet) -- a real dataset of ~43,000 furniture products with human-annotated relevance judgments.

### The Three Notebooks

| Notebook | Purpose | Key Concept |
|---|---|---|
| `1_Cheat_at_Search_Basic_Chat_Loop.ipynb` | Teaches the fundamentals of interacting with the OpenAI API: sending messages, getting structured outputs, and building a chat loop. | **Chat loop + Structured Outputs** |
| `1a_RAG_Wayfair_Dataset.ipynb` | Builds a complete Retrieval-Augmented Generation (RAG) system: the LLM generates search queries, a BM25 engine retrieves products, and the LLM summarizes results for the user. | **Classic RAG pattern** |
| `1b_Agentic_Search_Loop_Wayfair_Dataset.ipynb` | Upgrades from manual RAG to an agentic loop: the LLM autonomously decides when to call the search tool, executes multiple search rounds, and self-corrects its queries. | **Agentic tool-calling loop** |

### What Is RAG?

Retrieval-Augmented Generation (RAG) is a pattern where:

1. A user asks a question in natural language
2. The system **retrieves** relevant documents from a search index
3. Those documents are **fed back to the LLM** as context
4. The LLM **generates** a response grounded in real data

Without RAG, the LLM would hallucinate product names. With RAG, every recommendation comes from the actual Wayfair catalog.

### The Search Pipeline at a Glance

```
User: "Find me a modern geometric couch"
  |
  v
[LLM generates search query] --> SearchRequest(search_query="modern geometric couch sofa...", category=["Furniture"])
  |
  v
[BM25 keyword search over 43K products] --> Top 10 results with titles, descriptions, scores
  |
  v
[LLM reads results + original question] --> "Here are 5 couches with geometric lines..."
  |
  v
User: "Show me more L-shaped ones"
  |
  v
[Cycle repeats with conversation context]
```

---

## Phase 2 -- Architecture Analysis

### 2.1 Overall Code Organization

```
cheat-at-search/
├── CollabNBs/                          # Colab notebooks (the teaching material)
│   ├── 1_Cheat_at_Search_Basic_Chat_Loop.ipynb
│   └── 1a_RAG_Wayfair_Dataset.ipynb
│
├── cheat_at_search/                    # The Python library
│   ├── data_dir.py                     # Data management: paths, downloads, API keys
│   ├── wands_data.py                   # WANDS dataset loading (products, queries, labels)
│   ├── tokenizers.py                   # Text tokenization (Snowball stemming)
│   ├── search.py                       # Search evaluation orchestration (NDCG scoring)
│   ├── eval.py                         # DCG/NDCG grading of search results
│   ├── cache.py                        # Persistent LRU cache (disk-backed)
│   ├── logger.py                       # Centralized logging configuration
│   │
│   ├── strategy/                       # Search strategy implementations
│   │   ├── strategy.py                 #   Base class: SearchStrategy
│   │   ├── bm25.py                     #   BM25 keyword search
│   │   └── best.py                     #   Oracle/ceiling strategy using human labels
│   │
│   ├── model/                          # Pydantic data models for LLM structured output
│   │   ├── product.py                  #   Product, EnrichedProduct, ProductCategory
│   │   ├── query.py                    #   Query, StructuredQuery, QueryCategory
│   │   └── category_list.py            #   Literal type enums for categories
│   │
│   ├── enrich/                         # LLM enrichment pipeline
│   │   ├── enrich_client.py            #   Abstract base: EnrichClient
│   │   ├── openai_enrich_client.py     #   OpenAI implementation
│   │   ├── google_enrich_client.py     #   Google Gemini implementation
│   │   ├── cached_enrich_client.py     #   Caching wrapper
│   │   └── enrich.py                   #   AutoEnricher + ProductEnricher orchestrators
│   │
│   └── agent/                          # LLM agent with tool use
│       ├── search_client.py            #   Abstract Agent + SearchResult/SearchResults models
│       ├── openai_agent.py             #   OpenAI agent implementation (tool-calling loop)
│       └── pydantize.py                #   Converts Python functions into LLM-callable tools
│
└── pyproject.toml                      # Poetry project configuration
```

### 2.2 Architectural Layers

The codebase is organized into four layers. Each layer only depends on layers below it.

```
┌─────────────────────────────────────────────────┐
│  NOTEBOOKS (CollabNBs/)                         │  <-- User-facing teaching material
│  Orchestrate the full pipeline                  │
├─────────────────────────────────────────────────┤
│  AGENT / ENRICH Layer                           │  <-- LLM integration
│  agent/openai_agent.py, enrich/enrich.py        │
│  Manages LLM conversations, tool calls, caching │
├─────────────────────────────────────────────────┤
│  STRATEGY / MODEL Layer                         │  <-- Search + data models
│  strategy/bm25.py, model/query.py               │
│  BM25 search, Pydantic schemas for structured   │
│  outputs, evaluation (NDCG)                     │
├─────────────────────────────────────────────────┤
│  DATA / INFRA Layer                             │  <-- Foundation
│  data_dir.py, wands_data.py, tokenizers.py      │
│  Dataset loading, tokenization, path management │
└─────────────────────────────────────────────────┘
```

### 2.3 Key Architectural Patterns

**Strategy Pattern** -- Search algorithms (`BM25Search`, `BestPossibleResults`) inherit from a common `SearchStrategy` base class. You can swap search strategies without changing evaluation code.

**Abstract Base Classes** -- `EnrichClient` (abstract) has concrete implementations for OpenAI and Google. `Agent` (abstract) is implemented by `OpenAIAgent`.

**Lazy Loading / Module-Level `__getattr__`** -- The `wands_data` module uses Python's module-level `__getattr__` to lazily load datasets only when first accessed. This avoids loading 43K products on import.

**Caching Decorator** -- `StoredLruCache` persists LLM responses to disk so repeated enrichment calls don't re-query the API.

---

## Library Usage in the Notebooks

This section traces every `cheat_at_search` import in all three notebooks, explains what each library function does in context, and identifies how the library eliminates boilerplate so the notebooks can focus on teaching.

### Inventory of Library Imports

#### Notebook 1: Basic Chat Loop

| Cell | Import | Function Called | Purpose in Notebook |
|------|--------|----------------|---------------------|
| 1 | `from cheat_at_search.data_dir import mount` | `mount(use_gdrive=True)` | Set up persistent data storage on Google Drive |
| 2 | `from cheat_at_search.data_dir import key_for_provider` | `key_for_provider("openai")` | Obtain (or prompt for) the OpenAI API key |

Notebook 1 uses only **two library functions**, both from `data_dir`. The rest of the notebook is pure OpenAI API calls and Pydantic model definitions written inline.

#### Notebook 1a: RAG with Wayfair Dataset

| Cell | Import | Function Called | Purpose in Notebook |
|------|--------|----------------|---------------------|
| 1 | `from cheat_at_search.data_dir import mount` | `mount(use_gdrive=True)` | Set up persistent data storage on Google Drive |
| 2 | `from cheat_at_search.data_dir import key_for_provider` | `key_for_provider("openai")` | Obtain (or prompt for) the OpenAI API key |
| 3 | `from cheat_at_search.wands_data import corpus` | (attribute access triggers lazy load) | Load the 43K-product Wayfair dataset |
| 4 | `from cheat_at_search.tokenizers import snowball_tokenizer` | `snowball_tokenizer(keywords)` | Tokenize + stem text for BM25 search |

Notebook 1a uses **four library entry points**. Everything else -- the `search_furniture` function, the `SearchRequest` Pydantic model, the RAG loop -- is written directly in the notebook cells.

#### Notebook 1b: Agentic Search Loop

| Cell | Import | Function Called | Purpose in Notebook |
|------|--------|----------------|---------------------|
| 1 | `from cheat_at_search.data_dir import mount` | `mount(use_gdrive=True)` | Set up persistent data storage on Google Drive |
| 2 | `from cheat_at_search.data_dir import key_for_provider` | `key_for_provider("openai")` | Obtain (or prompt for) the OpenAI API key |
| 3 | `from cheat_at_search.wands_data import corpus` | (attribute access triggers lazy load) | Load the 43K-product Wayfair dataset |
| 4 | `from cheat_at_search.tokenizers import snowball_tokenizer` | `snowball_tokenizer(keywords)` | Tokenize + stem text for BM25 search |
| 7 | `from cheat_at_search.agent.pydantize import make_tool_adapter` | `make_tool_adapter(search_furniture)` | Convert a Python function into an LLM-callable tool specification |

Notebook 1b uses **five library entry points** -- the same four as 1a plus `make_tool_adapter`. This single new import is the key differentiator: it bridges the gap between "a Python function" and "a tool the LLM can autonomously call." The notebook also references (but has a bug where it does not import) `SearchResults` from `agent/search_client.py`, which would provide structured output for the final agent response.

### How Each Library Function Helps

#### `data_dir.mount()` -- Environment Bootstrapping

**Problem it solves**: Google Colab gives you a fresh filesystem every session. Downloaded datasets and API keys would be lost each time.

**What it does for the notebooks**: A single call to `mount(use_gdrive=True)` handles all of the following behind the scenes:

```python
# What mount() does internally:
# 1. Import and call Google Colab's drive.mount()
from google.colab import drive
drive.mount('/content/drive')

# 2. Point the library's global DATA_PATH to a persistent location
DATA_PATH = '/content/drive/MyDrive/cheat-at-search-data/'

# 3. Create the directory if it doesn't exist
pathlib.Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
```

**What the notebook avoids writing**: Without `mount()`, every notebook would need 5-6 lines of Colab-specific boilerplate at the top, plus conditional logic for non-Colab environments.

**Multiple environments supported**: The function signature `mount(use_gdrive=True, manual_path=None)` lets the same notebook work in three contexts:
- `mount(use_gdrive=True)` -- Colab with Google Drive persistence
- `mount(use_gdrive=False)` -- Colab without Drive (ephemeral local directory)
- `mount(use_gdrive=False, manual_path="/my/local/path")` -- Running locally outside Colab

This means a student can clone the repo and run locally without modifying code -- just change one argument.

#### `data_dir.key_for_provider()` -- Secure API Key Handling

**Problem it solves**: Notebooks need API keys, but hardcoding them is a security risk, and environment variables are not always available in Colab.

**What it does for the notebooks**: The call `key_for_provider("openai")` performs a three-step lookup:

```python
# Step 1: Check environment variable OPENAI_API_KEY
if os.getenv("OPENAI_API_KEY"):
    return os.getenv("OPENAI_API_KEY")

# Step 2: Check if previously loaded in this session
elif "OPENAI_API_KEY" in globals():
    return globals()["OPENAI_API_KEY"]

# Step 3: Prompt user interactively and save to keys.json
else:
    key = getpass.getpass("Enter your OPENAI_API_KEY: ")
    # Save to DATA_PATH/keys.json for future sessions
    json.dump({"OPENAI_API_KEY": key}, open(KEY_PATH, 'w'))
    return key
```

**What the notebook avoids writing**: Without this, each notebook would need ~15 lines of key management code, including `getpass` import, file I/O for persistence, and environment variable checking. Students would also be at risk of accidentally committing keys to git.

**Cross-notebook persistence**: Because keys are saved to `keys.json` inside `DATA_PATH` (which is on Google Drive after `mount()`), a student enters their API key once and it persists across all notebooks and all Colab sessions.

#### `wands_data.corpus` -- One-Line Dataset Access

**Problem it solves**: The WANDS dataset lives in a separate GitHub repository, is stored as tab-delimited CSVs, and needs several transformations before it is useful.

**What it does for the notebook**: The import `from cheat_at_search.wands_data import corpus` triggers this chain:

```
Step 1: Module __getattr__('corpus') fires
    │
    ▼
Step 2: _corpus() called
    │
    ├── fetch_wands() → git clone/pull the WANDS repo into DATA_PATH/wands_enriched/
    │
    ├── pd.read_csv('product.csv', sep='\t') → Load 43K products
    │
    ├── Split 'product_features' string on '|' into lists
    │
    ├── Normalize columns:
    │     df['doc_id'] = df['product_id']
    │     df['title'] = df['product_name']
    │     df['description'] = df['product_description']
    │
    ├── Parse category hierarchy:
    │     "Furniture / Bedroom Furniture / Beds" →
    │       category="Furniture", sub_category="Bedroom Furniture"
    │
    └── Return ready-to-use DataFrame
    │
    ▼
Step 3: globals()['corpus'] = df  (cached for future access)
```

**What the notebook avoids writing**: Without this, the notebook would need ~30 lines of git-clone logic, CSV parsing, column normalization, and category parsing. Students would need to understand the raw data format before they could focus on the search concepts.

**What the notebook gets**: A clean DataFrame with standardized columns:

```
corpus.columns:
  product_id, product_name, product_class, category hierarchy,
  product_description, product_features, rating_count, average_rating,
  review_count, features (list), doc_id, title, description,
  category, sub_category, cat_subcat
```

The `title` and `description` columns are ready for indexing. The `category` column is ready for filtering. No data wrangling needed in the notebook itself.

#### `tokenizers.snowball_tokenizer()` -- Text-to-Search-Tokens

**Problem it solves**: BM25 search requires tokenized text. Raw strings cannot be scored.

**What it does for the notebook**: Used in two places within notebook 1a:

**1. Indexing** (building the search index):

```python
corpus['title_snowball'] = SearchArray.index(corpus['title'].fillna(''), snowball_tokenizer)
corpus['description_snowball'] = SearchArray.index(corpus['description'].fillna(''), snowball_tokenizer)
```

`SearchArray.index()` calls `snowball_tokenizer` on every product's title and description to build an inverted index. The tokenizer transforms each product text:

```
"solid wood platform bed" → ["solid", "wood", "platform", "bed"]
"all-clad 7 qt . slow cooker" → ["all", "clad", "7", "qt", "slow", "cooker"]
```

**2. Querying** (inside the `search_furniture` function):

```python
for term in snowball_tokenizer(keywords):
    bm25_scores += corpus['title_snowball'].array.score(term) * 7
```

The same tokenizer processes the user's query so that query terms match indexed terms. This is critical -- if the index stems "running" to "run" but the query does not, there will be no match.

**What the notebook avoids writing**: The tokenizer handles three non-trivial text processing steps:
1. **ASCII folding**: Curly quotes, em-dashes, and other Unicode → plain ASCII
2. **Punctuation removal**: All punctuation → spaces (so "qt." becomes "qt")
3. **Snowball stemming**: Morphological reduction (so "couches" matches "couch")

Without the library tokenizer, the notebook would need ~15 lines of text processing code plus a PyStemmer dependency import.

### `agent.pydantize.make_tool_adapter()` -- Function-to-Tool Conversion (Notebook 1b)

**Problem it solves**: OpenAI's tool-calling API requires a specific JSON schema for each tool: a name, description, and parameters object. Writing this by hand for every function is tedious and error-prone. Keeping it in sync with the actual Python function is worse.

**What it does for the notebook**: A single call converts the notebook's `search_furniture` function into everything the LLM needs:

```python
from cheat_at_search.agent.pydantize import make_tool_adapter

search_tool = make_tool_adapter(search_furniture)
# Returns a 3-tuple: (ArgsModel, tool_spec, call_from_tool)
```

The three outputs:

1. **`ArgsModel`** -- A Pydantic model dynamically generated from the function signature. For `search_furniture(keywords: str, categories: Optional[list[Categories]] = None)`, it creates:
   ```python
   class Search_furnitureArgs(BaseModel):
       keywords: str
       categories: Optional[list[Categories]] = None
   ```

2. **`tool_spec`** -- The JSON that OpenAI expects, automatically generated from the function's name, docstring, and type hints:
   ```json
   {
     "type": "function",
     "name": "search_furniture",
     "description": "Search the available furniture products, get top 10...",
     "parameters": {
       "properties": {
         "keywords": {"title": "Keywords", "type": "string"},
         "categories": {"anyOf": [{"items": {"enum": ["Furniture", ...]}, "type": "array"}, {"type": "null"}]}
       },
       "required": ["keywords"]
     }
   }
   ```

3. **`call_from_tool`** -- A function that accepts the raw JSON arguments from the LLM, validates and deserializes them via `ArgsModel`, calls the original Python function, and serializes the result back to JSON.

**What the notebook avoids writing**: Without `make_tool_adapter`, the notebook would need ~30 lines of boilerplate: manually defining the JSON schema, writing argument parsing logic, handling serialization of the response. Crucially, the function's docstring, parameter names, type hints, and `Literal` enum values all become part of the tool specification automatically -- so the LLM sees them as prompt context when deciding how to call the tool.

**Why the docstring and types matter**: The LLM reads the tool_spec to decide *how* to use the tool. When it sees `description: "Don't expect sophisticated synonyms or semantic search. Just basic keyword with some stemming"`, it learns to send expanded keyword queries rather than natural language questions. When it sees the `categories` enum, it knows which filters are valid.

### What the Notebooks Build On Top of the Library

The library provides the *infrastructure*; the notebooks build the *application logic*:

| Responsibility | Handled by Library | Built in Notebook |
|---|---|---|
| Data storage & API keys | `mount()`, `key_for_provider()` | -- |
| Dataset acquisition & loading | `wands_data.corpus` | -- |
| Text tokenization & stemming | `snowball_tokenizer()` | -- |
| Function-to-tool conversion | `make_tool_adapter()` (1b only) | -- |
| Search index creation | -- | `SearchArray.index(...)` call |
| BM25 search function | -- | `search_furniture()` definition |
| Required-term (`+`) filtering | -- | `search_furniture()` logic |
| Category filtering | -- | `search_furniture()` logic (1b adds this) |
| Structured search request model | -- | `SearchRequest` Pydantic class (1a) |
| LLM query generation | -- | `openai.responses.parse()` calls (1a) |
| LLM result summarization | -- | `openai.responses.create()` calls |
| RAG chat loop orchestration | -- | The manual loop cell (1a) |
| Tool call dispatch + loop | -- | The `call_tool` + `agentic_search` loop (1b) |

This division is intentional for teaching: the library handles the "plumbing" (data loading, tokenization, key management, tool serialization) so students can focus on the novel concepts (RAG architecture, structured outputs, agentic tool-calling).

### Library Capabilities Available but Not Used in the Notebooks (Yet)

The library contains significant functionality that the current notebooks do not exercise. These represent future notebook material:

| Library Module | Capability | Potential Notebook Use |
|---|---|---|
| `strategy/bm25.py` | `BM25Search` class with configurable field boosts | Replace the inline `search_furniture()` with the reusable class |
| `strategy/best.py` | `BestPossibleResults` oracle strategy | Compare BM25 results against the ideal ranking |
| `eval.py` + `search.py` | NDCG evaluation framework | Measure how well the RAG/agentic pipeline ranks products |
| `model/query.py` | `StructuredQuery` with material/color/room facets | More sophisticated query understanding than `SearchRequest` |
| `model/product.py` | `ProductCategory`, `EnrichedProduct` | LLM-based product classification and enrichment |
| `enrich/` | `AutoEnricher`, `ProductEnricher` | Batch-enrich all 43K products with LLM-generated attributes |
| `agent/openai_agent.py` | `OpenAIAgent` with complete tool-calling loop | Replace the hand-written agentic loop with the library's production-ready version |
| `agent/search_client.py` | `SearchResults`, `SearchResult` structured models | Structured output for the agent's final answer (1b references but does not import) |
| `wands_data` | `judgments`, `enriched_products`, `product_embeddings` | Evaluation datasets and pre-computed embeddings |
| `cache.py` | `StoredLruCache` | Cache LLM enrichment calls to avoid API cost on re-runs |

The realized progression so far:
1. **Notebook 1**: OpenAI API basics -- chat loop, structured outputs
2. **Notebook 1a**: Manual RAG -- notebook code orchestrates query generation, search, summarization
3. **Notebook 1b**: Agentic search -- LLM autonomously calls tools, `make_tool_adapter` bridges Python functions to LLM tool specs

Likely future notebooks:
4. Use `OpenAIAgent` class to replace the hand-written agentic loop
5. Use `BM25Search` class + NDCG evaluation to measure search quality
6. Use `enrich/` to improve search quality with LLM-enriched product data

### Data Flow: How Library and Notebook Code Interact

This diagram shows which pieces of code are from the library vs. written in the notebook:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         NOTEBOOK CODE                                │
│                                                                      │
│  User: "modern geometric couch"                                      │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────┐                                    │
│  │ openai.responses.parse()    │  LLM generates SearchRequest        │
│  │ text_format=SearchRequest   │  search_query="modern geometric..." │
│  └──────────┬───────────────────┘  category=["Furniture"]            │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────────────┐  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │
│  │ search_furniture(keywords)  │    LIBRARY: snowball_tokenizer()│  │
│  │   for term in tokenizer():  │──▶│ stems "geometric" → "geometr"│  │
│  │   score += title * 7        │    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  │
│  │   score += desc * 4         │  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │
│  │   zero out missing required │    LIBRARY: corpus DataFrame    │  │
│  │   return top 10             │──▶│ 43K rows, pre-loaded via     │  │
│  └──────────┬───────────────────┘  │ wands_data.__getattr__       │  │
│             │                      └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  │
│             ▼                                                        │
│  ┌──────────────────────────────┐                                    │
│  │ openai.responses.create()   │  LLM reads results + question       │
│  │ input=[system, user, results]│  generates curated summary         │
│  └──────────┬───────────────────┘                                    │
│             │                                                        │
│             ▼                                                        │
│  "Here are 5 couches with geometric lines..."                        │
└──────────────────────────────────────────────────────────────────────┘

Library setup (runs once, before the loop):
  mount()              → DATA_PATH = Google Drive
  key_for_provider()   → OPENAI_KEY retrieved/prompted
  import corpus        → 43K products loaded, columns normalized
  SearchArray.index()  → inverted index built using snowball_tokenizer
```

### Why This Library/Notebook Split Works Well

**For students**: The notebooks stay focused. A student sees `from cheat_at_search.wands_data import corpus` and gets a ready-to-use DataFrame. They do not need to understand git cloning, CSV parsing, or column normalization to learn about RAG. Those details live in the library and can be explored later.

**For iteration**: The `search_furniture()` function is defined inline in the notebook, not imported from the library. This is deliberate -- it lets students modify the search logic (change boosts, add category filters, try different scoring) without touching library code. The function is small enough to understand in one screen, yet functional enough to power real search over 43K products.

**For progression**: The library contains more advanced versions of what the notebooks do manually. The notebook's inline `search_furniture()` is a simpler version of the library's `BM25Search` class. The notebook's manual RAG loop is a simpler version of the library's `OpenAIAgent`. This creates a natural learning path: understand the simple version first, then graduate to the library abstractions.

---

## Phase 3 -- Detailed Component Analysis

### 3.1 Notebook 1: Basic Chat Loop

This notebook is a primer on the OpenAI API. It teaches three concepts in sequence:

#### Concept A: Simple Message -> Response

```python
resp = openai.responses.create(
    model="gpt-5",
    input=[
        {"role": "system", "content": "Take on the personality of Homer Simpson"},
        {"role": "user", "content": "Hi Homer"}
    ]
)
```

The `input` list is the conversation history. Each item has a `role` ("system", "user", or "assistant") and `content`. The system message sets the LLM's behavior; the user message is the human's turn.

#### Concept B: Structured Outputs with Pydantic

Instead of getting free-form text, you can force the LLM to return data matching a schema:

```python
class HomerMessage(BaseModel):
    message: str = Field(..., description="The message from Homer")
    work_complaints_this_week: list[str] = Field([], description="Complaints from Homer this week")
    donuts_eaten: int = Field(..., description="How many Donuts has Homer eaten?")

resp = openai.responses.parse(
    model="gpt-5",
    input=[...],
    text_format=HomerMessage    # <-- Forces output to match this schema
)
resp.output_parsed  # Returns a HomerMessage instance, not raw text
```

The key difference is `.parse()` instead of `.create()`, and the `text_format` parameter. Under the hood, OpenAI constrains the token decoding so only tokens producing valid JSON matching the schema can be generated.

**Why this matters for search**: Later, the LLM will output `SearchRequest` objects with a `search_query` field and optional `category` filter -- structured data the search engine can consume directly.

#### Concept C: The Chat Loop

```python
inputs = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

for _ in range(5):
    resp = openai.responses.create(model="gpt-5", input=inputs)
    inputs += resp.output                              # Append assistant's response
    response_from_user = input(resp.output[-1].content[-1].text)
    inputs += [{"role": "user", "content": response_from_user}]  # Append user's reply
```

The `inputs` list grows each iteration: system -> user -> assistant -> user -> assistant -> ... This is how the LLM maintains conversation context. Each API call sees the entire history.

### 3.2 Notebook 1a: RAG with Wayfair Dataset

This notebook builds a complete RAG pipeline in progressive steps.

#### Step 1: Load the Wayfair Product Corpus

```python
from cheat_at_search.wands_data import corpus
corpus['category'] = corpus['category'].str.strip()
```

This single import triggers a chain of lazy-loading operations (see Section 3.4 on `wands_data`). The result is a DataFrame with ~43,000 rows, each representing a furniture product with columns like `title`, `description`, `category`, `product_features`, `rating_count`, etc.

#### Step 2: Build a Search Index

```python
from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer

corpus['title_snowball'] = SearchArray.index(corpus['title'].fillna(''), snowball_tokenizer)
corpus['description_snowball'] = SearchArray.index(corpus['description'].fillna(''), snowball_tokenizer)
```

`SearchArray` is a pandas extension type that stores an inverted index *inside a DataFrame column*. After indexing, each cell in `title_snowball` contains the stemmed tokens for that row's title, and the column as a whole supports BM25 scoring.

The `snowball_tokenizer` normalizes text by: ASCII-folding special characters, converting to lowercase, splitting on whitespace, and stemming each word with the Snowball algorithm (e.g., "running" -> "run", "furniture" -> "furnitur").

#### Step 3: Define the Search Function

```python
def search_furniture(keywords: str) -> list[dict[str, Union[str, int, float]]]:
    """Search the available furniture products, get top 10 furniture."""
    required_keywords = [term[1:] for term in keywords.split() if term.startswith("+")]
    bm25_scores = np.zeros(len(corpus))
    for term in snowball_tokenizer(keywords):
        bm25_scores += corpus['title_snowball'].array.score(term) * 7
        bm25_scores += corpus['description_snowball'].array.score(term) * 4

    for required_term in snowball_tokenizer(" ".join(required_keywords)):
        required_score = (corpus['title_snowball'].array.score(required_term) +
                          corpus['description_snowball'].array.score(required_term))
        bm25_scores[required_score == 0] = 0  # Zero out products missing required terms

    top_k_indices = np.argsort(bm25_scores)[-10:][::-1]
    # ... format results as list of dicts ...
```

Key details:
- **Field boosting**: Title matches get 7x weight, description matches get 4x. This means a term appearing in the title matters more than in the description.
- **Required terms**: Keywords prefixed with `+` (like `+couch`) are mandatory -- any product not containing them gets a score of zero.
- **BM25 scoring**: The `.array.score(term)` call returns a numpy array of BM25 scores across all 43K documents for that term. Scores are additive across terms.

#### Step 4: Define the Structured Search Request

```python
class SearchRequest(BaseModel):
    """A simple keyword search to the furniture search index."""
    search_query: str = Field(..., description="The search query")
    category: list[Categories] = Field([], description="Filter by category, empty for no filters")
```

`Categories` is a `Literal` type with values like `'Furniture'`, `'Home Improvement'`, `'Lighting'`, etc. By using `Literal`, the LLM can only output one of the predefined category names -- it cannot hallucinate categories.

#### Step 5: LLM Generates the Search Query

```python
system_prompt = "Users are coming to explore a catalog of furniture. Generate a search query"
resp = openai.responses.parse(
    model="gpt-5",
    input=[{"role": "system", "content": system_prompt},
           {"role": "user", "content": "Help me find a modern couch with geometric style"}],
    text_format=SearchRequest
)
# Result: SearchRequest(search_query="modern geometric couch sofa contemporary angular...", category=["Furniture"])
```

The LLM expands the user's brief "modern couch with geometric style" into a richer search query with synonyms and related terms. This is a form of **query expansion** -- traditionally a hard IR problem, now handled naturally by the LLM.

#### Step 6: Search -> LLM Summarization (The RAG Step)

```python
furniture = search_furniture(resp.output_parsed.search_query)  # BM25 retrieval

# Feed results back to LLM
inputs.append({"role": "user", "content": str(furniture)})
resp = openai.responses.create(model="gpt-5", input=inputs)
```

The search results (a list of dictionaries with id, title, description, score) are serialized as a string and injected into the conversation as a "user" message. The LLM reads these results and generates a curated summary, picking the most relevant products and explaining *why* they match the user's intent.

#### Step 7: The Full RAG Chat Loop

The notebook's final cell combines all pieces into an interactive loop:

```python
for _ in range(5):
    # 1. LLM generates search query from conversation context
    resp = openai.responses.parse(model="gpt-5", input=search_query_inputs, text_format=SearchRequest)

    # 2. BM25 retrieves products
    furniture = search_furniture(search_settings.search_query)

    # 3. LLM summarizes results for the user
    chat_inputs.append({"role": "user", "content": str(furniture)})
    resp = openai.responses.create(model="gpt-5", input=chat_inputs)

    # 4. Get next user message
    user_response = input("User: ")
    chat_inputs.append({"role": "user", "content": user_response})
    search_query_inputs.append({"role": "user", "content": user_response})
```

Notice there are **two parallel conversation tracks**:
- `search_query_inputs` -- used to generate search queries (system prompt: "Generate a search query")
- `chat_inputs` -- used for the user-facing conversation (system prompt: "Answer the user's request")

Both tracks receive the user's messages, but they have different system prompts because they serve different purposes. This is a design choice: the search query generator does not see the full search results, only the user's intent. The chat responder sees the search results but uses a different system prompt focused on summarization.

### 3.3 Notebook 1b: Agentic Search Loop

This notebook is the critical step from "manual RAG" (notebook 1a) to "agentic AI." The setup is identical to 1a (mount, load corpus, build index, define `search_furniture`), but the architecture for *how the LLM uses search* changes fundamentally.

#### Key Difference from 1a: Who Decides When to Search?

| Aspect | Notebook 1a (Manual RAG) | Notebook 1b (Agentic) |
|--------|--------------------------|----------------------|
| **Who decides to search** | The notebook code (always searches every turn) | The LLM (decides if/when/how to search) |
| **Who formulates the query** | A separate LLM call with `text_format=SearchRequest` | The LLM generates tool-call arguments directly |
| **Search execution** | Notebook calls `search_furniture()` directly | Notebook dispatches tool calls requested by the LLM |
| **Number of searches per turn** | Always exactly one | LLM may search zero, one, or multiple times |
| **Self-correction** | None -- one search, one answer | LLM can refine queries based on results (observed in output: it searched again with "geometric pattern sofa couch" after the first results) |

#### Step 1: Same Setup, Enhanced `search_furniture`

The `search_furniture` function in 1b adds **category filtering** compared to 1a:

```python
def search_furniture(keywords: str,
                     categories: Optional[list[Categories]] = None
                     ) -> list[dict[str, Union[str, int, float]]]:
    # ... same BM25 scoring as 1a ...

    if categories:
        bm25_scores[~corpus['category'].isin(categories)] = 0  # Zero out non-matching categories

    # ... same top-k selection ...
```

This is important because the LLM will now *choose* whether to filter by category. In 1a, the `SearchRequest` model had a category field, but it was separate from the search function. In 1b, the function itself accepts categories, and the LLM calls it directly with both arguments.

#### Step 2: Convert the Function to an LLM Tool

This is where notebook 1b diverges from 1a. Instead of having the notebook orchestrate everything, the function is registered as a tool the LLM can call:

```python
from cheat_at_search.agent.pydantize import make_tool_adapter

search_tool = make_tool_adapter(search_furniture)
tool_info = {search_furniture.__name__: search_tool}
```

`make_tool_adapter` inspects `search_furniture` and produces:
- A JSON tool specification derived from the function's name, docstring, and type annotations
- A deserializer that converts the LLM's JSON arguments into Python objects
- A serializer that converts the Python return value back to JSON

The notebook then pretty-prints the generated spec to show what the LLM will see. Key point: the function's **docstring** ("Don't expect sophisticated synonyms or semantic search. Just basic keyword with some stemming") and the **`Literal` enum** for categories become part of the LLM's prompt. This is how the LLM learns what the tool can and cannot do.

#### Step 3: Demonstrate Without vs. With Tools

The notebook makes two LLM calls to contrast behavior:

**Without tools** (no `tools=` parameter):
```python
resp = openai.responses.create(model="gpt-5", input=inputs)
# LLM responds with generic recommendations it makes up -- no grounding in real data
```

**With tools** (passing the tool specification):
```python
resp = openai.responses.create(
    model="gpt-5",
    input=inputs,
    tools=[tool[1] for tool in tool_info.values()],
)
```

When tools are provided, the LLM does NOT generate a text response. Instead, its output contains a `ResponseFunctionToolCall`:

```python
resp.output = [
    ResponseReasoningItem(...),  # Internal reasoning
    ResponseFunctionToolCall(
        name='search_furniture',
        arguments='{"keywords":"modern geometric couch sofa","categories":["Furniture"]}',
        call_id='call_3YvAmSiCjufIHweqrIUFOqbz',
        type='function_call'
    )
]
```

The LLM has *requested* that the notebook execute `search_furniture` with those specific arguments. It chose to filter to "Furniture" category and expanded the query with "sofa."

#### Step 4: Execute the Tool Call and Return Results

The notebook defines a `call_tool` helper:

```python
def call_tool(item) -> dict:
    tool_name = item.name                    # "search_furniture"
    tool = tool_info[tool_name]
    ToolArgsModel = tool[0]                  # The generated Pydantic model
    tool_fn = tool[2]                        # The wrapped callable

    # Deserialize LLM's JSON args -> Python, call the function, serialize result -> JSON
    fn_args = ToolArgsModel.model_validate_json(item.arguments)
    py_resp, json_resp = tool_fn(fn_args)

    return {
        "type": "function_call_output",
        "call_id": item.call_id,             # Links this result to the LLM's request
        "output": json_resp,                 # Search results as JSON string
    }
```

The tool response is appended to `inputs` and the LLM is called again. Critically, the LLM may respond with *another tool call* -- it examined the first batch of results and decided to search again with different keywords ("geometric pattern sofa couch"). This self-correction is the defining feature of agentic behavior.

#### Step 5: The Full Agentic Loop

The notebook's final function wraps everything into a while-loop:

```python
def agentic_search(query: str, summary=True) -> str:
    inputs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    tool_calls = True
    while tool_calls:
        resp = openai.responses.create(
            model="gpt-5",
            input=inputs,
            tools=[tool[1] for tool in tool_info.values()],
            reasoning={"effort": "auto", "summary": "auto" if summary else "none"},
        )
        inputs += resp.output

        # Print reasoning summaries (shows the LLM's thought process)
        for item in resp.output:
            if item.type == "reasoning":
                for summary_item in item.summary:
                    print("Reasoning:", summary_item.text)

        # Execute any tool calls the LLM requested
        tool_calls = False
        for item in resp.output:
            if item.type == "function_call":
                tool_calls = True
                tool_response = call_tool(item)
                inputs.append(tool_response)

    return resp
```

The loop terminates when the LLM produces a response with no tool calls -- meaning it is satisfied with the results and ready to give a final answer.

**Observed behavior from the notebook output**: For the query "Help me find a modern couch with geometric style":
1. LLM calls `search_furniture(keywords="modern geometric couch sofa", categories=["Furniture"])`
2. Receives 10 results
3. Calls `search_furniture(keywords="geometric pattern sofa couch", categories=["Furniture"])` -- a *second* search with refined keywords
4. Would combine both result sets and generate a final response

This multi-round search is something notebook 1a cannot do -- 1a always does exactly one search per user turn.

#### The System Prompt's Role

```python
system_prompt = """
Users are coming to explore a catalog of furniture.

Use the search tool (search_furniture) to help them.

Use trial and error to figure out how best use the search tool.
"""
```

The instruction "Use trial and error" explicitly encourages the agentic behavior: the LLM is told it can try multiple searches, learn from results, and refine its approach. This is a teaching point about how system prompts shape agent behavior.

#### Bug in the Notebook

The final `agentic_search` function references `SearchResults` in its `text_format=` parameter but never imports it:

```python
resp = openai.responses.create(
    ...
    text_format=SearchResults  # NameError: name 'SearchResults' is not defined
)
```

This would be fixed by adding `from cheat_at_search.agent.search_client import SearchResults`. The `SearchResults` model (from the library) provides a structured format for the agent's final answer, including ranked results with relevance scores and a self-evaluation. This is another example of library functionality available but not yet properly connected.

#### 1a vs 1b: Architecture Comparison

```
NOTEBOOK 1a (Manual RAG):                  NOTEBOOK 1b (Agentic):

 User input                                 User input
    │                                           │
    ▼                                           ▼
 [LLM #1: generate SearchRequest]           [LLM: decide what to do]
    │                                           │
    ▼                                           ├──▶ tool_call: search_furniture(...)
 notebook calls search_furniture()              │         │
    │                                           │         ▼
    ▼                                           │    notebook executes, returns results
 [LLM #2: summarize results]                   │         │
    │                                           │         ▼
    ▼                                           ├──▶ [LLM: more tool calls?]
 Response to user                               │    (may search again, refine, etc.)
                                                │         │
 Fixed: always 2 LLM calls, 1 search           ▼
 Two separate conversation tracks           Response to user
                                            
                                            Dynamic: N LLM calls, M searches
                                            Single unified conversation
```

### 3.4 Library: `data_dir.py` -- Data Management

> Note: Sections 3.4 through 3.12 cover the library components. These are unchanged from the earlier analysis; the numbering shifted to accommodate section 3.3 (Notebook 1b).

This module handles three concerns:

**1. Data Path Resolution**

```python
if os.environ.get("CHEAT_AT_SEARCH_DATA_PATH"):
    DATA_PATH = os.environ["CHEAT_AT_SEARCH_DATA_PATH"]
else:
    DATA_PATH = pathlib.Path(get_project_root()) / "data"
```

On Colab, `mount()` redirects `DATA_PATH` to Google Drive so data persists across sessions.

**2. Dataset Downloading**

`sync_git_repo()` clones/updates a git repository (the WANDS dataset) into the data directory. It handles: first-time clone (`git clone --depth=1`), updates (`git fetch + reset --hard`), and failed states (delete + re-clone).

**3. API Key Management**

`key_for_provider("openai")` checks, in order: environment variable `OPENAI_API_KEY`, a previously loaded global, or prompts the user interactively via `getpass` and saves to a JSON file. This lets notebooks work in Colab (where env vars aren't pre-set) without hardcoding keys.

### 3.4 Library: `wands_data.py` -- Dataset Access

This module provides the WANDS dataset as module-level attributes via lazy loading.

**Core datasets exposed:**

| Attribute | Type | Description |
|---|---|---|
| `corpus` / `products` | DataFrame (43K rows) | Product catalog: title, description, features, category hierarchy, ratings |
| `queries` | DataFrame | Human search queries used for evaluation |
| `judgments` / `labeled_queries` | DataFrame | Human relevance labels: "Exact", "Partial", or "Irrelevant" per query-product pair |
| `enriched_products` | DataFrame | Products enriched with LLM-generated attributes |
| `enriched_queries` | DataFrame | Queries enriched with LLM-parsed attributes |
| `product_embeddings` | numpy array | Pre-computed vector embeddings for products |

**Data loading functions** like `_corpus()` do the following:
1. Call `fetch_wands()` to ensure the git repo is cloned
2. Read a CSV file (`pd.read_csv(..., sep='\t')`)
3. Normalize columns: create common `doc_id`, `title`, `description` fields
4. Parse category hierarchy into `category` and `sub_category`

**Relevance labeling** in `_labels()`:

```python
df.loc[df['label'] == 'Exact', 'grade'] = 2
df.loc[df['label'] == 'Partial', 'grade'] = 1
df.loc[df['label'] == 'Irrelevant', 'grade'] = 0
```

These numeric grades feed into the NDCG evaluation to measure search quality.

### 3.5 Library: `tokenizers.py` -- Text Processing

Three tokenizers for different use cases:

```python
def snowball_tokenizer(text):
    text = text.translate(all_trans).replace("'", " ")  # ASCII fold + strip punctuation
    split = text.lower().split()                          # Lowercase + split
    return [stem_word(token) for token in split]           # Stem each token
```

Example:
- Input: `"all-clad 7 qt. slow cooker"`
- After folding + punctuation removal: `"all clad 7 qt  slow cooker"`
- After lowercase + split: `["all", "clad", "7", "qt", "slow", "cooker"]`
- After stemming: `["all", "clad", "7", "qt", "slow", "cooker"]` (these happen to not change much)

The `fold_to_ascii` dictionary maps special Unicode characters like curly quotes to plain ASCII equivalents. The `punct_trans` table maps all punctuation to spaces.

### 3.6 Library: `strategy/` -- Search Strategies

#### `SearchStrategy` (Base Class)

```python
class SearchStrategy:
    def __init__(self, corpus, top_k=5, workers=1):
        self.corpus = corpus

    def search_all(self, queries, k=10):
        # Runs self.search() for each query using ThreadPoolExecutor
        # Returns concatenated DataFrame of all results
        ...

    def search(self, query, k):
        raise NotImplementedError("Subclasses should implement this method.")
```

`search_all` parallelizes evaluation across queries using `ThreadPoolExecutor`. For each query, it calls `self.search()`, collects the top-K document indices and scores, and assembles them into a DataFrame with columns like `query_id`, `rank`, `score`, `doc_id`, `title`.

#### `BM25Search`

```python
class BM25Search(SearchStrategy):
    def __init__(self, corpus, title_boost=9.3, description_boost=4.1):
        # Index title and description if not already indexed
        if 'title_snowball' not in self.index:
            self.index['title_snowball'] = SearchArray.index(corpus['title'], snowball_tokenizer)
        ...

    def search(self, query, k=10):
        tokenized = snowball_tokenizer(query)
        bm25_scores = np.zeros(len(self.index))
        for token in tokenized:
            bm25_scores += self.index['title_snowball'].array.score(token) * self.title_boost
            bm25_scores += self.index['description_snowball'].array.score(token) * self.description_boost
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores
```

BM25 (Best Match 25) is a ranking function from information retrieval. For each query term, it scores every document based on term frequency, document length, and inverse document frequency. The `SearchArray` handles the BM25 math; this class applies field boosting (title 9.3x, description 4.1x) and aggregates scores across terms.

#### `BestPossibleResults`

This is an "oracle" strategy that uses the human relevance labels to return the best possible ranking. It is useful as a ceiling -- if BM25 gets NDCG of 0.6 and the oracle gets 1.0, you know how much room for improvement exists.

### 3.7 Library: `eval.py` -- Search Quality Measurement

```python
def grade_results(judgments, search_results, max_grade=None, k=10):
    graded_results = search_results.merge(judgments[['query_id', 'query', 'doc_id', 'grade']],
                                          on=['query_id', 'query', 'doc_id'], how='left')
    graded_results['grade'] = graded_results['grade'].fillna(0)
    rank_discounts = 1 / np.log2(2 ** graded_results['rank'])
    graded_results['discounted_gain'] = ((2 ** graded_results['grade']) - 1) * rank_discounts
    graded_results['idcg'] = idcg_max(max_grade=max_grade, k=k)
    return graded_results
```

This implements **NDCG (Normalized Discounted Cumulative Gain)**, the standard metric for evaluating ranked search results:
- **Gain**: `2^grade - 1` (grade 2 -> gain 3, grade 1 -> gain 1, grade 0 -> gain 0)
- **Discount**: `1 / log2(2^rank)` -- items at higher ranks (lower positions) matter more
- **DCG**: Sum of discounted gains across the top K results
- **NDCG**: DCG / IDCG (ideal DCG), normalized to [0, 1]

### 3.8 Library: `model/` -- Pydantic Data Models

These models serve as schemas that constrain LLM outputs.

#### Query Models

```python
class Query(BaseModel):
    keywords: str = Field(..., description="The original search query keywords")

class QueryCategory(Query):
    category: Categories = Field(description="Category of the product, if identified")
    sub_category: SubCategories = Field(description="Sub-category of the product")

class StructuredQuery(BaseModel):
    search_terms: str = Field(default="", description="A rebuilt / better search query")
    material: str = Field(default="", description="Material extracted from the query")
    color: str = Field(default="", description="Color mentioned in the query")
    furniture_type: str = Field(default="", description="Type of furniture mentioned")
    room: str = Field(default="", description="Room where the furniture would be placed")
    dimensions: List[str] = Field(default_factory=list, description="Any dimensions mentioned")
```

`StructuredQuery` decomposes a free-text query like "blue velvet dining chair for kitchen" into structured facets: material="velvet", color="blue", furniture_type="dining chair", room="kitchen". This enables faceted filtering on top of keyword search.

#### Product Models

`Product` and `EnrichedProduct` represent catalog items. `ProductCategory` and `ProductCategoryFullyQualified` classify products into the Wayfair taxonomy. The `@property` methods on `ProductCategoryFullyQualified` split a string like "Furniture / Bedroom Furniture" into separate category and sub-category fields.

### 3.9 Library: `agent/` -- LLM Agent with Tool Use

#### `pydantize.py` -- Function-to-Tool Converter

The most architecturally interesting module. `make_tool_adapter` takes any Python function and creates three things:

```python
ArgsModel, tool_spec, call_from_tool = make_tool_adapter(search_furniture)
```

1. **`ArgsModel`**: A Pydantic model dynamically created from the function's type hints
2. **`tool_spec`**: An OpenAI function-calling specification (JSON schema)
3. **`call_from_tool`**: A callable that deserializes JSON arguments and invokes the original function

This lets you register arbitrary Python functions as "tools" the LLM can call, without manually writing JSON schemas.

#### `OpenAIAgent` -- The Tool-Calling Loop

```python
class OpenAIAgent(Agent):
    def chat(self, user_prompt, inputs=None):
        tool_calls_found = True
        while tool_calls_found:
            resp = self.openai.responses.parse(model=self.model, input=inputs, tools=tools)
            inputs += resp.output
            tool_calls_found = False
            for item in resp.output:
                if item.type == "function_call":
                    tool_calls_found = True
                    # Execute the tool
                    fn_args = ToolArgsModel.model_validate_json(item.arguments)
                    py_resp, json_resp = tool_fn(fn_args)
                    inputs.append({"type": "function_call_output", "call_id": item.call_id, "output": json_resp})
        return resp, inputs, total_tokens
```

This is a **ReAct-style loop**: the LLM can decide to call tools, receive results, and decide whether to call more tools or give a final answer. The loop continues until the LLM produces a response with no tool calls.

### 3.10 Library: `enrich/` -- LLM Enrichment Pipeline

The enrichment system uses LLMs to add structured attributes to products and queries (e.g., classifying a product into a category, extracting materials, etc.).

**`EnrichClient`** (abstract) defines the interface: `enrich(prompt) -> BaseModel`.

**`AutoEnricher`** selects the right provider (OpenAI or Google) based on the model string prefix (e.g., `"openai/gpt-5"` vs `"google/gemini-2.0-flash"`), wraps it in a `CachedEnrichClient`, and provides batch enrichment with `enrich_all()`.

**`ProductEnricher`** orchestrates enriching an entire DataFrame: it generates prompts from each product row using a user-provided `prompt_fn`, calls `enrich_all`, and merges the structured results back into the DataFrame as new columns.

### 3.11 Library: `search.py` -- Evaluation Orchestration

This module ties strategies and evaluation together:

```python
def run_strategy(strategy, judgments, queries=None, num_queries=None, ...):
    results = strategy.search_all(available_queries)
    graded = grade_results(judgments, results, max_grade=max_grade, k=10)
    # Compute per-query NDCG
    dcgs = graded.groupby(["query", 'query_id'])["discounted_gain"].sum()
    ndcgs = dcgs / idcg
    return graded

def run_bm25(corpus, judgments):
    # Runs BM25, caches results to disk
    bm25 = BM25Search(corpus)
    graded_bm25 = run_strategy(bm25, judgments)
    graded_bm25.to_pickle(bm25_results_path / 'graded_bm25.pkl')
    return graded_bm25
```

`vs_ideal` creates a side-by-side comparison of actual results vs. the ideal ranking, useful for understanding where a strategy fails.

---

## Phase 4 -- Python Pattern Explanations

### 4.1 Module-Level `__getattr__` (Lazy Loading)

**Where used**: `wands_data.py`, `data_dir.py`

**What it does**: Python calls a module's `__getattr__` when you access an attribute that does not exist as a regular variable. This allows on-demand loading.

```python
# In wands_data.py
def __getattr__(name):
    """Load dataset lazily."""
    if name == 'corpus':
        ds = _corpus()          # Actually loads and parses the CSV
    elif name == 'judgments':
        ds = _labels()           # etc.
    globals()[name] = ds         # Cache so next access is instant
    return ds
```

**How it works**: When you write `from cheat_at_search.wands_data import corpus`, Python looks for `corpus` in the module's globals. Since it is not there at import time, `__getattr__('corpus')` fires, which loads the CSV, caches the DataFrame in `globals()`, and returns it.

**Why it is used**: Loading 43K products from CSV takes a few seconds. If the module eagerly loaded everything on import, importing `wands_data` would be slow even if you only needed `queries`. Lazy loading defers the cost until you actually need the data.

**Simple analogy**: Think of it like a library where books are kept in storage. The librarian (`__getattr__`) only retrieves a specific book when someone asks for it. Once retrieved, it stays on the shelf (`globals()`) for quick access next time.

### 4.2 Pydantic BaseModel and Field

**Where used**: Throughout `model/`, notebooks, `enrich/`, `agent/`

**What it does**: Pydantic's `BaseModel` provides data validation, serialization, and JSON schema generation.

```python
class SearchRequest(BaseModel):
    """A simple keyword search to the furniture search index."""
    search_query: str = Field(..., description="The search query")
    category: list[Categories] = Field([], description="Filter by category")
```

**Key features used**:
- `Field(...)` -- `...` (Ellipsis) means "required with no default"
- `Field([])` -- provides a default value (empty list)
- `description=` -- used by OpenAI's API as the field description in the JSON schema
- `.model_json_schema()` -- generates a JSON Schema that OpenAI uses for constrained decoding
- `.model_validate_json(json_string)` -- parses JSON into a validated Python object

**Why it is used for search**: When the LLM outputs structured data, you need *guarantees* about the shape. Pydantic ensures `search_query` is always a string and `category` is always a list of valid category names. Invalid output fails validation immediately rather than causing subtle downstream bugs.

### 4.3 `Literal` Types for Constrained Enums

**Where used**: `model/product.py`, `model/query.py`, `model/category_list.py`, notebook 1a

```python
Categories = Literal['Furniture', 'Home Improvement', 'Décor & Pillows', ...]

class QueryCategory(Query):
    category: Categories = Field(...)
```

**What it does**: `Literal['A', 'B', 'C']` creates a type that only allows the specified string values. When Pydantic generates a JSON schema from this, it becomes an `enum` constraint. When OpenAI's API sees this enum, it restricts token generation so the LLM can only output one of those exact strings.

**Why it matters**: Without `Literal`, the LLM might output "furniture" (lowercase) or "Home Décor" (not in the list) or "Sofas" (a subcategory). With `Literal`, you get exactly one of your predefined categories -- guaranteed.

### 4.4 `@property` Decorator

**Where used**: `model/product.py`, `model/query.py`, `agent/search_client.py`

```python
class ProductCategoryFullyQualified(BaseModel):
    full_category: FullyQualifiedCategories = Field(...)

    @property
    def category(self) -> str:
        return self.full_category.split('/')[0].strip()

    @property
    def sub_category(self) -> str:
        parts = self.full_category.split('/')
        return parts[1].strip() if len(parts) > 1 else 'No SubCategory Fits'
```

**What it does**: `@property` makes a method behave like an attribute. You access it as `obj.category` (no parentheses), not `obj.category()`.

**Why it is used here**: The LLM outputs a single `full_category` string like `"Furniture / Bedroom Furniture"`. The properties derive `category` and `sub_category` from it without the LLM needing to output them separately -- reducing the chance of inconsistency.

### 4.5 Abstract Base Classes (ABC)

**Where used**: `strategy/strategy.py`, `enrich/enrich_client.py`, `agent/search_client.py`

```python
from abc import ABC, abstractmethod

class EnrichClient(ABC):
    @abstractmethod
    def enrich(self, prompt: str) -> Optional[BaseModel]:
        pass

    @abstractmethod
    def str_hash(self) -> str:
        pass
```

**What it does**: `ABC` prevents direct instantiation of the base class. `@abstractmethod` forces subclasses to implement specific methods. If a subclass forgets to implement `enrich()`, Python raises `TypeError` at instantiation time.

**Why it is used**: This ensures all enrichment clients (OpenAI, Google, cached) share the same interface. Code that depends on `EnrichClient` works with any concrete implementation.

### 4.6 Dynamic Model Creation with `create_model`

**Where used**: `agent/pydantize.py`

```python
from pydantic import create_model
import inspect

def make_tool_adapter(func):
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    fields = {}
    for pname, p in sig.parameters.items():
        ann = hints.get(pname, Any)
        default = ... if p.default is inspect._empty else p.default
        fields[pname] = (ann, default)

    ArgsModel = create_model(f"{func.__name__.capitalize()}Args", **fields)
```

**What it does**: `create_model` builds a Pydantic model class at runtime from a dictionary of field definitions. `inspect.signature` extracts parameter names, types, and defaults from a function.

**Example**: Given `def search_furniture(keywords: str) -> list[dict]`, this creates:

```python
class Search_furnitureArgs(BaseModel):
    keywords: str
```

**Why it is used**: Instead of manually writing a Pydantic model + JSON schema + deserialization code for every tool function, `make_tool_adapter` generates all three automatically from the function's signature. This is metaprogramming -- code that writes code.

### 4.7 `TypeAdapter` for Generic Serialization

**Where used**: `agent/pydantize.py`

```python
from pydantic.type_adapter import TypeAdapter

ret_adapter = TypeAdapter(ret_ann)  # ret_ann is the function's return type annotation
py_result = ret_adapter.dump_python(result)
json_text = ret_adapter.dump_json(result).decode()
```

**What it does**: While `BaseModel` handles serialization for Pydantic models, `TypeAdapter` handles *any* type -- `list[dict]`, `tuple[str, int]`, `Optional[float]`, etc. It wraps Pydantic's serialization/validation for types that are not themselves BaseModel subclasses.

**Why it is used**: The return type of a tool function might be `list[dict[str, Union[str, int, float]]]` -- not a BaseModel. `TypeAdapter` can still serialize it to JSON for the LLM to consume.

### 4.8 `ThreadPoolExecutor` with `as_completed`

**Where used**: `strategy/strategy.py`, `enrich/enrich.py`

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=self.workers) as executor:
    futures = {}
    for _, query_row in queries.iterrows():
        future = executor.submit(self.search, query_row['query'], k)
        futures[future] = query_row

    for future in tqdm(as_completed(futures), total=len(futures)):
        query_row = futures[future]
        top_k, scores = future.result()
```

**What it does**: `ThreadPoolExecutor` runs functions in parallel threads. `executor.submit()` queues a function call and returns a `Future` -- a placeholder for the eventual result. `as_completed()` yields futures as they finish (not in submission order), which is ideal for progress bars.

**Why it is used**: When evaluating 480 queries against 43K products, running them serially takes minutes. Parallelizing across threads cuts wall-clock time proportionally. For I/O-bound LLM API calls in the enrichment pipeline, threading is even more beneficial since most time is spent waiting for network responses.

### 4.9 The `__call__` Dunder Method (Callable Objects)

**Where used**: `cache.py`

```python
class StoredLruCache:
    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        self.cache = {}

    def __call__(self, func):
        # ... load cache from disk ...
        def wrapper(*args, **kwargs):
            key = (args, tuple(kwargs.items()))
            if key in self.cache:
                return self.cache[key]
            result = func(*args, **kwargs)
            self.cache[key] = result
            self.save_cache()
            return result
        return wrapper
```

**What it does**: `__call__` makes an instance of the class callable like a function. This enables `StoredLruCache` to be used as a decorator:

```python
@StoredLruCache(maxsize=1000)
def expensive_llm_call(prompt):
    ...
```

**How it works step-by-step**:
1. `StoredLruCache(maxsize=1000)` creates an instance (calls `__init__`)
2. Python calls that instance with the decorated function: `instance(expensive_llm_call)` (calls `__call__`)
3. `__call__` returns `wrapper`, which replaces the original function
4. When `expensive_llm_call("some prompt")` is called, `wrapper` runs: check cache -> call original if miss -> save to disk

### 4.10 Dictionary Comprehensions and `str.maketrans`

**Where used**: `tokenizers.py`

```python
fold_to_ascii = dict([(ord(x), ord(y)) for x, y in zip(u"\u2018\u2019\u00b4\u201c\u201d\u2013-", u"'''\"\"--")])
punct_trans = str.maketrans({key: ' ' for key in string.punctuation})
all_trans = {**fold_to_ascii, **punct_trans}
```

**What it does**:
- `str.maketrans()` creates a translation table for `str.translate()` -- a fast, C-level character-by-character replacement.
- `{**dict1, **dict2}` merges two dictionaries (Python 3.5+ syntax). In case of key conflicts, the second dict wins.
- `ord()` converts characters to Unicode code points, which is what translation tables expect.

**Example**: `"don\u2019t".translate(all_trans)` becomes `"don t"` (curly apostrophe -> straight quote, then quote -> space via punctuation table).

### 4.11 `globals()` for Module-Level State

**Where used**: `wands_data.py`, `data_dir.py`

```python
globals()[name] = ds
return ds
```

**What it does**: `globals()` returns the module's global symbol table as a dictionary. Writing to it is equivalent to creating a module-level variable. After `globals()['corpus'] = df`, anyone importing `corpus` from this module gets `df`.

**Why it is used**: Combined with `__getattr__`, this creates a lazy-loading cache. The first access triggers loading; subsequent accesses find the value already in `globals()` and skip `__getattr__` entirely.

---

## Summary: How It All Fits Together

The three notebooks demonstrate a clear progression in how LLMs interact with search:

1. **Notebook 1** -- Learn the basics: how to talk to an LLM, how to get structured output, how to maintain conversation context.

2. **Notebook 1a** -- Manual RAG: the *notebook code* orchestrates everything -- it tells one LLM to generate a query, calls the search function, and tells another LLM to summarize results. The human programmer is the "agent."

3. **Notebook 1b** -- Agentic search: the *LLM itself* orchestrates everything -- it decides when to search, what keywords to use, whether to search again with refined queries, and when it has enough information to answer. The human programmer just runs the loop. This is a fundamentally different architecture, enabled by a single new library function (`make_tool_adapter`).

The library provides the infrastructure at each stage:
- **Data layer** (`data_dir`, `wands_data`, `tokenizers`) handles dataset acquisition and text processing -- used by all three notebooks
- **Search layer** (`strategy/`, `eval.py`, `search.py`) implements BM25 retrieval and NDCG evaluation -- available for future notebooks
- **Model layer** (`model/`) defines the structured schemas that bridge LLM output and search input -- partially used
- **Agent layer** (`agent/pydantize.py`, `agent/openai_agent.py`, `enrich/`) provides tool-calling infrastructure and enrichment -- `pydantize` is used in 1b, the rest is available for future notebooks

The core insight across the series: LLMs are good at understanding intent and generating natural language but bad at retrieval over large catalogs. Search engines are good at retrieval but bad at understanding nuanced intent. RAG (1a) combines both strengths. Agentic search (1b) goes further by letting the LLM decide *how* to combine them -- including the ability to try multiple strategies, refine queries, and self-correct.
