# Week 3 Notebooks Analysis: Agentic Search Loops over Wayfair Products

A comprehensive educational guide to the three Week 3 notebooks -- how they build progressively more sophisticated agentic search systems, the architecture behind them, every key class and function, and the Python patterns that make it all work.

---

## Table of Contents

- [Phase 1 - High-Level Overview (Search Functionality POV)](#phase-1---high-level-overview-search-functionality-pov)
- [Phase 2 - Architecture Analysis](#phase-2---architecture-analysis)
- [Phase 3 - Detailed Component Analysis](#phase-3---detailed-component-analysis)
- [Phase 4 - Python Pattern Explanations](#phase-4---python-pattern-explanations)
- [Cross-Week Meta-Patterns and Progression](#cross-week-meta-patterns-and-progression)

---

## Phase 1 - High-Level Overview (Search Functionality POV)

### What Problem Do These Notebooks Solve?

Traditional keyword search (BM25) matches documents by counting how well query terms appear in document text. It works surprisingly well, but has fundamental limitations: it cannot understand intent ("modern sofa" vs. "contemporary couch"), it cannot explore the search space strategically, and it makes one attempt -- if the first query does not return good results, no recovery is possible.

These three Week 3 notebooks build a system where an **LLM acts as a search agent** -- a reasoning system that can call a search tool multiple times, evaluate results, adjust its strategy, and return a curated top-10 ranking. The domain is the Wayfair WANDS dataset: ~43,000 furniture and home goods products with human relevance judgments for ~480 queries.

### The Three Notebooks

| Notebook | Subtitle | Core Innovation |
|----------|----------|-----------------|
| **3** - Baseline | Agentic Search Loop | An LLM uses BM25 search as a tool, issues multiple queries, returns ranked results |
| **3a** - Steer & Guide | Duplicate rejection + Feedback loop | Prevents the agent from re-searching similar queries; injects graded feedback to steer toward better results |
| **3b** - Extra Tools | Multi-tool orchestration | Three search strategies (BM25+category, BM25-only, embedding search) run as isolated sub-agents, then results are aggregated |

### What Each Notebook Demonstrates

**Notebook 3 (Baseline)** answers: "Can an LLM with a search tool outperform BM25 alone?" The answer is yes -- the agentic approach scores NDCG 0.396 vs. BM25's 0.342 on the test queries. The agent achieves this by trying multiple keyword combinations, filtering by category, and using its reasoning to rank results.

**Notebook 3a (Steer & Guide)** answers: "Can we make the agent smarter by preventing wasted searches and providing relevance feedback?" Two forces guide the agent: *repulsion* (duplicate detection blocks semantically similar queries, forcing exploration) and *attraction* (emoji-annotated feedback from human judgments tells the agent which results are good, so it can find more like them).

**Notebook 3b (Extra Tools)** answers: "Can multiple search strategies, each driven by its own agent, produce better results when combined?" Three independent sub-agents each use a different search tool (keyword+category, keyword-only, embedding), each gets 5 rounds of feedback, and then an orchestrator merges and deduplicates their results.

### The Progression in One Sentence

Single-shot agent with one tool --> feedback-guided agent with duplicate prevention --> multi-agent orchestration across diverse retrieval strategies.

---

## Phase 2 - Architecture Analysis

### Shared Foundation Across All Three Notebooks

All three notebooks share this core pipeline:

```
User Query (e.g., "small woven pouf")
    |
    v
[Optional: Query Interpretation via GPT-5]  (Notebook 3 only)
    |
    v
[System Prompt + Few-Shot Examples]
    |
    v
[Agent Loop: GPT model with tool-calling + structured output]
    |   |
    |   +--> Tool Call: search_wayfair(keywords, category, top_k)
    |   |       |
    |   |       +--> BM25 scoring (title 10x boost, description 1x)
    |   |       +--> Optional category filtering
    |   |       +--> Return top_k results as structured data
    |   |
    |   +--> (may loop: more tool calls with different queries)
    |   |
    |   +--> Final output: SearchResults {ranked_results: [doc_id, ...]}
    |
    v
[Validation: exactly 10 results, all doc_ids valid]
    |
    v
[NDCG evaluation against WANDS human judgments]
```

### How the Library Fits In

The `cheat_at_search` library provides the infrastructure:

| Library Component | Role |
|---|---|
| `cheat_at_search.data_dir` | Data loading, API key management, Google Drive mounting |
| `cheat_at_search.wands_data` | Loads the Wayfair corpus, queries, relevance judgments, product embeddings |
| `cheat_at_search.tokenizers` | Snowball stemmer for BM25 tokenization |
| `cheat_at_search.agent.pydantize` | Converts Python functions into OpenAI-compatible tool specifications |
| `cheat_at_search.strategy` | Base `SearchStrategy` class for pluggable evaluation |
| `cheat_at_search.search` | `run_strategy()`, `ndcgs()`, BM25 baseline comparison utilities |
| `searcharray` (external) | Pandas-native BM25 inverted index |

The notebooks build everything else inline: the search functions, the agent loop, the validation harness, and the evaluation wrappers.

### Architecture Progression Across Notebooks

#### Notebook 3: Single Agent, Single Tool

```
                 +-----------+
User Query ----->| agent_run |-----> SearchResults
                 |  (GPT-5)  |
                 +-----+-----+
                       |
              tool call: search_wayfair()
                       |
                 +-----v-----+
                 |   BM25    |
                 | (title +  |
                 | description)|
                 +-----------+
```

Key design decisions:
- **Query interpretation**: A separate GPT-5 call expands terse queries (e.g., "led 60" becomes "I am looking for an LED light fixture, 60 inches wide...")
- **Few-shot calibration**: 10 labeled examples (3 Exact, 3 Irrelevant, 4 Partial) are included in the system prompt
- **Error recovery**: If the agent returns the wrong number of results or invalid doc_ids, the harness injects a user-role error message and retries

#### Notebook 3a: Steer and Guide (Two Feedback Mechanisms)

```
                 +-----------+
User Query ----->|  search() |----> (validate)----> grade results ----> feedback
                 |  harness  |<---- (inject graded feedback as user msg) <---+
                 +-----+-----+           x5 attempts
                       |
              tool call: search_wayfair()
                       |
                 +-----v-----+         +---------------+
                 |   BM25    |<------->| SearchTracker  |
                 | + category|         | (MiniLM embed) |
                 +-----------+         +---------------+
                                       Rejects duplicate queries
```

Two new components:
1. **SearchTracker**: Uses `all-MiniLM-L6-v2` sentence embeddings to compute cosine similarity between the new query and all past queries. If similarity >= 0.95, the tool returns an error instead of results, forcing the agent to try different keywords.
2. **Graded feedback loop**: After each agent response, the harness grades results against human judgments using emoji (frowning face = bad, neutral face = partial, happy face = good) and sends the graded summary back as a user message. The agent gets 5 attempts to improve.

#### Notebook 3b: Multi-Agent Orchestration

```
                        +--------------------+
User Query ------------>| orchestrate_search |
                        +--------+-----------+
                                 |
                 +---------------+---------------+
                 |               |               |
          +------v------+ +-----v-------+ +-----v--------+
          | Sub-Agent 1 | | Sub-Agent 2 | | Sub-Agent 3  |
          | BM25+Category| | BM25 only  | | Embedding    |
          | (5 rounds)  | | (5 rounds)  | | (5 rounds)   |
          +------+------+ +-----+-------+ +-----+--------+
                 |               |               |
                 +-------+-------+-------+-------+
                         |
                  +------v------+
                  |  Aggregate  |
                  | dedup+sort  |
                  | by grade    |
                  +------+------+
                         |
                         v
                  Final Top-K Results
```

Three separate search tools, each given to an isolated sub-agent:
1. **`search_with_category`**: BM25 keyword search with optional category filtering
2. **`search_with_keywords`**: BM25 keyword search without category filtering
3. **`search_embeddings`**: Semantic vector search using MiniLM embeddings against pre-computed product embeddings

Each sub-agent runs independently (fresh conversation context) with 5 rounds of graded feedback. The orchestrator concatenates all results, deduplicates by doc_id, sorts by human-judgment grade, and takes the top-k.

### Why Context Isolation Matters

In 3b, each sub-agent gets a fresh `inputs` list (conversation history). This is deliberate:
- Keyword search feedback should not bias the embedding search agent's strategy
- Each tool has different strengths -- category-filtered search is narrow and precise, keyword search is broader, embedding search captures semantic meaning
- Isolation prevents the "echo chamber" effect where one poor strategy's failure messages pollute another strategy's reasoning

---

## Phase 3 - Detailed Component Analysis

### 3.1 Data Loading and Indexing

#### Loading the Wayfair Corpus

All three notebooks start the same way:

```python
from cheat_at_search.wands_data import corpus

corpus['category'] = corpus['category'].str.strip()
corpus['sub_category'] = corpus['sub_category'].str.strip()
```

`corpus` is a pandas DataFrame with 42,994 rows and columns including `product_id`, `product_name`, `product_description`, `product_features`, `category`, `sub_category`, `average_rating`, and more. The `.str.strip()` calls clean whitespace from category strings to prevent matching failures.

In Notebook 3b, an additional import appears:

```python
from cheat_at_search.wands_data import corpus, product_embeddings, judgments
```

`product_embeddings` is a pre-computed matrix of MiniLM sentence embeddings for all products, enabling the embedding search tool.

#### Building the BM25 Index

```python
from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer

corpus['title_snowball'] = SearchArray.index(corpus['title'].fillna(''), snowball_tokenizer)
corpus['description_snowball'] = SearchArray.index(corpus['description'].fillna(''), snowball_tokenizer)
corpus['category_snowball'] = SearchArray.index(corpus['category'].fillna(''), snowball_tokenizer)
```

`SearchArray` is a pandas extension type that stores an inverted index directly inside a DataFrame column. When you call `.array.score(term)`, it computes BM25 scores for that term across all 42,994 documents in one vectorized operation. The `snowball_tokenizer` applies Snowball stemming (e.g., "running" becomes "run", "furniture" becomes "furnitur") so that morphological variants match.

The `.fillna('')` prevents `NaN` values from causing indexing errors -- some products have empty descriptions.

### 3.2 The Search Functions (Tools)

#### Notebook 3: `search_wayfair()` (Baseline)

```python
def search_wayfair(keywords: str,
                   category: Optional[Categories] = None,
                   top_k: int = 5
                   ) -> list[dict[str, Union[str, int, float]]]:
```

This function is the heart of all three notebooks. The key design choices:

**Title boost (10x)**: Title matches score 10 times higher than description matches. This reflects the reality that product titles are carefully written to contain the most important keywords.

```python
bm25_scores = np.zeros(len(corpus))
for term in snowball_tokenizer(keywords):
    bm25_scores += corpus['title_snowball'].array.score(term) * 10
    bm25_scores += corpus['description_snowball'].array.score(term) * 1
```

**Category filtering as a mask**: When a category is provided, the function computes a boolean mask from the category index and multiplies it with the BM25 scores. Products outside the category get zeroed out.

```python
if category:
    cat_tokenized = snowball_tokenizer(category)
    category_mask = corpus['category_snowball'].array.score(cat_tokenized) > 0
    bm25_scores = bm25_scores * category_mask
```

**Top-k selection via argsort**:

```python
top_k_indices = np.argsort(bm25_scores)[-top_k:][::-1]
```

`np.argsort` sorts in ascending order, so `[-top_k:]` takes the last (highest) k elements, and `[::-1]` reverses to descending order.

#### The `Categories` Literal Type

```python
Categories = Literal['Furniture', 'Kitchen & Tabletop', 'Browse By Brand', ...]
```

This is not just documentation -- it is part of the LLM's prompt. When `make_tool_adapter()` converts this function into an OpenAI tool specification, the `Literal` type becomes an `enum` in the JSON schema. The LLM sees the exact list of valid categories and is constrained to choose only from that list. This prevents hallucinated category names.

#### Notebook 3a: `search_wayfair()` with `agent_state`

The 3a version adds two parameters:

```python
def search_wayfair(keywords: str,
                   category: Optional[Categories] = None,
                   top_k: int = 5,
                   agent_state: Optional[dict] = None
                   ) -> ToolSearchResults:
```

**`agent_state`** is a dictionary passed from the harness into the tool. It carries mutable state across tool invocations without polluting the LLM's conversation context. The tool uses it to store:

- `agent_state['search_tracker']`: A dictionary mapping category names to `SearchTracker` instances
- `agent_state['log']`: A list of `(keywords, category, was_skipped)` tuples for observability

The duplicate detection logic:

```python
duplicate_search = search_tracker.similar_search(keywords)
if duplicate_search is not None:
    error_msg = f"You searched '{keywords}', but you already searched for very similar '{duplicate_search}'..."
    return ToolSearchResults(search_results=[], error=error_msg)
```

When a duplicate is detected, the tool returns an error instead of results. The LLM sees this error message and must try a different approach.

#### Notebook 3b: Three Specialized Search Functions

**`search_with_category()`** -- identical to 3a's search, BM25 with optional category filter.

**`search_with_keywords()`** -- BM25 without category filtering. Simpler, broader results.

**`search_embeddings()`** -- Semantic vector search:

```python
def search_embeddings(keywords: str,
                      top_k: int = 5,
                      agent_state: Optional[dict] = None) -> ToolSearchResults:
    query_embedding = minilm.encode(keywords, convert_to_numpy=True)
    similarities = np.dot(product_embeddings, query_embedding)
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    ...
```

This encodes the query using `all-MiniLM-L6-v2`, computes dot-product similarity against all 42,994 pre-computed product embeddings, and returns the top-k. Unlike BM25, embedding search can find products that use entirely different words but have similar meaning (e.g., "cozy reading nook chair" matching "accent armchair").

### 3.3 The Tool Adapter System

#### `make_tool_adapter()`

```python
from cheat_at_search.agent.pydantize import make_tool_adapter

search_tool = make_tool_adapter(search_wayfair)
tool_info = {search_wayfair.__name__: search_tool}
```

`make_tool_adapter()` performs a critical transformation: it takes a regular Python function and produces a 3-tuple:

1. **`ToolArgsModel`** (Pydantic model): A dynamically created Pydantic model whose fields mirror the function's parameters. Used to validate and deserialize the LLM's JSON arguments.
2. **`tool_spec`** (dict): An OpenAI-compatible tool specification -- the function name, docstring as description, and parameter types encoded as JSON Schema.
3. **`tool_fn`** (callable): A wrapper that accepts a Pydantic model instance, unpacks it into the original function's arguments, calls the function, and returns both the Python object and a JSON-serialized string.

This means the function's **name**, **docstring**, **type annotations**, and **parameter names** all become part of the LLM's prompt. Writing clear, descriptive type hints is not just good practice -- it directly improves the agent's ability to use the tool correctly.

#### `call_tool()` (the dispatcher)

```python
def call_tool(tool_info, item) -> dict:
    tool_name = item.name                                    # LLM says which tool
    tool = tool_info[tool_name]
    ToolArgsModel = tool[0]
    tool_fn = tool[2]
    fn_args = ToolArgsModel.model_validate_json(item.arguments)  # Deserialize JSON -> Pydantic
    py_resp, json_resp = tool_fn(fn_args)                    # Call the actual function
    return {
        "type": "function_call_output",
        "call_id": item.call_id,                             # Match response to request
        "output": json_resp,
    }
```

In Notebooks 3a and 3b, `call_tool()` also passes `agent_state`:

```python
def call_tool(tool_info, agent_state, item) -> dict:
    ...
    py_resp, json_resp = tool_fn(fn_args, agent_state=agent_state)
    ...
```

### 3.4 The SearchTracker Class (Notebooks 3a, 3b)

```python
from sentence_transformers import SentenceTransformer

minilm = SentenceTransformer('all-MiniLM-L6-v2')

class SearchTracker:
    def __init__(self, similarity_threshold=0.95):
        self.queries = []
        self.query_embeddings = []
        self.similarity_threshold = similarity_threshold

    def similar_search(self, new_query: str) -> Optional[str]:
        new_embedding = minilm.encode(new_query, convert_to_numpy=True)
        for i, existing_embedding in enumerate(self.query_embeddings):
            similarity = np.dot(new_embedding, existing_embedding) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
            )
            if similarity >= self.similarity_threshold:
                return self.queries[i]  # Return the original query it duplicates
        self.queries.append(new_query)
        self.query_embeddings.append(new_embedding)
        return None  # No duplicate found
```

**How it works:**
1. Every new query is encoded into a 384-dimensional vector using MiniLM
2. Cosine similarity is computed against all previously seen queries
3. If any similarity >= 0.95, the function returns the text of the matching past query (indicating a duplicate)
4. If no match, the query is added to the tracker and `None` is returned

**Why per-category?** In 3a and 3b, there is one `SearchTracker` per category. Searching "modern chair" in the "Furniture" category and "modern chair" in "Outdoor" are treated as different searches because they return different products. The tracker correctly allows both.

**Why 0.95 threshold?** This is very strict -- "red couch" and "red sofa" have a similarity around 0.85-0.90, so both are allowed. Only near-identical reformulations like "red couch" and "red couches" are blocked.

### 3.5 The Agent Loop: `agent_run()`

This is the core LLM execution loop, shared (with minor variations) across all three notebooks:

```python
def agent_run(tool_info, text_format, inputs, model='gpt-5', summary=True):
    tool_calls = True
    resp = None
    while tool_calls:
        # Retry logic for API errors
        failing = True
        num_failures = 0
        while failing:
            try:
                resp = openai.responses.parse(
                    model=model,
                    input=inputs,
                    tools=[tool[1] for tool in tool_info.values()],
                    reasoning={"effort": "medium", "summary": "auto" if summary else "none"},
                    text_format=text_format
                )
                failing = False
            except Exception:
                failing = True
                num_failures += 1
                if num_failures > 3:
                    raise
                sleep(1)

        inputs += resp.output  # Append model output to conversation
        # ... print reasoning summaries and token usage ...

        # Check if the model wants to call more tools
        for item in resp.output:
            tool_calls = False
            if item.type == "function_call":
                tool_calls = True
                tool_response = call_tool(tool_info, item)
                inputs.append(tool_response)
    return resp, inputs
```

**Key OpenAI API features used:**

- **`openai.responses.parse()`**: The newer OpenAI "responses" API (vs. the older "chat completions"). Supports tool calling, structured output, and reasoning natively.
- **`text_format=SearchResults`**: Forces the LLM's final text output to conform to the `SearchResults` Pydantic model. The response is automatically parsed into `resp.output_parsed`.
- **`reasoning={"effort": "medium"}`**: Enables the model's chain-of-thought reasoning. "medium" is a balance between quality and token cost. The `"summary": "auto"` setting returns human-readable reasoning summaries.
- **`tools=[...]`**: List of tool specifications. The model decides when and how to call them.

**The loop pattern**: The outer `while tool_calls` loop continues as long as the model emits `function_call` outputs. In a single iteration, the model might make zero, one, or several tool calls. Each tool call response is appended to `inputs`, and the model is called again to process the new information. The loop terminates when the model produces its final text output (the `SearchResults`).

### 3.6 The Search Harness: `search()`

#### Notebook 3: Validation-Only Harness

```python
def search(query):
    inputs = [{"role": "system", "content": system_prompt},
              {"role": "user", "content": interpret_query(query)}]
    error = True
    while error:
        resp, inputs = agent_run(tool_info, text_format=SearchResults, inputs=inputs)
        # Validate: exactly 10 results
        if len(resp.output_parsed.ranked_results) != 10:
            inputs.append(_error_msg(f"Expected 10 ranked_results, got {num_results}"))
            continue
        # Validate: all doc_ids exist in corpus
        for item in resp.output_parsed.ranked_results:
            if item not in corpus['doc_id'].values:
                inputs.append(_error_msg(f"Doc id {item} is not in corpus"))
                continue
        error = False
    return resp.output_parsed
```

The `_error_msg()` function constructs a user-role message: `"Oh this isn't good, it turns out: {error}. Please try again"`. This conversational framing is deliberate -- it keeps the model in its helpful assistant persona rather than triggering defensive behavior.

#### Notebook 3a: Feedback Loop Harness

```python
def search(query, num_attempts=5):
    inputs = [{"role": "system", "content": system_prompt},
              {"role": "user", "content": query}]
    agent_state = {}
    attempt = 0
    while error or attempt < num_attempts:
        resp, inputs = agent_run(tool_info, text_format=SearchResults,
                                 inputs=inputs, agent_state=agent_state)
        # ... validation ...
        graded_results = grades(query, resp.output_parsed)
        graded_summary = "Here are your results reflecting whether they satisfy the user:\n"
        for title, doc_id, grade in graded_results:
            graded_summary += f"\n{title} (doc_id:{doc_id}) {grade}"
        graded_summary += "\n\nGiven this, try some additional searches..."
        inputs.append({'role': 'user', 'content': graded_summary})
        attempt += 1
    return resp.output_parsed
```

The crucial difference: after each attempt, the harness **grades every result** using human judgments (WANDS labels), formats them with emoji, and injects that feedback into the conversation. The LLM then reasons about which results are good and which need replacing.

#### Notebook 3b: Multi-Agent Orchestrator

```python
def orchestrate_search(keywords, top_k=10):
    # Sub-agent 1: BM25 + category (5 rounds)
    inputs = _reset_inputs(system_prompt, keywords)
    for attempt in range(5):
        resp_cat, labeled_cat = search(keywords, inputs, search_with_category)
        inputs = _label_resp(inputs, labeled_cat)

    # Sub-agent 2: BM25 only (5 rounds)
    inputs = _reset_inputs(system_prompt, keywords)  # Fresh context!
    for attempt in range(5):
        resp_lex, labeled_lex = search(keywords, inputs, search_with_keywords)
        inputs = _label_resp(inputs, labeled_lex)

    # Sub-agent 3: Embedding search (5 rounds)
    inputs = _reset_inputs(system_prompt, keywords)  # Fresh context!
    for attempt in range(5):
        resp_emb, labeled_emb = search(keywords, inputs, search_embeddings)
        inputs = _label_resp(inputs, labeled_emb)

    # Aggregate: concat, dedup, sort by grade
    all_results = merge_results(labeled_cat, merge_results(labeled_lex, labeled_emb))
    # ... sort by emoji score (happy > neutral > sad), take top_k ...
```

`_reset_inputs()` creates a fresh system prompt + user message. This is the **context isolation** -- each sub-agent starts clean, unaware of the other agents' searches or failures.

`merge_results()` deduplicates by `doc_id`, keeping the entry with the better grade when the same product appears from multiple search strategies.

### 3.7 Grading and Feedback (Notebooks 3a, 3b)

```python
def _grade_to_emoji(grade):
    if grade == 0: return '☹️'   # Irrelevant
    elif grade == 1: return '😑'  # Partial
    elif grade == 2: return '😃'  # Exact match
    return '☹️'

def grades(query, search_results):
    query_judgments = labeled_query_products[labeled_query_products['query'] == query]
    r_value = []
    for doc_id in search_results.ranked_results:
        title = corpus[corpus['doc_id'] == doc_id]['title'].iloc[0]
        doc_judgments = query_judgments[query_judgments['doc_id'] == doc_id]
        if len(doc_judgments) == 0:
            r_value.append((title, doc_id, _grade_to_emoji(None)))
        else:
            r_value.append((title, doc_id, _grade_to_emoji(int(doc_judgments['grade'].values[0]))))
    return r_value
```

The grading system maps WANDS human labels to emoji. The emoji choice is intentional: research suggests that LLMs respond well to emotional valence. A frowning face is a stronger negative signal than a numeric 0.

Importantly, the system prompt in 3a and 3b explicitly explains the emoji scale:

```
☹️ - ACTIVELY FRUSTRATES USER, do not include unless absolutely necessary
😑 - Meh results. OK in a pinch. But there could be better.
😃 - Solves users problem. Good job! Rank this highest
```

### 3.8 Query Interpretation (Notebook 3 Only)

```python
def interpret_query(keywords):
    system_prompt = """
        Interpret search queries for a home goods / furniture
        store like Wayfair into a description of what's needed...
        State it in the voice of the user "I am looking for <detailed info>"
    """
    resp = openai.responses.create(model="gpt-5", input=inputs)
    return resp.output[-1].content[-1].text
```

This is a **separate LLM call** (not the agent) that converts a terse query like "led 60" into "I am looking for an LED light fixture, approximately 60 inches wide or with 60 watts of power, suitable for home installation." The expanded query gives the search agent much better starting conditions.

This technique does not appear in Notebooks 3a and 3b, which instead rely on the feedback loop to compensate for terse queries.

### 3.9 Few-Shot Prompt Construction (Notebook 3)

```python
def build_few_shot_prompt(prompt, k=10) -> str:
    relevant = labeled[labeled['label'] == 'Exact'].sample(min(k // 3, ...), random_state=42)
    irrelevant = labeled[labeled['label'] == 'Irrelevant'].sample(min(k // 3, ...), random_state=42)
    partial = labeled[labeled['label'] == 'Partial'].sample(...)
    labeled = pd.concat([relevant, irrelevant, partial]).sample(frac=1, random_state=42)
    for item in labeled.to_dict(orient='records'):
        prompt += f"""
        User Query: {item['query']}
        Product Name: {item['title']}
        Human Label: {item['label']}
        """
    return prompt
```

This builds a balanced sample of labeled examples: roughly 1/3 "Exact" matches, 1/3 "Irrelevant", and the rest "Partial". The examples are shuffled and appended to the system prompt. This calibrates the LLM's sense of what humans consider relevant in the Wayfair domain -- a form of **in-context learning**.

### 3.10 Evaluation Framework

#### `SearchResults` (Structured Output Schema)

```python
class SearchResults(BaseModel):
    """The ranked, top 10 search results ordered most relevant to least."""
    results_summary: str = Field(description="The message from you summarizing what you found")
    ranked_results: list[int] = Field(description="Top ranked search results (their doc_ids)")
```

This Pydantic model is passed as `text_format` to `openai.responses.parse()`. The LLM is forced to return valid JSON matching this schema. `ranked_results` is a list of integer doc_ids, ordered from most to least relevant.

#### `AgenticSearchStrategy` (Evaluation Wrapper)

```python
class AgenticSearchStrategy(SearchStrategy):
    def search(self, query, k):
        resp = search(query)
        return (resp.ranked_results, [1.0] * len(resp.ranked_results))
```

This wraps the notebook's `search()` function into the library's `SearchStrategy` interface, which `run_strategy()` expects. The `[1.0] * len(...)` provides uniform scores since the agent's ranking is the signal, not individual scores.

#### NDCG Evaluation

```python
from cheat_at_search.search import run_strategy, ndcgs, graded_bm25

seed = 1234
np.random.seed(seed)
random_queries = np.random.choice(judgments['query'].unique(), 8)
selected_judgments = judgments[judgments['query'].isin(random_queries)]

strategy = AgenticSearchStrategy(corpus, workers=1)
graded_agentic = run_strategy(strategy, selected_judgments)

# Compare
ndcgs(graded_agentic).mean()      # e.g., 0.396
ndcgs(graded_bm25[...]).mean()    # e.g., 0.342
```

NDCG (Normalized Discounted Cumulative Gain) is the standard metric. It rewards putting the most relevant results at the top of the list. 1.0 is perfect ranking; higher is better. The seed ensures reproducible query selection across runs.

### 3.11 Tool Return Types (Pydantic Models)

#### Notebook 3: Plain Dictionaries

```python
results.append({
    'id': row['doc_id'],
    'title': row['title'],
    'description': row['description'],
    'category': row['category'],
    'score': row['score']
})
```

The baseline uses simple dictionaries as tool return values.

#### Notebooks 3a and 3b: Pydantic Models

```python
class ToolSearchResult(BaseModel):
    id: int = Field(description="The id of the product")
    title: str = Field(description="The title of the product")
    description: str = Field(description="The description of the product")
    category: str = Field(description="The category of the product")
    score: float = Field(description="The score of the product")

class ToolSearchResults(BaseModel):
    search_results: list[ToolSearchResult]
    error: Optional[str] = None
```

The upgrade to Pydantic models provides:
- **Schema enforcement**: The tool's return type is part of the OpenAI function spec
- **Error channel**: The `error` field allows tools to report issues (duplicate queries) within the structured response, rather than raising exceptions
- **Consistent serialization**: `model_dump_json()` produces clean JSON every time

---

## Phase 4 - Python Pattern Explanations

### 4.1 Pydantic Models for Structured Data

**What it is**: Pydantic is a Python library for data validation using type annotations. You define a class inheriting from `BaseModel`, and Pydantic automatically validates, serializes, and deserializes data.

**Where it appears**: Throughout all three notebooks for search results, tool arguments, and LLM output schemas.

```python
from pydantic import BaseModel, Field

class SearchResults(BaseModel):
    """The ranked, top 10 search results ordered most relevant to least."""
    results_summary: str = Field(description="...")
    ranked_results: list[int] = Field(description="...")
```

**Why it is used**: OpenAI's structured output feature requires a JSON schema. Pydantic models generate this schema automatically from the class definition. Additionally, `model_validate_json()` parses JSON strings into validated Python objects, and `model_dump_json()` serializes back. The `Field(description=...)` metadata flows into the JSON schema and becomes part of the LLM's prompt.

**Simple example**:

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float

p = Product(name="Chair", price=199.99)       # Validated on creation
p_json = p.model_dump_json()                   # '{"name":"Chair","price":199.99}'
p2 = Product.model_validate_json(p_json)       # Parse JSON -> Product
```

### 4.2 Literal Types for Constrained Values

**What it is**: `typing.Literal` restricts a parameter to a fixed set of values. It is like an enum but lighter -- no class definition needed.

```python
from typing import Literal

Categories = Literal['Furniture', 'Kitchen & Tabletop', 'Outdoor', ...]
```

**Why it is used**: When this type annotation appears in a function that becomes an OpenAI tool, the adapter converts it to a JSON Schema `enum`. The LLM sees the exact list of valid values and cannot invent categories. This prevents hallucinated category names that would silently return zero results.

**Simple example**:

```python
from typing import Literal

Color = Literal['red', 'green', 'blue']

def paint(wall: str, color: Color):
    print(f"Painting {wall} {color}")

paint("kitchen", "red")    # OK
paint("kitchen", "orange") # Type checker error (runtime: no error without validation)
```

### 4.3 Optional Parameters with Default Values

**What it is**: `Optional[T]` means the value can be `T` or `None`. Combined with `= None` as the default, it creates a parameter the caller can omit.

```python
from typing import Optional

def search_wayfair(keywords: str,
                   category: Optional[Categories] = None,
                   top_k: int = 5) -> list:
```

**Why it is used**: The LLM agent can choose whether to filter by category. If the query is broad (e.g., "red rug"), the agent might omit the category. If the query is specific (e.g., "outdoor patio dining set"), the agent might pass `category="Outdoor"`. The `Optional` annotation appears in the tool schema as a non-required parameter.

### 4.4 NumPy Vectorized Operations for BM25

**What it is**: Instead of looping over 42,994 documents one by one, NumPy performs operations on entire arrays at once using optimized C code.

```python
bm25_scores = np.zeros(len(corpus))              # Array of 42994 zeros
for term in snowball_tokenizer(keywords):
    bm25_scores += corpus['title_snowball'].array.score(term) * 10  # Score ALL docs at once
    bm25_scores += corpus['description_snowball'].array.score(term) * 1
```

**Why it is used**: Performance. Computing BM25 scores for 42,994 documents with a Python `for` loop over each document would take seconds. The vectorized approach takes milliseconds because `SearchArray.score()` computes all 42,994 scores in one call, returning a NumPy array.

**The argsort pattern**:

```python
top_k_indices = np.argsort(bm25_scores)[-top_k:][::-1]
```

This is a common NumPy idiom: `argsort()` returns indices that would sort the array in ascending order. `[-top_k:]` takes the last k (highest-scoring) indices. `[::-1]` reverses to descending order. The result: indices of the top-k highest-scoring documents.

### 4.5 The Tool-Calling Loop Pattern

**What it is**: A `while` loop that alternates between calling the LLM and executing tools until the LLM stops requesting tool calls.

```python
tool_calls = True
while tool_calls:
    resp = openai.responses.parse(model=model, input=inputs, tools=tools, ...)
    inputs += resp.output
    for item in resp.output:
        tool_calls = False
        if item.type == "function_call":
            tool_calls = True
            tool_response = call_tool(tool_info, item)
            inputs.append(tool_response)
```

**Why it is used**: LLM tool calling is inherently iterative. The model might:
1. First call search with "geometric couch" and get results
2. Then call search with "modern sofa" to explore alternatives
3. Then call search with "contemporary loveseat" in the "Furniture" category
4. Finally produce its structured output with ranked results

Each iteration appends the model's output and the tool responses to `inputs`, building up the conversation history. The model sees all prior searches and their results when deciding what to do next.

### 4.6 Conversation History as Mutable List

**What it is**: The `inputs` variable is a Python list of dictionaries, each representing a message (system, user, assistant, or tool response). It is mutated in-place across the loop.

```python
inputs = [{"role": "system", "content": system_prompt},
          {"role": "user", "content": query}]
# ... after agent_run:
inputs += resp.output           # Append model output
inputs.append(tool_response)    # Append tool response
inputs.append({'role': 'user', 'content': feedback})  # Append feedback
```

**Why it matters**: The entire conversation history is sent to the LLM on every call. This means:
- The model can see all previous tool calls and their results
- Feedback from the harness (validation errors, graded results) becomes part of the conversation
- The history grows with each iteration, increasing token usage

In Notebook 3b, `_reset_inputs()` creates a fresh list for each sub-agent, achieving context isolation.

### 4.7 Retry with Exponential Backoff (Simplified)

```python
failing = True
num_failures = 0
while failing:
    try:
        resp = openai.responses.parse(...)
        failing = False
    except Exception:
        failing = True
        num_failures += 1
        if num_failures > 3:
            raise
        sleep(1)
```

**What it is**: A retry pattern that catches API errors (rate limits, timeouts, server errors) and retries up to 3 times with a 1-second delay.

**Why it is used**: LLM API calls are network requests that can fail for transient reasons. The retry logic makes the notebooks more robust in Colab environments with variable network quality.

### 4.8 Boolean Masking in NumPy/Pandas

```python
category_mask = corpus['category_snowball'].array.score(cat_tokenized) > 0
bm25_scores = bm25_scores * category_mask
```

**What it is**: `> 0` produces a boolean array (`[True, False, True, ...]`). Multiplying a numeric array by a boolean array converts `True` to 1 and `False` to 0, effectively zeroing out non-matching entries.

**Why it is used**: This is an efficient way to filter without creating a new DataFrame. The BM25 scores for products outside the category become 0, so they cannot appear in the top-k results. No copying, no index alignment -- just element-wise multiplication.

### 4.9 Cosine Similarity via Dot Product and Norms

```python
similarity = np.dot(new_embedding, existing_embedding) / (
    np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
)
```

**What it is**: Cosine similarity measures the angle between two vectors, ignoring magnitude. Two vectors pointing in the same direction have similarity 1.0; perpendicular vectors have 0.0; opposite vectors have -1.0.

**Why it is used in `SearchTracker`**: Two queries like "red leather couch" and "red leather sofa" will have embeddings pointing in nearly the same direction (similarity ~0.92). The tracker uses a high threshold (0.95) to catch only near-identical reformulations.

### 4.10 Class Inheritance for Strategy Pattern

```python
from cheat_at_search.strategy import SearchStrategy

class AgenticSearchStrategy(SearchStrategy):
    def search(self, query, k):
        resp = search(query)
        return (resp.ranked_results, [1.0] * len(resp.ranked_results))
```

**What it is**: The Strategy design pattern -- defining a common interface (`SearchStrategy`) and plugging in different implementations. The `run_strategy()` evaluation function works with any `SearchStrategy` subclass.

**Why it is used**: The library can evaluate BM25, agentic search, and any future strategy using the same `run_strategy()` pipeline. You swap the strategy object; the evaluation code stays unchanged.

### 4.11 Dictionary as Mutable Shared State (`agent_state`)

```python
agent_state = {}
# Inside search tool:
agent_state['search_tracker'] = search_trackers  # Mutates the dict
agent_state['log'] = query_log                   # Mutates the dict
```

**What it is**: In Python, dictionaries are mutable reference types. Passing a dict to a function gives the function a reference to the same object -- changes inside the function are visible to the caller.

**Why it is used**: The agent loop calls the LLM, which triggers tool calls, which execute Python functions. The tool functions need to share state (tracked queries, logs) across invocations, but this state should not be visible to the LLM. The `agent_state` dict provides this "hidden channel" between the harness and the tools, bypassing the conversation history.

### 4.12 Lazy Module Loading via `__getattr__`

The library uses a Python pattern for deferred loading:

```python
# In wands_data.py
def __getattr__(name):
    if name == "corpus":
        return _load_corpus()
    raise AttributeError(f"module has no attribute {name}")
```

**What it is**: When you import `from cheat_at_search.wands_data import corpus`, Python calls `__getattr__("corpus")` on the module. The data is loaded from disk only at this point, not when the module is first imported.

**Why it is used**: The WANDS corpus is large. Lazy loading means importing the module is instant; the actual data loading happens only when you access the specific attribute you need.

---

## Cross-Week Meta-Patterns and Progression

### The Three-Week Arc

The notebooks across all three weeks tell a coherent story about improving search quality with LLMs:

| Week | Theme | Key Question |
|------|-------|-------------|
| **Week 1** | Foundations | Can an LLM interact with a search engine? (Chat loop -> RAG -> Agentic loop) |
| **Week 2** | Query Understanding | Can an LLM improve the *query* before searching? (Synonyms, categorization) |
| **Week 3** | Agent Autonomy | Can an LLM autonomously *strategize* about searching? (Feedback, multi-tool, orchestration) |

### Recurring Design Patterns Across Weeks

**1. BM25 as the universal baseline**: Every notebook measures improvement against BM25. The BM25 implementation (title 10x boost, description 1x) stays constant, providing a stable comparison point.

**2. Structured output everywhere**: From Week 1's basic `SearchRequest` to Week 3's `SearchResults`, every LLM interaction uses Pydantic models as output schemas. The evolution:
- Week 1: `SearchRequest(search_query, categories)` -- the LLM generates a search query
- Week 2: `QueryWithSynonyms`, `QueryCategory` -- the LLM enriches queries
- Week 3: `SearchResults(ranked_results)` -- the LLM produces final rankings

**3. Incremental complexity**: Each notebook adds one new concept:
- 1 -> 1a: Add RAG (retrieve then generate)
- 1a -> 1b: Add tool calling (agent decides when to search)
- 1b -> 3: Add structured output + validation loop
- 3 -> 3a: Add feedback + duplicate detection
- 3a -> 3b: Add multiple tools + orchestration

**4. The tool = Python function pattern**: In every agent notebook, a Python function with type hints and a docstring becomes an LLM tool via `make_tool_adapter()`. The function signature is the interface contract between the harness and the LLM.

**5. NDCG as the North Star metric**: Every experiment ends with `ndcgs(graded_results).mean()` compared to the BM25 baseline. This grounds all the LLM-based improvements in a quantitative measure.

### The Progression of Agent Sophistication

```
Week 1: LLM generates ONE query -> search -> present results
           (no iteration, no self-correction)

Week 2: LLM improves the QUERY before searching
           (synonyms, categories -- still single-shot search)

Week 3 Baseline: LLM uses search MULTIPLE TIMES with different keywords
           (iterative, but no external feedback)

Week 3 Steer: LLM gets FEEDBACK on results and adjusts
           (closed-loop control with graded evaluation)

Week 3 Extra: MULTIPLE AGENTS with different tools, ORCHESTRATED
           (ensemble of strategies with aggregation)
```

### What Changes Between Notebooks and What Stays Fixed

**Fixed across all Week 3 notebooks:**
- The Wayfair WANDS dataset (42,994 products)
- BM25 scoring formula (title 10x, description 1x, Snowball stemming)
- The evaluation framework (`run_strategy`, `ndcgs`, same 8 random queries with seed 1234)
- The `SearchResults` Pydantic output schema
- The OpenAI Responses API calling pattern

**What evolves:**
- Number of search tools: 1 -> 1 -> 3
- Feedback mechanism: none -> emoji grading -> emoji grading per sub-agent
- Agent state: none -> SearchTracker -> SearchTracker per tool
- LLM model: GPT-5 (baseline) -> GPT-5-mini (3a, 3b) -- smaller model with feedback compensates
- Query preprocessing: interpret_query (baseline) -> removed in 3a/3b (feedback loop replaces it)
- Context management: single shared context -> isolated contexts per sub-agent

### Key Insight: Feedback Replaces Prompt Engineering

A notable pattern: Notebook 3 invests in better *starting conditions* (query interpretation, few-shot examples) to get good results in one shot. Notebooks 3a and 3b drop these in favor of *iterative feedback*. The implicit lesson: giving an agent the ability to try, fail, and improve is more powerful than trying to get the perfect first prompt.

This mirrors a broader trend in AI systems: moving from "get it right the first time" to "get it right eventually through iteration."
