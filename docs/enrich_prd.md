# Enrichment functionality

We use "enrich" to take queries, documents etc and ask an LLM (typically OpenAI) to enrich them in some way. This is a broad category that can include things like:

- Query expansion: taking a query and asking the LLM to expand it with related terms or concepts.
- Query categorization: taking a query and asking the LLM to categorize it into one or more categories.
- Document enrichment: taking a document and asking the LLM to extract key information, summarize it, or generate metadata.


## Enrichment clients

LLM enrichment works via the `LLMEnrichClient` interface. It accepts a prompt
and returns a structured Pydantic response:

`DecisionEnrichClient` is a parallel interface for constrained decisions. It
accepts a state, a question, and a mapping of choice keys to optional
descriptions. `score` returns a tuple containing the choice probability mapping
and confidence score. `decide` returns the highest-probability choice only when
its probability and confidence meet `probability_threshold` and
`confidence_threshold`; otherwise it returns `None`. Both thresholds are in
the range 0–1 and default to `0.0`. Unlike LLM enrichment, decision enrichment
does not use a Pydantic response model.


## Core Implementations

We implement OpenAI enrichment through `OpenAIEnricher`, which implements
`LLMEnrichClient`.

It takes a prompt and returns a structured response (the Pydantic BaseModel
instance).

It's noted that unlike other libraries, we're not trying to normalize acceptable schemas between these providers. We take their structured outputs as-is.

We implement Jev decisions through `JevEnrichClient`, which implements
`DecisionEnrichClient` using `TypeSafeClient.system_one`. It passes the supplied
state to Jev and turns the question and choices into a Typesafe `Choice`.


## Cached implementation

A wrapper implementation `CachedEnrichClient` stores LLM responses in a cache
for later reuse. `CachedDecisionEnrichClient` caches Jev choice probabilities
and confidence, allowing both `score` and thresholded `decide` to reuse the same
result. Its request key includes the client/model signature, thresholds, state,
question, and choices. State and choices are serialized as canonical JSON so
mapping insertion order does not change a key. Use `force_refresh=True` on
`score`, `decide`, or `decide_all` to bypass an existing entry.
Entries do not expire automatically; pin a Jev model version or refresh/delete
the cache when a moving model alias such as `jev-latest` changes.


## Auto enrichment helpers

`AutoLLMEnricher` wraps the OpenAI LLM client with caching and batch enrichment.
`AutoEnricher` remains an alias for `AutoLLMEnricher` for compatibility.

```python
enricher = AutoLLMEnricher(
      model="openai/gpt-4.1-nano",
      system_prompt="Your task is to create novel, never seen before, furniture, home goods, or hardware classification that best fit a search query. ",
      response_model=QueryClassification
)
```

`AutoDecisionEnricher` performs Jev decisions using a fixed set of choices. Its
model defaults to `jev/jev-latest`; a `jev/` model can be specified explicitly.
It accepts the same probability and confidence thresholds at construction.
`decide_all` accepts either one question to apply to every state or a parallel
sequence of questions, and preserves input state order:

```python
decision_enricher = AutoDecisionEnricher(
    choices={
        "furniture": "A furniture product",
        "lighting": "A lighting product",
    },
    probability_threshold=0.7,
    confidence_threshold=0.5,
)

category = decision_enricher.decide(
    state={"query": "red chair", "title": "Armchair"},
    question="Which product category best matches the query?",
)
probabilities, confidence = decision_enricher.score(
    state={"query": "red chair", "title": "Armchair"},
    question="Which product category best matches the query?",
)
categories = decision_enricher.decide_all(
    states=[{"query": "red chair"}, {"query": "desk lamp"}],
    questions="Which product category best matches the query?",
)
```

Then usually its used in a function that will return

```python
def get_prompt_fully_qualified(query):
        prompt = f"""

        Classify this query:

        {query}

        """

        return prompt

def fully_classified(query):
    prompt = get_prompt_fully_qualified(query)
    return enricher.enrich(prompt)
```

That's useful wrapper because now we can just apply this to dataframe of queries

```python
queries['classification'] = queries['query'].apply(fully_classified)
```

## Vocabulary enricher

One common enrichment use case is to map a query into a controlled vocabulary. We might do that by sending a large Literal to OpenAI. But we might also
do that by the following 2 step process

1. Instantiate an AutoLLMEnricher. Have it just return a str.
2. Compare cosine similarity of generated item to embeddings of the vocabulary

For example,

System prompt: You assist by extracting colors from queries
User prompt: Extract the color from this query: "crimson chair"
Vocaburaly: ["red", "blue", "green", "yellow", "black", "white"]

LLM returns "crimson"

Then we can compare the embedding of "crimson" to the embeddings of the vocabulary and return the closest match, likely "red".

The LLM may return no match, ie

System prompt: You assist by extracting colors from queries
User prompt: Extract the color from this query: "chair"
Vocaburaly: ["red", "blue", "green", "yellow", "black", "white"]

LLM returns [] or None or ""

So we don't try to resolve, and just return None

The VocabualryEnricher class wraps AutoLLMEnricher to provide this functionality.

It takes at construction

- Same params as AutoLLMEnricher
- A list of vocabulary items

Then it exposes a method

- `resolve(prompt: str) -> Optional[str]`

Whatever the LLM returns from prompt (should be a string) will be resolved to the closest vocaburaly item and returned.
