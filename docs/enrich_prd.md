# Enrichment functionality

We use "enrich" to take queries, documents etc and ask an LLM (typically OpenAI) to enrich them in some way. This is a broad category that can include things like:

- Query expansion: taking a query and asking the LLM to expand it with related terms or concepts.
- Query categorization: taking a query and asking the LLM to categorize it into one or more categories.
- Document enrichment: taking a document and asking the LLM to extract key information, summarize it, or generate metadata.


## Enrich client

Enrichment works via the "EnrichClient" interface.

Its assumed that an implementation calls some LLM and returns some structured response in a pydantic BaseModel, you see this with:

```python
class EnrichClient(ABC):

    def __init__(self, response_model: BaseModel):
        self.response_model = response_model

    @abstractmethod
    def enrich(self, prompt: str) -> Optional[BaseModel]:
        pass
```


## Core Implementations

We implement google / openai enrichments.

They take a prompt and return a structured response (the pydantic BaseModel instance)

It's noted that unlike other libraries, we're not trying to normalize acceptable schemas between these providers. We take their structured outputs as-is.


## Cached implementation

A wrapper implementation `CachedEnrichClient` stores the responses in a cache for later reuse.


## AutoEnricher

Much of the work to enrich is actually done via the AutoEnricher class. It wraps the openai / google in a cached enricher - instantiating depending on model type passed.

```
enricher = AutoEnricher(
     model="openai/gpt-4.1-nano",
     system_prompt="Your task is to create novel, never seen before, furniture, home goods, or hardware classification that best fit a search query. ",
     response_model=QueryClassification
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
