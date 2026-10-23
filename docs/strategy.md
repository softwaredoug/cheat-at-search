# SearchStrategy

A SearchStrategy is a class that runs a query, returns ranked results. It's used to
study different relevance approaches. IE a canonical strategy in this repo, is BM25,

```python
from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer

class BM25Search(SearchStrategy):
    def __init__(self, corpus,
                 title_boost=9.3,
                 description_boost=4.1):
        super().__init__(corpus)
        self.index = corpus
        self.title_boost = title_boost
        self.description_boost = description_boost

        if 'title_snowball' not in self.index and 'title' in corpus:
            self.index['title_snowball'] = SearchArray.index(
                corpus['title'], snowball_tokenizer)
        if 'description_snowball' not in self.index and 'description' in corpus:
            self.index['description_snowball'] = SearchArray.index(
                corpus['description'], snowball_tokenizer)

    def search(self, query, k=10):
        """Dumb baseline lexical search"""
        tokenized = snowball_tokenizer(query)
        bm25_scores = np.zeros(len(self.index))
        for token in tokenized:
            bm25_scores += self.index['title_snowball'].array.score(token) * self.title_boost
            bm25_scores += self.index['description_snowball'].array.score(
                token) * self.description_boost
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores
```

You can see above the general flow

- Possible tokenization
- Indexing using SearchArray into pandas SearchArray columns

Then later when searching,

- Tokenize the query
- Loop over the tokens and score them using the SearchArray index
- Return two arrays:
  - The top k indices of the results
  - The scores of those results

## `search_all` function

The base class `SearchStrategy` implements `search_all`. It takes a set of judgments and runs
queries, gathers results, and creates a larger dataframe of all the results concatt'd into
a DataFrame.

`search_all` may cache results optionally. Which is useful for long running / expensive processes
when we want to repeat their analysis.

`search_all` accepts a `round` value (default `0`), adds it to each result row,
and includes it in the cache identity. `k` is also part of the cache identity.

`run_strategy(..., rounds=R)` selects the query population once, then runs that
same selection for each round numbered `0` through `R - 1`. Each round is
searched and graded separately before the resulting dataframes are concatenated;
the returned rows therefore retain a `round` column and per-round metrics.

### `_search_all_single` serializes queries

The default strategy. Serially calls `search` one query at a time.

### `_search_all_batch` runs queries in parallel

- Used when the strategy implements `search_batch`. Which processes
multiple queries at once. Useful for embedding strategies that can batch queries into a single embedding call.

## Resource Reporting

"Resources" can mean things like latency, token usage, number of tool calls, number
of API calls, or anythnig we often want to minimize the usage of.

Some strategies might use resources. And when we study their search performance, we also
want to study how their resource usage.

Strategies can optionally also report resource usage on a query. If a `search` method
returns a 3rd item in the tuple, that's assumed to be a dictionary of resource usage.

The resource dictionary is a string -> float | int dictionary

In the `search_all` dataframe being computed, each key in the resource dictionary becomes a column,

ie if the dictionary for query "red shoes" is

{
"latency": 0.2, "tokens": 100
}

Then the resulting dataframe will look something like

| query     | doc_id | ... | latency | tokens |
|-----------|--------|-----|---------|--------|
| red shoes | 123    | ... | 0.2     | 100    |
| red shoes | 1234   | ... | 0.2     | 100    |

We repeat the resource per row for that query. And of course anotherh query would have a different
resource.

Importantly, this is cached along with the results. So later, if you rerun the strategy, you get
the cached resource usage statistics to help your analyzis
