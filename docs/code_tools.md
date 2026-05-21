# Code tools

This doc defines the various code tools and their usages

## Context: autoresearch

A user incorporates the tools in an autoresearch loop to improve ranking code to produce more relevant search results

## Training vs validation data

Queries can be split into two sets

1. Validation - used for evaluating search strategies and gating changes. If the validation does not improve, the change is not accepted.
2. Training - agent allowed full visibility into this data, and can use it for training or learning.

## Reranker class

The coding tools are implemented in a codegen class with methods corresponding to each tool.

This class exists to encapsulate coding functionality assuming a specific directory structure / file layout.

### Construction

Instantiate `Reranker` directly and then call `tools()` to expose the functions an agent can call. Parameters include:


- corpus: pd.Dataframe - the corpus to search over
- judgments: pd.Dataframe - ground truth evals used to generate training / validation sets
- code_dir: str -- The working directory of the code and any other artifacts
- tool_fns: list[callable] -- List of search primitives used in the reranker code
- module_name: str -- A name for the module being worked on (used for the filename of the code patch)
                     Its assumed a function exists with the same name as module_name
- guardrail_fns: List = None -- Guards executed before applying a patch. If fail, patch rejected with error
- training_queries: List[str] -- A list of queries for training, used for generating feedback and training evals. These are sampled from the judgments dataframe
- validation_queries: Optional[List[str]] -- An optional list of queries for validation, used for gating whether a patch gets applied permanently. These are sampled from the judgments dataframe. If not provided, no validation check occur
- eval_margin: float  - margin to improve training or validation by for acceptance
- logger: Optional[Logger] = None -- logger for logging training results and other info
- files: a dictionary of filenames to descriptions, used to build the tool description. files always adds any files managed by the coding tools (the module_name.py and queries.csv files) to the docstring, so they can be easily accessed for inspection.

### Reranker code signature

An agent will be working on code with the following structure. Its expected that module_name.py exists with a function module_name(...) of this signature:

```python
def module_name(query: str, top_k: int, tool1, tool2, ...) -> List[Tuple[int, float]]:
    """
    query: the search query
    top_k: the number of results to return

    tool1, tool2, ...: the tools used during code generation. IE a BM25 search function, etc
    """
```

The function returns up to top_k document ids ranked in order of relevance.

### Tools returned by tools()

`tools()` returns a set of tools (free functions) that an agent can use to interact with the reranker code + environment. These include:

- search - run the reranker code for a single query and return the results
- evaluate - try out a code patch and return training eval results. Patch not permanently saved. Like a scratch space for experimentation. Returns per-query eval results on the training set, used for feedback and debugging. An empty patch can be used to analyze the current reranker.
- commit_patch - try out and apply a code patch permanently if it improves validation results by margin. Only returns success / failure, not the eval results. If failure, can inspect the training eval results from evaluate to debug why it failed
- grep - a tool for searching through past training runs, used for debugging and inspecting past results. Created by make_run_path_grep_tool factory function described below. It takes a glob + regex and searches past training runs and other assets for insights or to triage individual queries.

These tools all just forward to the underlying Reranker instance.


### CodegenSearchStrategy

A "SearchStrategy" here is implemented to take source code, pass in deps, eval it, and run it with the expected signature. It implement `search` to search using the rerank code.


### Other guardrails

Other optional guardrails may be used before applying a patch. They will return error / warning in evaluate and commit_patch.

They're usually cheap checks (unlike validation evals) that can filter out obviously bad patches before running the more expensive evals

### Logging training results

Under `code_dir` its expected the working reranker code exists

Its also important to track past training (*NOT VALIDATION*) results at a granular level when `evaluate` gets run

1. Create a timestamped directory under `code_dir`/training for this training run
2. Track the following within this directory:

It should contain:

- reranker.py - the patched code that was run
- queries.csv - a CSV containing query, NDCG delta, a query_path to find the results in a subdirectory of the timestamped directory
- query_path for each training query run

Here query_path is a glob of the query

- query_path is a directory for this query, a glob of the query name, it exists under the timestamped directory

Within query_path we have

- results.csv - a CSV containing the results for this query. Formatted: query, rank, doc_id, title, description

Every CSV here should have a header
