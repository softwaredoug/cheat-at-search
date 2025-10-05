from cheat_at_search.esci_data import corpus, judgments
from cheat_at_search.agent.strategy import ReasoningSearchStrategy
from cheat_at_search.strategy.strategy import SearchStrategy
from cheat_at_search.agent.openai_search_client import OpenAISearchClient, OpenAIChatAdapter
from cheat_at_search.search import run_strategy
from cheat_at_search.strategy import BM25Search, BestPossibleResults
from cheat_at_search.agent.history import save_queries, get_past_queries, index
from cheat_at_search.agent.judgments import make_judgments_tool
from cheat_at_search.tokenizers import snowball_tokenizer
from typing import List, Dict, Literal, Optional
from searcharray import SearchArray
import numpy as np
import pandas as pd
import sys
from time import perf_counter


if __name__ == "__main__":
    bm25 = BM25Search(corpus)
    graded_bm25 = run_strategy(bm25, judgments, num_queries=1000)
    bm25_ndcg = graded_bm25['ndcg'].mean()
    print(f"Baseline NDCG: {bm25_ndcg}")
