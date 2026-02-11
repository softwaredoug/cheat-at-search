# Week 1 & Week 2 Notebook Progression Analysis

## Week 1: Foundations - From Basic Chat to Agentic Search

### 1. `1_Cheat_at_Search_Basic_Chat_Loop.ipynb`
**Title:** Basic chat loop

**Description:** 
- Introduces the fundamentals of the Chat Loop with the OpenAI API
- Demonstrates structured outputs using Pydantic models
- Shows how to build conversation context iteratively

**Key Concepts:**
- Basic chat loop pattern (system/user messages)
- Structured outputs with Pydantic BaseModel
- Constrained token decoding via JSON schema
- Iterative conversation building with context accumulation

**What it does:**
- Simple example with Homer Simpson character
- Shows how to use `openai.responses.create()` and `openai.responses.parse()`
- Demonstrates building conversation history step-by-step

---

### 2. `1a_RAG_Wayfair_Dataset.ipynb`
**Title:** Classic RAG

**Description:**
- Uses the Wayfair dataset (45K+ furniture/home goods products)
- Demonstrates "single shot" classic RAG pattern
- Does exactly one search, retrieves results, and asks agent to incorporate them

**Key Concepts:**
- Classic RAG (Retrieval-Augmented Generation) pattern
- Single-shot retrieval approach
- Wayfair Annotated Dataset (WANDS) introduction
- Product corpus with categories, descriptions, features

**What it does:**
- Loads Wayfair product corpus
- Performs one search query
- Retrieves relevant products
- Agent incorporates retrieved results into answer

---

### 3. `1b_Agentic_Search_Loop_Wayfair_Dataset.ipynb`
**Title:** Let's build an agentic search loop!

**Description:**
- Basic agentic search loop implementation
- Agent has access to furniture catalog
- Agent receives user preferences
- Agent uses search tool to recommend furniture
- Unrolls the loop step-by-step to understand the mechanics

**Key Concepts:**
- Agentic search loop (agent with tools)
- Agent using search as a tool/function
- Multi-step reasoning and search
- Step-by-step loop unrolling for understanding

**What it does:**
- Sets up agent with search capabilities
- Agent receives user preferences
- Agent decides when/how to search
- Agent uses search results to make recommendations

---

## Week 2: Query Understanding & Optimization

### 1. `1_Cheat_at_Search_with_LLMs_Synonyms_from_LLMs.ipynb`
**Title:** Synonyms generation with an LLM

**Description:**
- Explores using LLMs to generate query synonyms
- Expands queries with synonyms to improve search
- Evaluates impact on NDCG (Normalized Discounted Cumulative Gain)

**Key Concepts:**
- LLM-generated synonyms for query expansion
- Query understanding via synonym generation
- NDCG evaluation metric
- Improving search through query expansion

**What it does:**
- Takes user queries
- Uses LLM to generate synonyms
- Expands queries with synonyms
- Measures improvement via NDCG against baseline

---

### 2. `2_Cheat_at_Search_Perfect_Categorization.ipynb`
**Title:** Query -> Category Perfect Classification

**Description:**
- Theoretical maximum performance experiment
- Uses ground truth category labels for perfect classification
- Shows impact of perfect query understanding
- Builds classifier that returns ground truth categories
- Filters/boosts products in those categories

**Key Concepts:**
- Perfect categorization (theoretical upper bound)
- Ground truth classification
- Category-based filtering/boosting
- Understanding maximum possible performance

**What it does:**
- Classifies queries into ground truth categories
- Filters/boosts products matching those categories
- Measures performance improvement vs baseline
- Establishes theoretical maximum for comparison

---

### 3. `2a_Cheat_at_Search_Query_Categories_List.ipynb`
**Title:** Query -> Classifications LIST

**Description:**
- Extends categorization to multiple categories per query
- Classifies queries into a LIST of classifications (not just one)
- Multi-label classification approach

**Key Concepts:**
- Multi-label classification
- List of categories per query
- Handling queries that span multiple categories
- More nuanced query understanding

**What it does:**
- Classifies each query into multiple categories
- Uses list of categories for better query understanding
- Applies multi-category filtering/boosting

---

## Overall Progression Summary

### Week 1: Building Blocks
**Theme:** From basic chat to agentic search

1. **Foundation** (`1_Basic_Chat_Loop`): Learn the basic chat loop pattern and structured outputs
2. **RAG Introduction** (`1a_RAG`): Understand classic RAG - single-shot retrieval and generation
3. **Agentic Loop** (`1b_Agentic_Search`): Build agentic search where agent uses search as a tool

**Progression:**
- Simple chat → RAG with retrieval → Agent with search tool
- Single-turn → Multi-turn → Agentic multi-step reasoning
- Basic API usage → Search integration → Tool-using agents

---

### Week 2: Query Understanding
**Theme:** Improving search through better query understanding

1. **Synonym Expansion** (`1_Synonyms`): Use LLMs to expand queries with synonyms
2. **Perfect Categorization** (`2_Perfect_Categorization`): Theoretical maximum with ground truth categories
3. **Multi-Category Classification** (`2a_Query_Categories_List`): Classify queries into multiple categories

**Progression:**
- Query expansion → Query categorization → Multi-label categorization
- LLM-based synonyms → Ground truth categories → Multi-category lists
- Improving recall → Perfect understanding → Nuanced understanding

---

## Key Concepts Building Across Weeks

### 1. **Search Architecture Evolution**
- Week 1: Basic retrieval → RAG → Agentic search loop
- Week 2: Query expansion → Category filtering → Multi-category filtering

### 2. **Query Understanding**
- Week 1: Direct query → Retrieved context → Agent reasoning
- Week 2: Synonym expansion → Category classification → Multi-category classification

### 3. **Evaluation & Optimization**
- Week 1: Basic functionality demonstration
- Week 2: NDCG evaluation, baseline comparison, theoretical maximum

### 4. **Agent Capabilities**
- Week 1: Agent receives context → Agent uses search tool
- Week 2: (Prepares for Week 3) Better query understanding enables better agent decisions

---

## Leading into Week 3

The progression sets up Week 3's advanced agentic search loops by:

1. **Week 1 provides:**
   - Understanding of agentic loops
   - How agents use search tools
   - Multi-step reasoning patterns

2. **Week 2 provides:**
   - Query understanding techniques (synonyms, categorization)
   - Evaluation frameworks (NDCG, baselines)
   - Understanding of theoretical maximums

3. **Week 3 will likely combine:**
   - Agentic search loops (Week 1)
   - Advanced query understanding (Week 2)
   - More sophisticated agent reasoning
   - Iterative refinement and multi-turn search strategies

The notebooks show a clear path from basic chat → RAG → agentic search → query understanding → advanced agentic search with better query understanding.
