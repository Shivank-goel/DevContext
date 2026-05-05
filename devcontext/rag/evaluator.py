"""RAGAS evaluation pipeline (stub)."""
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from devcontext.config.settings import settings
from devcontext.tools.docs_tools import search_docs
from devcontext.agents.docs_agent import docs_agent
from devcontext.agents import AgentState


def get_ragas_llm():
    """Wrap Ollama LLM for RAGAS."""
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.0
    )
    return LangchainLLMWrapper(llm)


def get_ragas_embeddings():
    """Wrap Ollama embeddings for RAGAS."""
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def build_eval_dataset(test_queries: list[str]) -> Dataset:
    """
    Run each query through the docs_agent and collect:
    - question
    - answer (from LLM)
    - contexts (retrieved chunks)
    for RAGAS evaluation.
    """
    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
    }

    for query in test_queries:
        print(f"  Running: {query}")

        # get retrieved context
        search_result = search_docs(query)
        contexts = [chunk["content"] for chunk in search_result["chunks"]]

        # get LLM answer via docs_agent
        state: AgentState = {
            "query": query,
            "filepath": None,
            "agent": "",
            "file_content": None,
            "diff": None,
            "retrieved_context": None,
            "response": None,
            "error": None
        }
        result = docs_agent(state)
        answer = result.get("response") or "No response generated."

        rows["question"].append(query)
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)

    return Dataset.from_dict(rows)


def run_evaluation(test_queries: list[str] = None) -> dict:
    """
    Full RAGAS evaluation pipeline.
    Returns dict of metric scores.
    """
    if test_queries is None:
        test_queries = [
            "How does the RAG pipeline work?",
            "What agents does DevContext have?",
            "What is the MCP server used for?"
        ]

    print(f"Building eval dataset for {len(test_queries)} queries...")
    dataset = build_eval_dataset(test_queries)


    print("Running RAGAS evaluation...")
    ragas_llm = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

     # key fix — increase timeout, disable parallel jobs for local Ollama
    run_config = RunConfig(
        timeout=300,      # 5 mins per call
        max_retries=2,
        max_workers=1     # sequential — Ollama can't handle parallel LLM calls
    )

    # context_precision omitted: it requires a ground-truth `reference` column;
    # see https://docs.ragas.io — add `reference` to build_eval_dataset() to use it.
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config
    )

    df = results.to_pandas()
    scores = {
        "faithfulness": round(float(df["faithfulness"].mean()), 3),
        "answer_relevancy": round(float(df["answer_relevancy"].mean()), 3),
    }

    return scores


def print_eval_report(scores: dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 40)
    print("RAGAS Evaluation Report")
    print("=" * 40)
    print(f"Faithfulness      : {scores['faithfulness']:.3f}  {'✅' if scores['faithfulness'] > 0.7 else '⚠️'}")
    print(f"Answer Relevancy  : {scores['answer_relevancy']:.3f}  {'✅' if scores['answer_relevancy'] > 0.7 else '⚠️'}")
    print("=" * 40)

    avg = sum(scores.values()) / len(scores)
    print(f"Overall Average   : {avg:.3f}")

    if avg >= 0.8:
        print("Rating: Production Ready ✅")
    elif avg >= 0.6:
        print("Rating: Needs Improvement ⚠️")
    else:
        print("Rating: Poor — revisit chunking and prompts ❌")
    print("=" * 40 + "\n")