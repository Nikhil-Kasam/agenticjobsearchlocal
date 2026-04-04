"""
evals/ragas_eval.py — RAG pipeline quality evaluation using Ragas.

Tests the ChromaDB + vLLM pipeline against 10 synthetic job-resume pairs
derived from Nikhil's actual resume content.

Metrics:
  - context_precision: Did RAG retrieve the right resume chunks?
  - context_recall:    Did we miss any relevant chunks?
  - faithfulness:      Does the LLM score match the retrieved context?

Run: python evals/ragas_eval.py
Output: evals/ragas_results.json
"""

import asyncio
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness
from datasets import Dataset
from database import VectorDBClient
from llm_client import LocalLLM


# ── 10 Synthetic test cases ────────────────────────────────────────────────────
# Format: (query, expected_relevant_context_keywords, expected_score_range)
TEST_CASES = [
    {
        "question": "Senior ML Engineer at Databricks — PyTorch, LLMs, MLflow, distributed training",
        "ground_truth": "Strong match. Candidate has PyTorch, LLMs, MLflow, and distributed systems experience.",
        "keywords": ["PyTorch", "MLflow", "LLM", "Machine Learning"],
    },
    {
        "question": "AI Research Engineer at OpenAI — NLP, Transformer architectures, RLHF, Python",
        "ground_truth": "Strong match. Candidate has deep NLP, BERT/GPT experience and Python expertise.",
        "keywords": ["NLP", "Python", "Deep Learning", "Hugging Face"],
    },
    {
        "question": "Data Engineer at Snowflake — SQL, Apache Spark, ETL pipelines, cloud data",
        "ground_truth": "Good match. Candidate has SQL, Apache Spark, and Snowflake experience.",
        "keywords": ["SQL", "Apache Spark", "Snowflake", "AWS"],
    },
    {
        "question": "MLOps Engineer at Google — Kubernetes, CI/CD, Docker, model serving, SageMaker",
        "ground_truth": "Good match. Candidate has Docker, Kubernetes, and AWS SageMaker experience.",
        "keywords": ["Docker", "Kubernetes", "CI/CD", "AWS SageMaker"],
    },
    {
        "question": "Computer Vision Engineer — TensorFlow, CNNs, image classification, GPU training",
        "ground_truth": "Moderate match. Candidate has TensorFlow and deep learning but not specifically CV.",
        "keywords": ["TensorFlow", "Deep Learning", "Python"],
    },
    {
        "question": "Quantitative Researcher at Two Sigma — Statistics, Python, financial modeling, backtesting",
        "ground_truth": "Low match. Candidate lacks quantitative finance background.",
        "keywords": ["Python", "Machine Learning"],
    },
    {
        "question": "LLM Fine-tuning Engineer at Anthropic — RLHF, LoRA, QLoRA, model alignment, Python",
        "ground_truth": "Strong match. Candidate has LLM fine-tuning and Hugging Face experience.",
        "keywords": ["LLMs", "Fine-Tuning", "Hugging Face", "Python"],
    },
    {
        "question": "Backend Engineer at Stripe — Java, distributed systems, REST APIs, microservices",
        "ground_truth": "Moderate match. Candidate has Java skills but primarily ML-focused.",
        "keywords": ["Java", "Python"],
    },
    {
        "question": "GCP Vertex AI ML Engineer — Vertex AI, BigQuery, Python, model deployment",
        "ground_truth": "Strong match. Candidate has direct GCP Vertex AI experience.",
        "keywords": ["GCP Vertex AI", "Python", "Machine Learning"],
    },
    {
        "question": "Robotics ML Engineer at Boston Dynamics — ROS, C++, SLAM, control systems",
        "ground_truth": "Low match. Candidate has C++ but no robotics or control systems background.",
        "keywords": ["C++"],
    },
]


async def run_evaluation():
    print("=" * 60)
    print("  RAGAS RAG Pipeline Evaluation")
    print("  Testing ChromaDB retrieval + Qwen3.5-27B scoring")
    print("=" * 60)

    db = VectorDBClient()
    llm = LocalLLM()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n  [{i}/{len(TEST_CASES)}] {case['question'][:60]}…")

        # RAG retrieval
        retrieved = db.search_resume(case["question"], k=4)
        context_chunks = [retrieved] if retrieved else ["No context retrieved"]

        # LLM evaluation
        evaluation = await llm.async_score_match(case["question"], retrieved)
        answer = (
            f"Score: {evaluation.match_score}/100. "
            f"Recommendation: {evaluation.recommendation}. "
            f"Reasoning: {evaluation.reasoning}"
        )

        questions.append(case["question"])
        answers.append(answer)
        contexts.append(context_chunks)
        ground_truths.append(case["ground_truth"])

        print(f"     Score: {evaluation.match_score} | Rec: {evaluation.recommendation}")
        print(f"     Retrieved {len(retrieved.split())} words of context")

    # Build Ragas dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\n  Running Ragas metrics…")
    results = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness],
    )

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for metric, score in results.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<25} {score:.3f}  {bar}")

    target_precision = 0.70
    passed = results.get("context_precision", 0) >= target_precision
    print(f"\n  Target: context_precision >= {target_precision}")
    print(f"  Status: {'✅ PASSED' if passed else '❌ BELOW TARGET'}")

    # Save results
    output = {
        "metrics": dict(results),
        "test_cases": len(TEST_CASES),
        "target_context_precision": target_precision,
        "passed": passed,
    }
    os.makedirs("evals", exist_ok=True)
    with open("evals/ragas_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved to evals/ragas_results.json")
    return output


if __name__ == "__main__":
    asyncio.run(run_evaluation())
