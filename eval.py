"""
Automated evaluation against golden_dataset_for_RAG_evaluation.xlsx.

Computes what can be measured automatically (retrieval precision/recall,
citation grounding) and uses the local Qwen2-7B model as an LLM judge for
what can't be checked by exact match (factual correctness) — flagged as
such, since a same-family model judging its own output is a triage tool,
not a substitute for human review.

Run:
    python eval.py
Writes:
    eval_report.csv
"""

import re
import pandas as pd
import torch

from rag import retrieve, generate, tokenizer, generation_model, DEVICE

GOLDEN_PATH = "golden_dataset_for_RAG_evaluation.xlsx"
REPORT_PATH = "eval_report.csv"


def load_golden_dataset(path: str = GOLDEN_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Golden Dataset")
    return df[df["S/N"] != "EX"]  # drop the example row


def parse_expected_sources(cell) -> set[str]:
    """
    'fund_factsheet_safe.pdf' or
    'clients_portfolio.json (CL002); policy_investment_suitability.pdf (Section 4)'
    -> {'clients_portfolio.json', 'policy_investment_suitability.pdf'}
    """
    if not isinstance(cell, str):
        return set()
    sources = set()
    for part in cell.split(";"):
        name = re.sub(r"\(.*?\)", "", part).strip()  # drop "(CL002)"-style annotations
        if name:
            sources.add(name)
    return sources


def retrieval_metrics(retrieved_sources: set[str], expected_sources: set[str]) -> dict:
    precision = (
        len(retrieved_sources & expected_sources) / len(retrieved_sources)
        if retrieved_sources else 0.0
    )
    recall = (
        len(retrieved_sources & expected_sources) / len(expected_sources)
        if expected_sources else None
    )
    return {"context_precision": precision, "context_recall": recall}


def extract_cited_sources(answer_text: str) -> set[str]:
    """Pull every '[Source: filename]' tag out of the generated answer."""
    return set(s.strip() for s in re.findall(r"\[Source:\s*([^\],\]]+)\]", answer_text))


def citation_grounding(cited: set[str], retrieved: set[str]) -> float:
    """
    Fraction of cited sources that were actually among the retrieved chunks —
    a faithfulness proxy: catches the model citing something it wasn't given.
    """
    if not cited:
        return 0.0
    return len(cited & retrieved) / len(cited)


def llm_judge_correctness(generated: str, expected: str) -> str:
    """
    Ask the local Qwen2-7B model to judge factual correctness against the
    expected answer. Triage only — spot-check the "No"/"Partial" rows by
    hand, and don't treat "Yes" here as equivalent to a human grader's sign-off.
    """
    prompt = (
        "You are grading a chatbot answer against a reference answer. "
        "Reply with exactly one word: Yes, No, or Partial.\n\n"
        f"Reference answer: {expected}\n\n"
        f"Chatbot answer: {generated}\n\n"
        "Does the chatbot answer convey the same key facts as the reference, "
        "with no contradictions?"
    )
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        output_ids = generation_model.generate(
            input_ids, max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def run_eval():
    df = load_golden_dataset()
    rows = []

    for _, row in df.iterrows():
        query = row["Sample query"]
        expected_answer = row["Expected answer"]
        expected_sources = parse_expected_sources(row["Citation sources"])

        results = retrieve(query, k=5)
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
        retrieved_sources = {m.get("source", "") for m in metadatas}

        generated_answer = generate(query, chunks, metadatas)
        cited_sources = extract_cited_sources(generated_answer)

        metrics = retrieval_metrics(retrieved_sources, expected_sources)
        grounding = citation_grounding(cited_sources, retrieved_sources)
        correctness = llm_judge_correctness(generated_answer, expected_answer)

        rows.append({
            "S/N": row["S/N"],
            "Sample query": query,
            "Expected answer": expected_answer,
            "Generated answer": generated_answer,
            "Expected sources": "; ".join(sorted(expected_sources)),
            "Retrieved sources": "; ".join(sorted(retrieved_sources)),
            "Context Precision": metrics["context_precision"],
            "Context Recall": metrics["context_recall"],
            "Citation Grounding": grounding,
            "LLM-judged correctness (spot-check manually)": correctness,
        })

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False)

    print(f"Wrote {len(report)} rows to {REPORT_PATH}\n")
    print(report[[
        "S/N", "Context Precision", "Context Recall",
        "Citation Grounding",
        "LLM-judged correctness (spot-check manually)",
    ]].to_string(index=False))


if __name__ == "__main__":
    run_eval()