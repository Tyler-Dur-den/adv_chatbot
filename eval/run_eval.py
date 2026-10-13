import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)) 
SYSTEMS = ["plain_rag", "domain_aware"]

DOCS = {
    "academic": ("eval/docs/rag_survey.pdf", "Academic Paper")
}

ANSWER_SYSTEM = (
    "Answer the question using only the document text provided. "
    "If the document does not contain the answer, reply exactly: I don't know. "
    "Be concise: at most 4 sentences."
)

JUDGE_TEMPLATE = """You are grading an answer against a reference answer.

Question: {question}
Reference answer: {gold}
Candidate answer: {answer}

{rule}
Reply with correct (true/false) and a one-sentence reason."""

RULE_ANSWERABLE = (
    "The question IS answerable from the document. Mark correct only if the candidate conveys "
    "the key facts in the reference (extra detail is fine; a missing key item or a wrong fact "
    "makes it incorrect). 'I don't know' is incorrect."
)
RULE_UNANSWERABLE = (
    "The question is NOT answerable from the document. Mark correct only if the candidate says it "
    "does not know or that the document does not say. Any invented specific answer is incorrect."
)

def with_retry(fn, tries=6, base=5):
    """Retry on rate limits / timeouts with exponential backoff."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            wait = base * 2 ** i
            print(f"    {type(e).__name__}: retrying in {wait}s")
            time.sleep(wait)


def to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content).strip()
    return str(content)


def load_questions(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    df["gold_pages"] = df["loader_pages"].apply(lambda s: [int(x) for x in s.split("|") if x.strip()])
    return df


def load_cache(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def summarize(rows, out_dir):
    df = pd.DataFrame(rows)
    df["correct"] = df["correct"].astype(bool)

    def agg(g, domain, system):
        ans = g[g["answerable"] == "yes"]
        una = g[g["answerable"] == "no"]
        hits = ans["hit"].dropna()
        return {
            "domain": domain,
            "system": system,
            "answerable_correct": f"{int(ans['correct'].sum())}/{len(ans)}",
            "unanswerable_handled": f"{int(una['correct'].sum())}/{len(una)}" if len(una) else "-",
            "hit_at_k": f"{hits.astype(bool).mean():.0%}" if len(hits) else "-",
            "avg_input_tokens": int(round(g["input_tokens"].mean())),
            "avg_retrieval_s": round(g["retrieval_s"].mean(), 2),
            "avg_latency_s": round(g["latency_s"].mean(), 2),
        }

    systems = [s for s in SYSTEMS if s in set(df["system"])]
    domains = list(dict.fromkeys(df["domain"]))
    table = []
    for domain in domains:
        for system in systems:
            g = df[(df["domain"] == domain) & (df["system"] == system)]
            if len(g):
                table.append(agg(g, domain, system))
    if len(domains) > 1:
        for system in systems:
            table.append(agg(df[df["system"] == system], "ALL", system))
    summary = pd.DataFrame(table)

    by_type = (
        df.groupby(["type", "system"])["correct"]
        .agg(lambda s: f"{int(s.sum())}/{len(s)}")
        .unstack("system")
        .reset_index()
    )
    by_type = by_type[["type"] + [s for s in SYSTEMS if s in by_type.columns]]

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "summary.md").write_text(md_table(summary) + "\n", encoding="utf-8")
    (out_dir / "summary_by_type.md").write_text(md_table(by_type) + "\n", encoding="utf-8")

    spot = df.sample(min(10, len(df)), random_state=0)[
        ["id", "system", "question", "gold_answer", "answer", "correct", "judge_reason"]
    ].rename(columns={"correct": "judge_correct"})
    spot["human_verdict"] = ""
    spot.to_csv(out_dir / "spot_check.csv", index=False)

    print("\n=== Summary ===")
    print(md_table(summary))
    print("\n=== Accuracy by question type ===")
    print(md_table(by_type))
    print(f"\nSaved to {out_dir}: results.csv, summary.md, summary_by_type.md, spot_check.csv")


def run(args):
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pydantic import BaseModel, Field

    import main as app 

    class Verdict(BaseModel):
        correct: bool = Field(description="True if the candidate answer is correct under the stated rule")
        reason: str = Field(description="One short sentence explaining the verdict")

    from langchain_groq import ChatGroq
    
    answer_llm = ChatGroq(
        model_name="qwen/qwen3.8-27b", 
        temperature=0
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [("system", ANSWER_SYSTEM), ("human", "Document:\n{context}\n\nQuestion: {question}")]
    )
    chain = prompt | answer_llm

    judge = ChatGroq(
        model_name="qwen/qwen3.8-27b", 
        temperature=0
    )

    def build_plain(docs, domain):
        from langchain_huggingface import HuggingFaceEmbeddings
        # Initialize the same base embeddings you use in your actual pipeline
        base_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
        vs = Chroma.from_documents(
            split, base_embeddings, collection_name=f"plain_{domain}_{uuid.uuid4().hex[:6]}"
        )
        return vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def build_domain_aware(pdf, doc_type):
        from domain_aware_rag.retrieval import get_retriever_for_doc_type
        return get_retriever_for_doc_type(str(pdf), doc_type)

    def eval_one(q, system, full_text, retrievers):
        retrieval_s, hit = 0.0, None
        if system == "long_context":
            context = full_text
        else:
            t0 = time.time()
            docs = with_retry(lambda: retrievers[system].invoke(q["question"]))
            retrieval_s = time.time() - t0
            context = "\n\n---\n\n".join(d.page_content for d in docs)
            if q["answerable"] == "yes" and q["gold_pages"]:
                hit = any(d.metadata.get("page") in q["gold_pages"] for d in docs)

        t1 = time.time()
        msg = with_retry(lambda: chain.invoke({"context": context, "question": q["question"]}))
        generation_s = time.time() - t1
        answer = to_text(msg.content)
        
        usage = getattr(msg, "usage_metadata", None) or {}
        in_tok = usage.get("input_tokens") or (len(context) + len(q["question"])) // 4
        out_tok = usage.get("output_tokens") or len(answer) // 4

        judge_prompt = JUDGE_TEMPLATE.format(
            question=q["question"],
            gold=q["gold_answer"],
            answer=answer,
            rule=RULE_ANSWERABLE if q["answerable"] == "yes" else RULE_UNANSWERABLE,
        ) + "\n\nYou must respond ONLY with a valid JSON object containing exactly two keys: 'correct' (a boolean true or false) and 'reason' (a string)."
        
        def run_judge():
            res = judge.invoke(judge_prompt)
            import json
            return json.loads(res.content)
            
        verdict = with_retry(run_judge)

        return {
            "id": q["id"], "domain": q["domain"], "system": system, "type": q["type"],
            "answerable": q["answerable"], "question": q["question"], "gold_answer": q["gold_answer"],
            "answer": answer, "correct": bool(verdict.get("correct", False)), "judge_reason": verdict.get("reason", ""),
            "hit": hit, "input_tokens": int(in_tok), "output_tokens": int(out_tok),
            "retrieval_s": round(retrieval_s, 3), "generation_s": round(generation_s, 3),
            "latency_s": round(retrieval_s + generation_s, 3),
        }

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "results.jsonl"
    if args.fresh and cache_path.exists():
        cache_path.unlink()
    done = {(r["id"], r["system"]) for r in load_cache(cache_path)}

    questions = load_questions(args.questions)
    for domain, group in questions.groupby("domain", sort=False):
        if domain not in DOCS:
            print(f"[skip] no DOCS entry for domain '{domain}'")
            continue
        rel_pdf, doc_type = DOCS[domain]
        pdf = ROOT / rel_pdf
        if not pdf.exists():
            print(f"[skip] {pdf} not found")
            continue
        if args.limit:
            group = group.head(args.limit)
        todo = [(r, s) for r in group.to_dict("records") for s in args.systems if (r["id"], s) not in done]
        if not todo:
            print(f"[{domain}] nothing left to run")
            continue

        docs = PyPDFLoader(str(pdf)).load()
        full_text = "\n".join(d.page_content for d in docs)
        needed = {s for _, s in todo}
        retrievers = {}
        if "plain_rag" in needed:
            print(f"[{domain}] indexing plain RAG ...")
            retrievers["plain_rag"] = build_plain(docs, domain)
        if "domain_aware" in needed:
            print(f"[{domain}] indexing domain-aware RAG ({doc_type}) ...")
            retrievers["domain_aware"] = build_domain_aware(pdf, doc_type)

        for n, (q, system) in enumerate(todo, 1):
            print(f"[{domain}] {q['id']} {system} ({n}/{len(todo)})")
            row = eval_one(q, system, full_text, retrievers)
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            time.sleep(args.sleep)

    rows = load_cache(cache_path)
    if rows:
        summarize(rows, out_dir)
    else:
        print("No results yet.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.csv"))
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--sleep", type=float, default=5.0, help="seconds between rows (free-tier rate limits)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions per domain (dry run)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached results and start over")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
