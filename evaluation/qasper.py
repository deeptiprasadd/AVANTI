"""Evidence-retrieval evaluation on QASPER.

QASPER (Dasigi et al., 2021) is a set of questions that annotators wrote
about NLP papers, each with the paragraphs that support the answer. This
measures how well AVANTI's retriever finds those paragraphs, against plain
baselines and with each of its additions switched off one at a time.

    python -m evaluation.qasper --data qasper-test-v0.3.json --limit 100

Scoring is per paragraph. AVANTI ranks sentences, so a paragraph is ranked
by its best sentence. A question counts only if it is answerable and at
least one annotator cited a text paragraph (figure and table evidence cannot
be retrieved from text, so it is ignored). Gold evidence is the union over
annotators.

Metrics: hit@k (any gold paragraph in the top k), recall@k (share of gold
paragraphs in the top k) and MRR (reciprocal rank of the first gold
paragraph), each with a 95% bootstrap interval over questions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from avanti import analyze, qa, sections
from avanti.textutil import normalize_spaces, sent_tokenize

KS = (1, 3, 5, 10)


@dataclass
class Question:
    paper_id: str
    text: str
    gold: List[int]          # indices into the paper's paragraph list


@dataclass
class Paper:
    paper_id: str
    paragraphs: List[str]
    kinds: List[str]         # canonical section bucket per paragraph
    datasets: List[str]
    questions: List[Question]


# ------------------------------------------------------------------ loading


def _kind(section_name: str) -> str:
    """Canonical bucket for a QASPER section name ("Methods ::: Data")."""
    parts = [p.strip() for p in (section_name or "").split(":::")]
    for part in (parts[0], parts[-1]):
        bucket, _ = sections.classify_heading(part)
        if bucket and bucket != "other":
            return bucket
    return ""


def load(path: str, limit: Optional[int] = None) -> List[Paper]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    papers: List[Paper] = []
    for paper_id, entry in raw.items():
        paragraphs, kinds = [], []
        if entry.get("abstract"):
            paragraphs.append(entry["abstract"])
            kinds.append("abstract")
        for block in entry.get("full_text", []):
            kind = _kind(block.get("section_name") or "")
            for para in block.get("paragraphs", []):
                if para and para.strip():
                    paragraphs.append(para)
                    kinds.append(kind)

        position = {text: i for i, text in reversed(list(enumerate(paragraphs)))}
        questions = []
        for item in entry.get("qas", []):
            gold = set()
            answerable = False
            for ann in item.get("answers", []):
                answer = ann.get("answer", ann)
                if answer.get("unanswerable"):
                    continue
                answerable = True
                for ev in answer.get("evidence", []):
                    if ev in position:
                        gold.add(position[ev])
            if answerable and gold:
                questions.append(Question(paper_id, item["question"], sorted(gold)))

        if questions:
            text = "\n".join(paragraphs)
            papers.append(Paper(
                paper_id, paragraphs, kinds, analyze.find_datasets(text), questions,
            ))
        if limit and len(papers) >= limit:
            break
    return papers


# ------------------------------------------------------------------ systems

# A system takes a paper and returns, for every question, paragraph indices
# from best to worst. Paragraphs it never ranks are appended unordered, so
# every system is scored over the same candidate set.
System = Callable[[Paper, Question], List[int]]


def _passages(paper: Paper) -> List[qa.Passage]:
    out = []
    for i, (text, kind) in enumerate(zip(paper.paragraphs, paper.kinds)):
        for sent in sent_tokenize(text, min_chars=40):
            sent = normalize_spaces(sent)
            if len(sent.split()) >= 6:
                # `page` carries the paragraph index, so a ranked answer maps
                # straight back to its paragraph.
                out.append(qa.Passage(sent, page=i, section="", kind=kind))
    return out


_index_cache: Dict[str, qa.PaperIndex] = {}


def _index(paper: Paper) -> qa.PaperIndex:
    if paper.paper_id not in _index_cache:
        _index_cache[paper.paper_id] = qa.PaperIndex(
            _passages(paper), datasets=paper.datasets
        )
    return _index_cache[paper.paper_id]


def _rank_paragraphs(answers: Sequence[qa.Answer], n: int) -> List[int]:
    ranked: List[int] = []
    for a in answers:
        if a.page not in ranked:
            ranked.append(a.page)
    ranked += [i for i in range(n) if i not in ranked]
    return ranked


def avanti(**switches) -> System:
    def run(paper: Paper, q: Question) -> List[int]:
        index = _index(paper)
        answers = index.ask(
            q.text, k=len(index) or 1, min_score=0.0, **switches
        )
        return _rank_paragraphs(answers, len(paper.paragraphs))
    return run


def tfidf(paper: Paper, q: Question) -> List[int]:
    """Cosine similarity over TF-IDF vectors of whole paragraphs."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    try:
        docs = vec.fit_transform(paper.paragraphs)
    except ValueError:
        return list(range(len(paper.paragraphs)))
    sims = (docs @ vec.transform([q.text]).T).toarray().ravel()
    return [int(i) for i in np.argsort(-sims, kind="stable")]


def dense_system(model_name: str) -> Optional[System]:
    """Embedding retrieval over whole paragraphs, if fastembed is installed.

    fastembed runs the model on onnxruntime, so no PyTorch is needed.
    """
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return None
    model = TextEmbedding(model_name)
    cache: Dict[str, np.ndarray] = {}

    def unit(rows) -> np.ndarray:
        arr = np.array(list(rows), dtype=np.float32)
        return arr / np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-9)

    def run(paper: Paper, q: Question) -> List[int]:
        if paper.paper_id not in cache:
            cache[paper.paper_id] = unit(model.passage_embed(paper.paragraphs))
        query = unit(model.query_embed([q.text]))[0]
        sims = cache[paper.paper_id] @ query
        return [int(i) for i in np.argsort(-sims, kind="stable")]
    return run


def fuse(a: System, b: System, k: int = 60) -> System:
    """Reciprocal rank fusion of two systems."""
    def run(paper: Paper, q: Question) -> List[int]:
        score: Dict[int, float] = {}
        for system in (a, b):
            for rank, i in enumerate(system(paper, q)):
                score[i] = score.get(i, 0.0) + 1.0 / (k + rank + 1)
        return sorted(score, key=lambda i: -score[i])
    return run


def build_systems(dense_model: Optional[str]) -> Dict[str, System]:
    systems: Dict[str, System] = {
        "TF-IDF cosine": tfidf,
        "BM25 (plain)": avanti(expand=False, section_prior=False, cues=False),
        "AVANTI (full)": avanti(),
        "  - no expansion": avanti(expand=False),
        "  - no section prior": avanti(section_prior=False),
        "  - no intent cues": avanti(cues=False),
    }
    if dense_model:
        dense = dense_system(dense_model)
        if dense is None:
            print("fastembed not installed; skipping dense systems",
                  file=sys.stderr)
        else:
            systems["Dense"] = dense
            systems["AVANTI + Dense (RRF)"] = fuse(systems["AVANTI (full)"], dense)
    return systems


# ------------------------------------------------------------------ metrics


def per_question(ranking: List[int], gold: List[int]) -> Dict[str, float]:
    gold_set = set(gold)
    first = next((r for r, i in enumerate(ranking) if i in gold_set), None)
    out = {"MRR": 0.0 if first is None else 1.0 / (first + 1)}
    for k in KS:
        top = set(ranking[:k])
        out[f"hit@{k}"] = float(bool(top & gold_set))
        out[f"recall@{k}"] = len(top & gold_set) / len(gold_set)
    return out


def bootstrap(values: np.ndarray, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(papers: List[Paper], systems: Dict[str, System]):
    names = list(systems)
    rows: Dict[str, List[Dict[str, float]]] = {n: [] for n in names}
    for paper in papers:
        for q in paper.questions:
            for name in names:
                rows[name].append(per_question(systems[name](paper, q), q.gold))
    return rows


def summarize(rows: Dict[str, List[Dict[str, float]]]) -> Dict[str, dict]:
    summary = {}
    for name, items in rows.items():
        summary[name] = {}
        for metric in items[0]:
            values = np.array([r[metric] for r in items])
            lo, hi = bootstrap(values)
            summary[name][metric] = {"mean": float(values.mean()), "lo": lo, "hi": hi}
    return summary


def paired_difference(rows, a: str, b: str, metric: str = "MRR"):
    """Mean of (a - b) per question with a bootstrap interval."""
    diff = np.array([x[metric] - y[metric] for x, y in zip(rows[a], rows[b])])
    lo, hi = bootstrap(diff)
    return float(diff.mean()), lo, hi


def table(summary: Dict[str, dict], n_questions: int) -> str:
    metrics = ["MRR"] + [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS]
    width = max(len(n) for n in summary) + 2
    lines = [f"{n_questions} answerable questions", ""]
    lines.append("system".ljust(width) + "".join(m.rjust(10) for m in metrics))
    for name, vals in summary.items():
        lines.append(
            name.ljust(width) + "".join(f"{vals[m]['mean']:10.3f}" for m in metrics)
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="QASPER json (dev or test)")
    ap.add_argument("--limit", type=int, help="use only the first N papers")
    ap.add_argument("--dense-model", help="e.g. BAAI/bge-small-en-v1.5 (needs fastembed)")
    ap.add_argument("--out", help="write full results as json")
    args = ap.parse_args(argv)

    papers = load(args.data, args.limit)
    n = sum(len(p.questions) for p in papers)
    if not n:
        print("no answerable questions with text evidence found", file=sys.stderr)
        return 1

    systems = build_systems(args.dense_model)
    rows = evaluate(papers, systems)
    summary = summarize(rows)
    print(table(summary, n))

    print("\npaired MRR difference vs plain BM25 (95% interval):")
    for name in systems:
        if name.strip() in ("BM25 (plain)",):
            continue
        d, lo, hi = paired_difference(rows, name, "BM25 (plain)")
        print(f"  {name.strip():28s} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"papers": len(papers), "questions": n, "summary": summary}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
