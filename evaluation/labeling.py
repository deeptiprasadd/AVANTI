"""Hand-labelled evaluation of the structured read-out and the references.

QASPER measures question answering. It says nothing about the other things
AVANTI reports: the title, the datasets and metrics, the problem /
contribution / limitation sentences, and the parsed bibliography. This
measures those on a small set of papers judged by hand.

    python -m evaluation.labeling make  --pdfs papers/ --out sheets/
    (fill in the CSV files in sheets/ -- see evaluation/LABELING.md)
    python -m evaluation.labeling score --sheets sheets/

`make` runs the analysis on every PDF and writes one CSV per paper with the
tool's output already filled in. The labeller only answers y/n per row and
lists what the tool missed, which is much faster than writing answers from
scratch. Recall is therefore measured against what the labeller found, not
against an exhaustive list.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

COLUMNS = ["paper", "task", "question", "prediction", "verdict", "entry"]

FINDING_KINDS = {
    "objective": "problem the paper addresses",
    "contribution": "contribution (what the authors built or propose)",
    "limitation": "limitation the authors admit",
}
REFERENCE_SAMPLE = 10
MAX_PER_TASK = 12


# ------------------------------------------------------------------- make


def _sample(items: Sequence, n: int) -> List:
    """n items spread evenly through the list, in order."""
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def make_rows(paper_id: str, analysis) -> List[Dict[str, str]]:
    """Label-sheet rows for one analysed paper."""
    rows: List[Dict[str, str]] = []

    def add(task: str, question: str, prediction: str = "", entry: str = "") -> None:
        rows.append({"paper": paper_id, "task": task, "question": question,
                     "prediction": prediction, "verdict": "", "entry": entry})

    add("title", "Is this the paper's actual title?", analysis.title)

    brief = analysis.brief
    for name in brief.datasets[:MAX_PER_TASK]:
        add("dataset", "Is this a dataset the paper uses or evaluates on?", name)
    add("dataset_missed",
        "Datasets the paper uses that are NOT listed above. Separate with ; "
        "and leave blank if none. (type in the entry column)")

    names = list(getattr(brief, "metric_names", []))
    names += [hit.name for hit in brief.metrics]
    seen = set()
    for name in names:
        key = name.lower()
        if key not in seen and len(seen) < MAX_PER_TASK:
            seen.add(key)
            add("metric", "Is this a metric the paper reports?", name)
    add("metric_missed",
        "Metrics the paper reports that are NOT listed above. Separate with ; "
        "and leave blank if none. (type in the entry column)")

    for kind, label in FINDING_KINDS.items():
        for finding in getattr(brief, kind)[:3]:
            add(f"finding:{kind}",
                f"Does this sentence correctly state the {label}?",
                finding.text)

    refs = analysis.references
    add("reference_count",
        "How many entries does the paper's reference list really have? "
        "(type the number in the entry column)",
        str(len(refs)))
    for ref in _sample(refs, REFERENCE_SAMPLE):
        year = ref.year_label or (str(ref.year) if ref.year else "")
        add("reference",
            "Are authors, title and year all correct for this entry? "
            "(y only if all three are right)",
            f"{ref.authors} | {ref.title} | {year}")
    return rows


def write_sheet(rows: Iterable[Dict[str, str]], path: str) -> None:
    # utf-8-sig so Excel opens accents and symbols correctly.
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def make(pdf_dir: str, out_dir: str) -> int:
    from avanti import pipeline

    os.makedirs(out_dir, exist_ok=True)
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdfs:
        print(f"no PDFs in {pdf_dir}", file=sys.stderr)
        return 1
    for path in pdfs:
        paper_id = os.path.splitext(os.path.basename(path))[0]
        target = os.path.join(out_dir, paper_id + ".csv")
        if os.path.exists(target):
            print(f"skip {paper_id}: sheet exists (not overwriting your labels)")
            continue
        with open(path, "rb") as fh:
            analysis = pipeline.run(fh.read())
        write_sheet(make_rows(paper_id, analysis), target)
        print(f"wrote {target}")
    return 0


# ------------------------------------------------------------------ score


def read_sheets(sheet_dir: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(glob.glob(os.path.join(sheet_dir, "*.csv"))):
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _yes(verdict: str) -> Optional[bool]:
    v = (verdict or "").strip().lower()
    if v in ("y", "yes", "1", "true"):
        return True
    if v in ("n", "no", "0", "false"):
        return False
    return None


def _count_items(entry: str) -> int:
    return len([x for x in (entry or "").split(";") if x.strip()])


# Per paper, every score is a pair of counts; a pooled score is the ratio of
# the sums, and resampling papers (not rows) gives honest intervals.
Tally = Dict[str, Tuple[float, float]]


def tally_paper(rows: List[Dict[str, str]]) -> Tuple[Tally, int]:
    """Counts for one paper, and how many judgeable rows were left blank."""
    t: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
    blank = 0

    def add(name: str, num: float, den: float) -> None:
        t[name][0] += num
        t[name][1] += den

    missed = {"dataset": 0, "metric": 0}
    true_refs: Optional[int] = None
    predicted_refs: Optional[int] = None
    for r in rows:
        task = r["task"]
        if task in ("dataset_missed", "metric_missed"):
            missed[task.split("_")[0]] = _count_items(r["entry"])
        elif task == "reference_count":
            predicted_refs = int(r["prediction"] or 0)
            entry = (r["entry"] or "").strip()
            true_refs = int(entry) if entry.isdigit() else None
            if true_refs is None:
                blank += 1
        else:
            ok = _yes(r["verdict"])
            if ok is None:
                blank += 1
                continue
            if task == "title":
                add("title accuracy", ok, 1)
            elif task in ("dataset", "metric"):
                add(f"{task} precision", ok, 1)
                add(f"{task} recall", ok, 0)
            elif task.startswith("finding:"):
                add(f"{task.split(':')[1]} precision", ok, 1)
            elif task == "reference":
                add("reference parse accuracy", ok, 1)

    for kind in ("dataset", "metric"):
        # Recall: correct predictions over correct predictions + missed ones.
        tp = t[f"{kind} recall"][0]
        t[f"{kind} recall"][1] = tp + missed[kind]
    if true_refs and predicted_refs is not None:
        add("reference count within 10%",
            abs(predicted_refs - true_refs) <= 0.1 * true_refs, 1)
    return {k: (v[0], v[1]) for k, v in t.items()}, blank


def _ratio(tallies: List[Tally], name: str) -> Optional[float]:
    num = sum(t[name][0] for t in tallies if name in t)
    den = sum(t[name][1] for t in tallies if name in t)
    return num / den if den else None


def _f1(tallies: List[Tally], kind: str) -> Optional[float]:
    p, r = _ratio(tallies, f"{kind} precision"), _ratio(tallies, f"{kind} recall")
    if p is None or r is None or p + r == 0:
        return None
    return 2 * p * r / (p + r)


def score(rows: List[Dict[str, str]], n_boot: int = 1000, seed: int = 0):
    by_paper: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_paper[r["paper"]].append(r)

    tallies, blank = [], 0
    for paper_rows in by_paper.values():
        t, b = tally_paper(paper_rows)
        tallies.append(t)
        blank += b

    metrics = sorted({name for t in tallies for name in t})
    getters = {name: (lambda ts, n=name: _ratio(ts, n)) for name in metrics}
    getters["dataset F1"] = lambda ts: _f1(ts, "dataset")
    getters["metric F1"] = lambda ts: _f1(ts, "metric")

    rng = np.random.default_rng(seed)
    out = {}
    for name, fn in getters.items():
        point = fn(tallies)
        if point is None:
            continue
        samples = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(tallies), size=len(tallies))
            value = fn([tallies[i] for i in pick])
            if value is not None:
                samples.append(value)
        lo, hi = np.percentile(samples, [2.5, 97.5]) if samples else (point, point)
        out[name] = {"value": point, "lo": float(lo), "hi": float(hi)}
    return out, len(by_paper), blank


def report(result, papers: int, blank: int) -> str:
    lines = [f"{papers} papers", ""]
    width = max((len(k) for k in result), default=10) + 2
    lines.append("measure".ljust(width) + "value    95% interval over papers")
    for name, v in sorted(result.items()):
        lines.append(f"{name.ljust(width)}{v['value']:.3f}    [{v['lo']:.3f}, {v['hi']:.3f}]")
    if blank:
        lines.append(f"\n{blank} rows were left blank and are not counted.")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="analyse PDFs and write label sheets")
    m.add_argument("--pdfs", required=True)
    m.add_argument("--out", required=True)
    s = sub.add_parser("score", help="score filled-in label sheets")
    s.add_argument("--sheets", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "make":
        return make(args.pdfs, args.out)
    rows = read_sheets(args.sheets)
    if not rows:
        print(f"no sheets in {args.sheets}", file=sys.stderr)
        return 1
    print(report(*score(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
