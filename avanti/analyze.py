"""Pull structured findings out of a paper.

A wall of summary text is hard to act on. What a reader actually wants to
know is: what problem is this, what did they build, what did they measure it
on, what numbers came out, and what does it not do. This module extracts
each of those separately, so the answer is a short labelled list rather than
prose the reader has to mine themselves.

Everything here is cue-phrase plus TF-IDF scoring -- no model, no API key.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from .textutil import clean_sentences, normalize_spaces

# ----------------------------------------------------------------- findings


@dataclass
class Finding:
    """One extracted statement, with where it came from."""

    kind: str          # objective | contribution | dataset | result | limitation | future
    text: str
    page: Optional[int] = None
    score: float = 0.0


# Cue phrases per category. Weight reflects how strongly the phrase implies
# the category -- "we propose" is a much stronger contribution signal than
# the bare word "novel".
_CUES: Dict[str, Sequence[Tuple[str, float]]] = {
    "objective": (
        (r"\bthis (?:paper|work|study|article)\s+(?:aims?|seeks?|sets out)", 3.0),
        (r"\bwe (?:aim|seek|set out|attempt) to\b", 3.0),
        (r"\bthe (?:aim|goal|objective|purpose) of this\b", 3.0),
        (r"\bwe (?:address|tackle|study|investigate|examine)\b", 2.0),
        (r"\bthe problem of\b", 1.5),
        (r"\bmotivated by\b", 1.2),
        (r"\bin this (?:paper|work|study),? we\b", 1.5),
    ),
    "contribution": (
        (r"\bwe (?:propose|present|introduce|develop|design|build)\b", 3.0),
        (r"\bthis (?:paper|work) (?:proposes|presents|introduces|describes)\b", 3.0),
        (r"\bour (?:main )?contributions?\b", 3.0),
        (r"\bwe (?:show|demonstrate|prove|find|observe)\b", 2.0),
        (r"\bwe (?:extend|adapt|improve|generalize)\b", 2.0),
        (r"\bto (?:the best of )?our knowledge\b", 1.5),
        (r"\b(?:novel|new) (?:approach|method|framework|architecture|model)\b", 1.5),
    ),
    "dataset": (
        (r"\b(?:we (?:use|used|employ|evaluate on)|evaluated? on|trained? on)\b"
         r"[^.]{0,60}\b(?:dataset|datasets|corpus|benchmark|database)\b", 3.0),
        (r"\b(?:dataset|corpus|benchmark|database) (?:of|contains?|consists?|comprises?)\b", 2.5),
        (r"\b(?:consists? of|contains?|comprises?)\s+[\d,]+\s+\w+", 2.0),
        (r"\b(?:we )?(?:collected|gathered|annotated|curated)\b", 2.0),
        (r"\b(?:training|test|validation) (?:set|split)\b", 1.5),
    ),
    "limitation": (
        (r"\b(?:a |one |main |key )?limitations?\b", 3.0),
        (r"\bhowever,?\s", 1.2),
        (r"\b(?:drawback|shortcoming|caveat|weakness)\b", 3.0),
        (r"\bwe (?:do|did) not\b", 1.8),
        (r"\b(?:fails?|failed) to\b", 2.0),
        (r"\b(?:restricted|limited|constrained) (?:to|by)\b", 2.0),
        (r"\b(?:threats? to validity)\b", 3.0),
        (r"\bonly (?:considers?|applies|works)\b", 2.0),
    ),
    # Not used by the brief, which reads results as numbers via
    # find_metrics; question answering uses it to spot a result sentence.
    "result": (
        (r"\b(?:outperform(?:s|ed)?|surpass(?:es|ed)?)\b", 2.5),
        (r"\b(?:achieves?|achieved|attains?|reaches)\b[^.]{0,60}\d", 2.5),
        (r"\bstate[- ]of[- ]the[- ]art\b", 1.5),
        (r"\b(?:improves?|improved|improvements?)\b[^.]{0,50}\b(?:by|over)\b", 2.0),
        (r"\bahead of\b", 1.5),
        (r"\d(?:\.\d+)?\s*(?:%|points?\b|BLEU\b|mAP\b|AP\b|F1\b|PQ\b|mIoU\b|top-1\b)",
         1.5),
    ),
    "future": (
        (r"\bfuture work\b", 3.0),
        (r"\bin (?:the )?future\b", 2.5),
        (r"\bwe (?:plan|intend|hope) to\b", 3.0),
        (r"\b(?:further|additional) (?:research|work|study|investigation)\b", 2.0),
        (r"\bwould be (?:an )?interesting\b", 1.5),
        (r"\bremains? (?:an )?open\b", 2.0),
    ),
}

_COMPILED_CUES = {
    kind: [(re.compile(p, re.I), w) for p, w in cues]
    for kind, cues in _CUES.items()
}

# --------------------------------------------------------------- metrics

# Metric names worth reporting when they appear next to a number.
_METRIC_NAMES = (
    "accuracy", "precision", "recall", "f1", "f-1", "f-score", "f1-score",
    "auc", "auroc", "auprc", "bleu", "rouge", "meteor", "cider", "map",
    "miou", "iou", "dice", "rmse", "mae", "mse", "mape", "r2",
    "sensitivity", "specificity", "error rate", "top-1", "top-5",
    "perplexity", "ppl", "wer", "cer", "eer", "psnr", "ssim",
    "correlation", "kappa", "auc-roc",
)

_METRIC_ALT = "|".join(re.escape(m) for m in _METRIC_NAMES)

# "accuracy of 94.2%", "94.2% accuracy", "F1 = 0.91", "BLEU score of 28.4"
_METRIC_PATTERNS = (
    re.compile(rf"\b({_METRIC_ALT})\b[^.\d]{{0,25}}?([-+]?\d+(?:\.\d+)?)\s*(%?)", re.I),
    re.compile(rf"([-+]?\d+(?:\.\d+)?)\s*(%?)\s*\b({_METRIC_ALT})\b", re.I),
)

# Require an inequality or the explicit term: a bare "P = 0.1" in a paper is
# usually a dropout rate or a probability parameter, not a significance test.
_PVALUE = re.compile(
    r"(?:\bp\s*[<≤]\s*0?\.\d+"
    r"|\bp[-\s]values?\s*(?:of|=|was|were)?\s*0?\.\d+)",
    re.I,
)

# Well-known dataset names, so they are recognized even without the word
# "dataset" nearby.
_KNOWN_DATASETS = (
    "ImageNet", "CIFAR-10", "CIFAR-100", "CIFAR", "MNIST", "Fashion-MNIST",
    "COCO", "MS COCO", "PASCAL VOC", "Cityscapes", "ADE20K", "KITTI",
    "SQuAD", "GLUE", "SuperGLUE", "SST-2", "MultiNLI", "SNLI", "WMT",
    "WikiText", "Penn Treebank", "LibriSpeech", "AudioSet", "UCF101",
    "Kinetics", "CheXpert", "MIMIC-III", "MIMIC-CXR", "TCGA", "UK Biobank",
    "OpenWebText", "Common Crawl", "C4", "The Pile", "LAION",
)
_DATASET_RE = re.compile(
    r"\b(" + "|".join(re.escape(d) for d in _KNOWN_DATASETS) + r")\b"
)

# A capitalized name immediately before "dataset"/"benchmark"/"corpus".
_NAMED_DATASET = re.compile(
    r"\b([A-Z][A-Za-z0-9\-]{1,20}(?:\s+[A-Z][A-Za-z0-9\-]{1,20}){0,2})\s+"
    r"(?:dataset|benchmark|corpus|database)\b"
)


@dataclass
class MetricHit:
    """A measured number found in the text."""

    name: str
    value: str
    unit: str
    sentence: str
    page: Optional[int] = None


# "Recall that in task 2 ..." is a verb, and "2" a task number, not a score.
_RECALL_VERB = re.compile(r"\brecall\s+that\b", re.I)

# Metrics a paper uses are often named without a number beside them -- in a
# table caption, or "we measure precision, recall and F-score". Each entry is
# (name shown, pattern, distinctive). A distinctive name ("BLEU") is a metric
# on one mention in an evaluating context; a common word ("accuracy",
# "precision") has to be used that way twice, or it is just prose.
_METRIC_FORMS = (
    ("accuracy", re.compile(r"\b[Aa]ccuracy\b"), False),
    ("precision", re.compile(r"\b[Pp]recision\b"), False),
    ("recall", re.compile(r"\b[Rr]ecall\b(?!\s+that\b)"), False),
    ("F1", re.compile(r"\bF\s?-?\s?1(?:[- ]?[Ss]cores?)?\b|\bF-[Ss]cores?\b"
                      r"|\bF-[Mm]easure\b"), True),
    # Small caps come out of a PDF as "B LEU", so the first letter may be
    # split from the rest.
    ("BLEU", re.compile(r"\bB ?LEU\b"), True),
    ("ROUGE", re.compile(r"\bR ?OUGE(?:-[A-Z0-9]+)?\b"), True),
    ("METEOR", re.compile(r"\bM ?ETEOR\b"), True),
    ("AUC", re.compile(r"\bAU(?:ROC|PRC|C)\b"), True),
    ("mAP", re.compile(r"\bmAP\b"), True),
    ("MRR", re.compile(r"\bMRR\b"), True),
    ("NMI", re.compile(r"\bNMI\b"), True),
    ("IoU", re.compile(r"\bm?IoU\b"), True),
    ("EER", re.compile(r"\bEER\b"), True),
    ("minDCF", re.compile(r"\bmin-?DCF\b", re.I), True),
    ("WER", re.compile(r"\bWER\b"), True),
    ("RMSE", re.compile(r"\bRMSE\b"), True),
    ("hits@k", re.compile(r"\b[Hh]its@\s?\w+"), True),
    ("perplexity", re.compile(r"\b[Pp]erplexity\b"), False),
    ("correlation", re.compile(r"\b(?:Pearson|Spearman|Kendall)\b"), False),
)

# What makes a mention an evaluation rather than prose: a number or percent
# beside it, or the vocabulary of measuring. "Recall" also needs a sibling
# metric, since "we recall" and "recall the problem" are ordinary English.
_EVAL_CONTEXT = re.compile(
    r"\d|%|\b(?:metrics?|scores?|measured?|measures|evaluat\w*|report\w*|"
    r"table|achiev\w*|compar\w*)\b", re.I
)
_RECALL_CONTEXT = re.compile(
    r"%|@|\bprecision\b|\bF\s?-?\s?1\b|\bF-?scores?\b|\bF-?measure\b|"
    r"\bmetrics?\b|\bmeasured?\b|\baccuracy\b", re.I
)
_CONTEXT_CHARS = 70


def find_metric_names(text: str, limit: int = 8) -> List[str]:
    """Names of the metrics the paper evaluates with, most-used first."""
    flat = normalize_spaces(text)
    found: List[Tuple[int, str]] = []
    for name, pattern, distinctive in _METRIC_FORMS:
        used = 0
        for m in pattern.finditer(flat):
            near = flat[max(0, m.start() - _CONTEXT_CHARS) : m.end() + _CONTEXT_CHARS]
            check = _RECALL_CONTEXT if name == "recall" else _EVAL_CONTEXT
            if check.search(near):
                # Another metric beside it ("precision, recall and F-score")
                # settles that this is an evaluation, not prose.
                sibling = any(
                    p.search(near) for n, p, _ in _METRIC_FORMS if n != name
                )
                used += 2 if sibling else 1
        if used >= (1 if distinctive else 2):
            found.append((used, name))
    found.sort(key=lambda t: -t[0])
    return [name for _, name in found[:limit]]


def find_metrics(text: str, page_lookup=None, limit: int = 25) -> List[MetricHit]:
    """Extract reported metric values with the sentence that stated them."""
    hits: List[MetricHit] = []
    seen = set()

    for sent in clean_sentences(text, min_chars=20):
        for pattern in _METRIC_PATTERNS:
            for m in pattern.finditer(sent):
                if _RECALL_VERB.match(m.group(0)):
                    continue
                groups = m.groups()
                # Pattern 1 is (name, value, unit); pattern 2 is (value, unit, name).
                if groups[0] and groups[0][0].isalpha():
                    name, value, unit = groups[0], groups[1], groups[2]
                else:
                    value, unit, name = groups[0], groups[1], groups[2]

                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                # A percentage above 100 or a wild value is a false match
                # (often a citation number or a year).
                if unit == "%" and not (0 <= numeric <= 100):
                    continue
                if not unit and abs(numeric) > 10000:
                    continue
                # No metric here is meaningfully negative; a minus sign means
                # we matched across a range such as "top-1/-5".
                if numeric < 0:
                    continue
                # An "@" anywhere in the match means the number is a threshold
                # or cut-off, not a score: "mAP@0.5", "Recall@10", and
                # "mAP @ IoU = 0.5" all state the evaluation setting.
                if "@" in m.group(0):
                    continue

                key = (name.lower(), value, unit)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    MetricHit(
                        name=name.upper() if len(name) <= 3 else name.lower(),
                        value=value,
                        unit=unit,
                        sentence=sent,
                        page=page_lookup(sent) if page_lookup else None,
                    )
                )
                if len(hits) >= limit:
                    return hits

        for m in _PVALUE.finditer(sent):
            key = ("p-value", m.group(0), "")
            if key not in seen:
                seen.add(key)
                hits.append(
                    MetricHit("p-value", m.group(0), "", sent,
                              page_lookup(sent) if page_lookup else None)
                )
    return hits


def find_datasets(text: str, limit: int = 12) -> List[str]:
    """Dataset and benchmark names mentioned in the paper."""
    counts: Counter = Counter()

    for m in _DATASET_RE.finditer(text):
        counts[m.group(1)] += 1
    for m in _NAMED_DATASET.finditer(text):
        name = m.group(1).strip()
        # Skip sentence-initial noise words that happen to be capitalized.
        if name.lower() in {"the", "this", "our", "a", "an", "we", "in", "for",
                            "on", "these", "each", "all", "both", "other",
                            "same", "large", "public", "real", "new"}:
            continue
        counts[name] += 1

    return [name for name, _ in counts.most_common(limit)]


# -------------------------------------------------------------- key phrases


def key_phrases(text: str, top_k: int = 14) -> List[str]:
    """Salient multi-word phrases, ranked by TF-IDF mass.

    Sentences are the documents, so a phrase scores highly when it is
    important but not spread uniformly across the whole paper. This replaces
    raw word-frequency counting, which just returned common English words.
    """
    sents = clean_sentences(text, min_chars=25)
    if len(sents) < 3:
        return []

    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            max_df=0.55,       # drop phrases that appear nearly everywhere
            min_df=2,          # and one-off phrases
            sublinear_tf=True,
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-]{2,}\b",
        )
        X = vec.fit_transform(sents)
    except ValueError:
        return []

    mass = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    ranked = sorted(zip(terms, mass), key=lambda t: -t[1])

    # Prefer longer phrases, and drop a phrase fully contained in one we
    # already kept ("neural" when "neural network" is already in).
    chosen: List[str] = []
    for term, _ in ranked:
        if len(chosen) >= top_k:
            break
        if any(term in kept and term != kept for kept in chosen):
            continue
        chosen = [k for k in chosen if k not in term or k == term]
        chosen.append(term)
    return chosen[:top_k]


# ---------------------------------------------------------- cue extraction


def _salience(sents: List[str]) -> Dict[int, float]:
    """TF-IDF energy per sentence, normalized to 0..1."""
    if len(sents) < 3:
        return {i: 0.5 for i in range(len(sents))}
    try:
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        X = vec.fit_transform(sents)
    except ValueError:
        return {i: 0.5 for i in range(len(sents))}
    energy = X.power(2).sum(axis=1).A1
    lo, hi = float(energy.min()), float(energy.max())
    if hi <= lo:
        return {i: 0.5 for i in range(len(sents))}
    return {i: (float(e) - lo) / (hi - lo) for i, e in enumerate(energy)}


def cue_score(sentence: str, kind: str) -> float:
    """How strongly a sentence is phrased as one category (0 if not at all)."""
    return sum(
        weight for pattern, weight in _COMPILED_CUES.get(kind, ())
        if pattern.search(sentence)
    )


# Extra evidence for the paper read-out only. The cue weights above are shared
# with question answering, whose ranking is tuned on them, so what the read-out
# alone needs lives here. Checked against papers read by hand, the bare cues
# picked "we perform an extensive evaluation" as the problem, and a sentence
# beginning "However" -- about earlier work, or answering an objection -- as
# the paper's own limitation.
def _rx(*patterns: Tuple[str, float]):
    return tuple((re.compile(p, re.I), w) for p, w in patterns)


_BRIEF_BONUS = {
    # A gap in what is known or can be done, or a question the paper answers.
    # The bare words "problem" and "challenge" are not enough: background
    # about the topic ("cyberbullying is a serious problem") uses them too.
    "objective": _rx(
        (r"\b(?:however|but|yet|despite|although)\b[^.]{0,120}\b(?:lack\w*|"
         r"limited|difficult\w*|challeng\w+|unclear|unknown|fail\w*|struggle\w*|"
         r"suffer\w*|remain\w*|not (?:well |yet )?(?:understood|explored|"
         r"studied))\b", 1.5),
        (r"\b(?:little|few|no) (?:is|are|work|studies|research)\b[^.]{0,40}"
         r"\b(?:known|understood|done|explored|studied|exist\w*)\b", 1.5),
        (r"\bremains? (?:an? )?(?:open|unsolved|challenging|difficult)\b", 1.0),
        (r"\b(?:is|are) (?:often |still |very )?(?:difficult|challenging) to\b", 1.0),
        # "whether" is left out: "Whether in literature or in popular culture"
        # is not a question the paper asks.
        (r"\b(?:to what extent|the (?:extent|degree) to which|"
         r"how (?:to|well|much))\b", 1.5),
    ),
    # The authors speaking about their own work and what it cannot do.
    "limitation": _rx(
        (r"\b(?:we|our)\b[^.]{0,80}\b(?:cannot|can ?not|unable|not able|"
         r"restricted to|limited to|not (?:been )?(?:evaluated|tested|"
         r"considered|addressed|explored))\b", 1.0),
    ),
}

_BRIEF_PENALTY = {
    "objective": _rx(
        # What they did, not what they set out to solve.
        (r"\bwe (?:have )?(?:propos|present|introduc|develop|describ|perform|"
         r"conduct|build|built|design)\w*", 2.0),
        # A result.
        (r"\b(?:outperform\w*|state[- ]of[- ]the[- ]art)\b|\d\s*%", 2.0),
        (r"\b(?:organi[sz]ed|structured) as follows\b|\bthe (?:rest|remainder) of\b",
         3.0),
        (r"\bwe (?:look|take a look|talk|discuss)\b", 1.5),
        (r"^\s*while (?:they|these|previous|prior|existing)\b", 1.5),
    ),
    "limitation": _rx(
        # About earlier work, not the paper's own.
        (r"\b(?:existing|previous|prior|earlier|traditional|these) "
         r"(?:work|approaches|methods|models|systems|studies)\b", 1.5),
        # "However ..." with no "we" or "our": not the authors' admission.
        (r"^\s*however,?\s(?!.*\b(?:we|our)\b)", 1.0),
        (r"\bshould not matter\b|\bnot a (?:big |major )?(?:problem|issue)\b", 2.0),
    ),
}

# A bare "however" says nothing about limits; it needs another cue beside it.
_HOWEVER = re.compile(r"\bhowever,?\s", re.I)
_HOWEVER_WEIGHT = 1.2


def _brief_score(sent: str, kind: str) -> Tuple[float, float]:
    """(cue strength, adjustment) for the read-out. A sentence is kept when
    either the cues or the extra evidence is positive."""
    cues = cue_score(sent, kind)
    if kind == "limitation" and _HOWEVER.search(sent):
        cues -= _HOWEVER_WEIGHT
    bonus = sum(w for p, w in _BRIEF_BONUS.get(kind, ()) if p.search(sent))
    penalty = sum(w for p, w in _BRIEF_PENALTY.get(kind, ()) if p.search(sent))
    return cues, bonus - penalty


def find_by_cues(
    text: str,
    kind: str,
    limit: int = 4,
    page_lookup=None,
) -> List[Finding]:
    """Find the sentences that best express one category."""
    if kind not in _COMPILED_CUES:
        return []

    sents = clean_sentences(text, min_chars=40)
    if not sents:
        return []

    sal = _salience(sents)
    scored: List[Finding] = []
    tuned = kind in _BRIEF_BONUS

    for i, sent in enumerate(sents):
        cues, adjust = _brief_score(sent, kind)
        if cues <= 0 and not (tuned and adjust > 0):
            continue
        # Prefer sentences that are informative as well as cue-bearing, and
        # mildly penalize very long ones, which tend to ramble.
        length_penalty = 0.4 if len(sent.split()) > 55 else 0.0
        total = cues + adjust + 0.8 * sal.get(i, 0.0) - length_penalty
        # A read-out sentence that the evidence argues against is dropped, not
        # shown last: an empty slot is better than a wrong sentence.
        if tuned and total < 1.0:
            continue
        scored.append(
            Finding(
                kind=kind,
                text=sent,
                page=page_lookup(sent) if page_lookup else None,
                score=total,
            )
        )

    scored.sort(key=lambda f: -f.score)

    # Drop near-duplicates by token overlap.
    kept: List[Finding] = []
    for f in scored:
        tokens = set(f.text.lower().split())
        if any(
            len(tokens & set(k.text.lower().split())) / max(1, len(tokens)) > 0.6
            for k in kept
        ):
            continue
        kept.append(f)
        if len(kept) >= limit:
            break
    return kept


# Names that "<Name> dataset" picks up but that are a place data comes from,
# not a dataset.
_PLATFORMS = frozenset({
    "twitter", "facebook", "instagram", "youtube", "reddit", "weibo", "flickr",
    "amazon", "google", "yahoo", "linkedin", "tiktok", "whatsapp",
})

# A dataset named only to say what the paper hopes to become -- "like the Penn
# Treebank, SQuAD and ImageNet have driven progress" -- is not one it uses.
_ANALOGY = re.compile(
    r"\b(?:driven progress|similar to how|just as|analogous to|in the same way|"
    r"notable examples)\b",
    re.I,
)

_LEADING_WORD = re.compile(r"^(?:(?:the|dataset|corpus|a|an)\s+)+", re.I)

# One-word "names" that are only an adjective or a kind of data.
_GENERIC_NAMES = frozenset({
    "phonetic", "acoustic", "image", "images", "text", "speech", "audio",
    "video", "training", "testing", "test", "validation", "evaluation",
    "development", "public", "private", "target", "source", "english",
    "french", "german", "chinese", "spanish", "large", "small", "new",
})


def clean_datasets(names: List[str], text: str) -> List[str]:
    """Drop platforms, analogies and repeats from the dataset names found.

    Repeats are the same name differing in case, a leading "The" or "Dataset",
    or a shorter form of a longer one ("SST" beside "SST-2").
    """
    sentences = clean_sentences(text, min_chars=20)
    cleaned: List[str] = []
    for name in names:
        name = _LEADING_WORD.sub("", name).strip()
        if not name or name.lower() in _PLATFORMS | _GENERIC_NAMES:
            continue
        mentions = [s for s in sentences if name in s]
        if mentions and all(_ANALOGY.search(s) for s in mentions):
            continue
        cleaned.append(name)

    out: List[str] = []
    for name in cleaned:
        low = name.lower()
        if any(other.lower().startswith(low + "-") for other in cleaned):
            continue
        if any(kept.lower() == low for kept in out):
            continue
        out.append(name)
    return out


@dataclass
class Brief:
    """The structured read-out for a paper."""

    objective: List[Finding]
    contribution: List[Finding]
    dataset_mentions: List[Finding]
    limitation: List[Finding]
    future: List[Finding]
    datasets: List[str]
    metrics: List[MetricHit]
    phrases: List[str]
    # Metrics the paper evaluates with, including ones named without a number.
    metric_names: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.objective, self.contribution, self.limitation,
             self.future, self.metrics, self.datasets)
        )


def brief(paper_text: str, sections_map: Optional[Dict[str, str]] = None,
          page_lookup=None, metric_text: Optional[str] = None) -> Brief:
    """Build the structured read-out.

    Where the paper's sections are known, each category is searched in the
    section most likely to contain it -- limitations belong to the
    discussion, not the introduction -- which cuts false positives sharply.

    `metric_text` is the paper's text including small print and table
    captions, up to the bibliography: that is where a metric is named, and
    the analysis text drops it.
    """
    sections_map = sections_map or {}

    def source(*names: str) -> str:
        for n in names:
            if sections_map.get(n):
                return sections_map[n]
        return paper_text

    intro = source("introduction", "abstract")
    # The abstract states the problem most plainly, so the problem is looked
    # for there as well as in the introduction.
    framing = "\n".join(
        t for t in (sections_map.get("abstract"), sections_map.get("introduction")) if t
    ) or intro
    method = source("method", "introduction")
    data = source("experiments", "method")
    disc = source("discussion", "conclusion", "results")
    results_text = source("results", "experiments")

    objective = find_by_cues(framing, "objective", 4, page_lookup)
    contribution = find_by_cues(
        intro + "\n" + method, "contribution", 6, page_lookup
    )

    # A paper's thesis sentence matches both "objective" and "contribution"
    # cues; print it once, under the heading it matches more strongly. "In
    # this work we propose the Transformer" is a contribution ("we propose"
    # outweighs "in this work, we"), and filing it as the problem left the
    # contribution slot to a stray "we observe" from the ablations.
    both = {f.text for f in objective} & {f.text for f in contribution}
    to_contribution = {
        t for t in both if cue_score(t, "contribution") > cue_score(t, "objective")
    }
    objective = [f for f in objective if f.text not in to_contribution][:3]
    contribution = [
        f for f in contribution if f.text not in both - to_contribution
    ][:4]

    return Brief(
        objective=objective,
        contribution=contribution,
        dataset_mentions=find_by_cues(data, "dataset", 3, page_lookup),
        limitation=find_by_cues(disc, "limitation", 4, page_lookup),
        future=find_by_cues(disc, "future", 3, page_lookup),
        datasets=clean_datasets(find_datasets(paper_text), paper_text),
        metrics=find_metrics(results_text or paper_text, page_lookup),
        phrases=key_phrases(paper_text),
        metric_names=find_metric_names(
            metric_text
            or "\n".join(t for t in (paper_text, sections_map.get("abstract")) if t)
        ),
    )
