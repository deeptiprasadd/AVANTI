# Evidence retrieval on QASPER

Task: given a question about a paper, rank the paper's paragraphs so the
ones annotators cited as evidence come first. Only answerable questions with
cited text paragraphs are scored (figure and table evidence is ignored), and
gold evidence is the union over annotators. See `qasper.py` for details.

- dev: 882 questions
- test: 1,294 questions

Reproduce:

```
pip install -r requirements-eval.txt
python -m evaluation.qasper --data qasper-test-v0.3.json --dense-model BAAI/bge-small-en-v1.5
```

The retrieval weights (`_CUE_WEIGHT`, `_RELATED_WEIGHT` in `avanti/qa.py`)
were chosen by grid search on the QASPER train split. Dev was run after the
weights were fixed, and the test split was run once, after dev, with no
further changes.

## Results

MRR, with hit@3 and recall@10 for context.

| System | dev MRR | test MRR | test hit@3 | test recall@10 |
|---|---|---|---|---|
| TF-IDF cosine | 0.355 | 0.384 | 0.454 | 0.597 |
| BM25 (plain) | 0.354 | 0.392 | 0.479 | 0.620 |
| AVANTI (BM25 + steering) | 0.385 | 0.427 | 0.523 | 0.660 |
| - no expansion | 0.379 | 0.419 | 0.512 | 0.654 |
| - no section prior | 0.373 | 0.411 | 0.507 | 0.653 |
| - no intent cues | 0.377 | 0.423 | 0.524 | 0.650 |
| Dense (bge-small, quantized) | 0.431 | 0.487 | 0.571 | 0.660 |
| Dense + plain BM25 (RRF) | 0.442 | 0.483 | 0.566 | 0.681 |
| AVANTI + Dense (RRF) | 0.460 | 0.503 | 0.595 | 0.708 |

Paired MRR differences (mean over questions, 95% bootstrap interval):

| Comparison | dev | test |
|---|---|---|
| AVANTI vs plain BM25 | +0.030 [+0.018, +0.041] | +0.035 [+0.025, +0.045] |
| Dense vs plain BM25 | +0.077 [+0.052, +0.103] | +0.095 [+0.074, +0.120] |
| Dense + BM25 vs plain BM25 | +0.088 [+0.070, +0.106] | +0.091 [+0.075, +0.107] |
| AVANTI + Dense vs plain BM25 | +0.106 [+0.086, +0.126] | +0.111 [+0.094, +0.129] |
| AVANTI + Dense vs Dense + BM25 | +0.018 [+0.009, +0.028] | +0.020 [+0.011, +0.029] |

## What this shows

- AVANTI's steering improves on plain BM25 on both splits, with intervals
  that exclude zero.
- Steering still helps on top of dense retrieval. Fusing AVANTI with the
  dense retriever beats fusing plain BM25 with it by about 0.02 MRR on both
  splits, again with intervals that exclude zero.
- Of the three parts of the steering, the section prior and related-word
  expansion help on both splits. The intent-cue term helps on dev (+0.008)
  and is close to zero on test (+0.004), so its contribution is not
  established.
- A small dense retriever alone is stronger than AVANTI alone (0.487 vs
  0.427 on test). Adding plain BM25 to dense helps on dev and does nothing
  on test (0.483 vs 0.487).
- The steering applies only to questions that match one of its six intents.
  On dev that was 577 of 882; on the rest AVANTI ranks exactly like plain
  BM25.

## Limits

- Paragraph-level evidence retrieval, not answer quality.
- One dense model: a quantized ONNX build of bge-small-en-v1.5 run through
  fastembed, no fine-tuning. A larger model would probably score higher.
- The intervals come from resampling questions. They do not cover the
  choice of tuning grid or of embedding model.
- The intent share (577 of 882) was measured on dev only.


# Read-out and references (hand-labelled)

Ten papers per set, all NLP papers from the QASPER test split, drawn at
random with a fixed seed (`random.Random(7)` over the sorted ids). Each row
the tool produced was judged against the PDF; reference counts were checked
against the rendered reference pages. How the rows were judged is in
`LABELING.md`. 95% intervals resample papers, so with ten papers they are wide.

- **Set 1** (`sheets/`, ten papers) was labelled before the fixes described
  below, and was looked at while making them.
- **Set 2** (`sheets_final/`, ten different papers) was downloaded after the
  fixes and not looked at until then. It is the fair measurement. No change
  was made to the tool after seeing it.

| Measure | Set 1, before fixes | Set 2, after fixes |
|---|---|---|
| Title correct | 1.00 [1.00, 1.00] | 0.90 [0.70, 1.00] |
| Contribution sentences correct | 0.82 [0.67, 1.00] | 0.85 [0.74, 0.96] |
| Problem ("objective") sentences correct | 0.25 [0.00, 0.33] | 0.78 [0.43, 1.00] |
| Limitation sentences correct | 0.44 [0.13, 0.71] | no limitation sentences returned |
| Datasets: precision | 0.56 [0.29, 0.80] | 0.83 [0.33, 1.00] |
| Datasets: recall | 0.50 [0.22, 0.81] | 0.42 [0.11, 0.69] |
| Datasets: F1 | 0.53 [0.29, 0.72] | 0.56 [0.18, 0.82] |
| Metrics: precision | 0.50 [0.00, 1.00] | 0.82 [0.60, 1.00] |
| Metrics: recall | 0.07 [0.00, 0.17] | 0.53 [0.26, 0.80] |
| Metrics: F1 | 0.13 [0.10, 0.29] | 0.64 [0.41, 0.86] |
| Reference count within 10% of the real one | 0.90 [0.70, 1.00] | 0.50 [0.20, 0.80] |
| Sampled references parsed fully correctly | 0.77 [0.57, 0.91] | 0.76 [0.59, 0.89] |

The fixes between the sets: metric names are found without a number beside
them (and "Recall that..." is no longer a metric); the problem sentence must
state a gap or a question; a bare "However" no longer makes a limitation;
platforms, analogies and repeats are dropped from the dataset list; and
reference entries in the `Authors: Title.` form are split correctly. (Four
other bibliography styles were fixed before set 1 was labelled.)

## What this shows

- The metric, problem-sentence and dataset-precision fixes carry over to
  papers the code had not been tuned on. The two sets are different papers,
  so the before/after figures are indicative, not a controlled comparison.
- Reference extraction does not generalise as well as the first set
  suggested. Over both sets, 14 of 20 reference counts are within 10%. On set
  2 the counts were too low on five papers (0 of 2, 17 of 19, 21 of 28,
  38 of 43, 22 of 29). Causes seen: the `Authors (Year). Title.` style, two
  columns running together, and entries cut short.
- Dataset recall is low: datasets named without the word "dataset" or
  "corpus" beside them (DUC 2002, MultiWOZ, IWSLT 2015, CCPC1.0) are missed.
- One title was wrong: a ligature extracted as "ffi E cient".

## Limits

- Ten papers per set, one field (NLP), one source (arXiv).
- A single source made all of these judgements. A check on three papers
  (`spotcheck/`) was filled in, but every row it judged was marked correct,
  including entries that are visibly wrong, so it does not discriminate and no
  agreement figure is reported. Kappa on it is 0.000.
- Recall for datasets and metrics is measured against what the labeller found
  in each paper, not an exhaustive list.
- Limitation sentences could not be assessed on set 2, because the tool
  returned none for those papers.
