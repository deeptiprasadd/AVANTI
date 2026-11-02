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
