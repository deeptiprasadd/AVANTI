# Evidence retrieval on QASPER (dev split)

Task: given a question about a paper, rank the paper's paragraphs so the
ones annotators cited as evidence come first. 882 answerable questions
over 281 papers with text evidence (figure and table evidence is ignored).
Gold evidence is the union over annotators. See `qasper.py` for details.

Reproduce:

```
pip install -r requirements-eval.txt
python -m evaluation.qasper --data qasper-dev-v0.3.json --dense-model BAAI/bge-small-en-v1.5
```

The retrieval weights (`_CUE_WEIGHT`, `_RELATED_WEIGHT` in `avanti/qa.py`)
were chosen by grid search on the QASPER train split. Dev was run after
the weights were fixed.

| System | MRR | hit@1 | hit@3 | hit@10 | recall@10 |
|---|---|---|---|---|---|
| TF-IDF cosine | 0.355 | 0.184 | 0.415 | 0.746 | 0.614 |
| BM25 (plain) | 0.354 | 0.173 | 0.426 | 0.777 | 0.645 |
| AVANTI (BM25 + steering) | 0.385 | 0.203 | 0.472 | 0.796 | 0.664 |
| - no expansion | 0.379 | 0.200 | 0.458 | 0.788 | 0.657 |
| - no section prior | 0.373 | 0.187 | 0.455 | 0.794 | 0.660 |
| - no intent cues | 0.377 | 0.195 | 0.465 | 0.793 | 0.664 |
| Dense (bge-small, quantized) | 0.431 | 0.259 | 0.510 | 0.815 | 0.675 |
| Dense + plain BM25 (RRF) | 0.442 | 0.269 | 0.524 | 0.831 | 0.695 |
| AVANTI + Dense (RRF) | 0.460 | 0.287 | 0.548 | 0.842 | 0.715 |

Paired MRR difference against plain BM25 (95% bootstrap interval over
questions): AVANTI +0.030 [+0.018, +0.041]; Dense +0.077 [+0.052, +0.103];
Dense + BM25 +0.088 [+0.070, +0.106]; AVANTI + Dense +0.106 [+0.086, +0.126].

## What this shows

- AVANTI's section- and intent-aware ranking improves on plain BM25, and
  removing any of its three parts lowers MRR.
- The gain only applies to questions that match one of its six intents
  (577 of 882). On the other 305 it ranks exactly like plain BM25.
- A small dense retriever alone is stronger than AVANTI alone.
- Fusing AVANTI with the dense retriever scores higher than fusing plain
  BM25 with it (0.460 vs 0.442 MRR). The paired interval for that gap is not
  computed in the run above; the harness now prints it.

## Limits

- Dev split only. The test split has not been used.
- Paragraph-level retrieval, not answer quality.
- One dense model, quantized ONNX build via fastembed, no fine-tuning.
- Single run with deterministic ranking; the intervals come from resampling
  questions, not from repeated training.
