---
title: AVANTI
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Reads a research PDF the way you would, and cites its answers.
---

# AVANTI

A research paper analyzer that reads a PDF the way you would — structure
first, then the claims, then the numbers.

Drop in a paper and get a structured read-out: what problem it addresses,
what the authors built, what data they used, what they measured, and what
they admit it cannot do. Then ask the paper questions and get answers cited
to a page.

Everything runs locally. No API key required. Scanned PDFs are read with OCR.

---

## Why the extraction matters

Most PDF tools call `page.extract_text()` and hope. On a two-column paper
that reads straight across the page, so the two columns interleave and
figure captions land in the middle of sentences:

```
ImageNet test set, and won the 1st place in the ILSVRC
x
2015 classification competition. The extremely deep rep-
weight layer
resentationsalsohaveexcellentgeneralizationperformance
F (x) relu x on other recognition tasks, and lead us to further win the
```

No summarizer can recover from that. AVANTI reconstructs the real reading
order instead:

1. pull words with positions and font sizes, at a tolerance that doesn't
   fuse neighbouring words together
2. discard rotated margin text (the arXiv stamp down the left edge)
3. measure the body font size across the whole document, then drop text far
   from it — captions, footnotes, page numbers
4. exclude detected table regions, whose cells otherwise flatten into
   pseudo-sentences
5. detect a column gutter and read each column in turn
6. strip repeated running heads, de-hyphenate, fix ligature damage

Same page, after:

```
are comparably good or better than the constructed solution
(or unable to do so in feasible time).
In this paper, we address the degradation problem by
introducing a deep residual learning framework. Instead of hoping each
few stacked layers directly fit a desired underlying mapping, we
explicitly let these layers fit a residual mapping.
```

On the Transformer paper this recovers 4,465 words where the naive call
found 992 — the rest had been fused into unusable tokens up to 88
characters long.

---

## What it does

**Structured read-out.** Problem addressed, what the authors did, reported
results, data used, stated limitations, future work — each pulled separately
with cue-phrase plus TF-IDF scoring, and cited to a page.

**Ask the paper.** A TF-IDF index over every sentence, so you can ask
questions and get back the paper's own sentences with page and section
citations. It's retrieval, not generation: nothing is invented, and you can
check every answer against the PDF.

**Section-aware summary.** Sections are detected and budgeted separately,
weighted by both importance and length, so the method and results get the
space they deserve instead of the introduction dominating.

**Metrics extraction.** Reported accuracy, F1, BLEU, mAP, p-values and the
rest, with the sentence that stated each one. Thresholds like `mAP @ IoU =
0.5` are correctly *not* treated as results.

**Tables.** Extracted separately as data rather than smeared into the prose.

**Scanned PDFs.** A scan has no text layer at all, so nothing downstream has
anything to work with. Those pages are rasterized and read optically
instead, then fed through the same cleaning, column-ordering and analysis
path as a digital paper. It triggers automatically when a PDF turns out to
have no usable text, and the read-out is flagged so you know the text came
from OCR.

**arXiv search.** Find a paper and analyze it in place, without downloading
anything by hand.

**Optional AI report.** If a Gemini key is configured, an extra panel writes
a critical report. The panel is hidden entirely when no key is set.

---

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m streamlit run app.py
```

On macOS or Linux use `.venv/bin/` instead of `.venv/Scripts/`.

### Optional: OCR for scanned PDFs

```bash
.venv/Scripts/pip install -r requirements-ocr.txt
```

This pulls in `rapidocr-onnxruntime`, which bundles its own models and needs
no system packages -- notably no Tesseract install. Budget about ten seconds
per page on CPU.

### Optional: the AI panel

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then paste a key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
into that file. It is gitignored and must never be committed.

---

## Layout

```
app.py                 Streamlit UI
avanti/
  extract.py           PDF bytes -> layout-correct text
  sections.py          text -> Abstract / Method / Results / ... split
  textutil.py          sentences, citation stripping, prose filtering
  analyze.py           structured findings, metrics, datasets, phrases
  summarize.py         section-aware extractive summary
  qa.py                searchable index, answers cited to a page
  sources.py           arXiv search and PDF download
  ocr.py               optical reading for scanned pages
  llm.py               optional Gemini report
tests/
  test_pipeline.py     extraction, sections, analysis, summarization
  test_ocr_and_llm.py  OCR reading order, Gemini model selection
```

Each stage is a pure function over text, so it can be tested without a PDF
or a network call:

```bash
.venv/Scripts/python -m pytest tests/ -q
```

---

## Limitations

Worth knowing before you trust the output:

- **OCR is slow and imperfect.** About ten seconds a page on CPU, and it
  recovers text without font information -- so on a scan the title is
  guessed from line length rather than type size, and column detection
  works from noisier boxes. Install `requirements-ocr.txt` to enable it;
  without it, a scanned PDF is detected and reported, not silently empty.
- **Heavy inline math leaves gaps.** Equations are laid out as individually
  positioned glyphs; they're filtered out, so a sentence built around a
  formula can read with pieces missing.
- **Borderless tables partly leak.** Table detection needs ruling lines. A
  table with none is caught only by the sentence-level prose filter, so the
  occasional stray cell value still gets through.
- **Cue-phrase extraction is not comprehension.** "Stated limitations" finds
  sentences that *sound* like limitations. A paper that buries a caveat
  without signalling it will not surface here.
- **Summaries are extractive.** Sentences are selected from the paper, not
  rewritten, so the result can read a little disjointedly. That's the
  trade-off for never inventing anything.
- **Layout heuristics assume a standard paper.** Three-column layouts,
  posters, and slide decks are not handled.

---

## License

MIT
