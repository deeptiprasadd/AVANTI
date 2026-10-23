# AVANTI

![tests](https://github.com/deeptiprasadd/AVANTI/actions/workflows/tests.yml/badge.svg)

A research paper analyzer that reads a PDF the way you would: structure
first, then the claims, then the numbers.

## Live demos

Two front ends, one shared analysis engine:

| Version | Link | Notes |
|---|---|---|
| Web app (FastAPI + custom UI) | https://avanti-vxf8.onrender.com | Includes OCR for scanned PDFs. Free tier: the first visit after a quiet spell takes about a minute to wake up. |
| Streamlit app | https://avanti-research-paper-analyzer2025.streamlit.app/ | No OCR (the free build has no room for it). |

## What it does

- **Structured read-out.** What problem the paper addresses, what the authors
  built, the data they used, the numbers they report, and the limitations
  they admit.
- **Ask the paper.** Questions are answered with sentences from the paper
  itself, cited to a page and section. Ranking is BM25, steered by what the
  question is asking for: problem questions prefer the abstract and
  introduction, limitations prefer the discussion, and dataset questions
  also search for the paper's own dataset names.
- **References.** The bibliography is parsed into authors, title, year and
  arXiv/DOI links. Each entry is matched to the paper's in-text citations,
  so you can see which prior work the paper leans on most. Exports to CSV.
- **Compare two papers.** Problem, method, data and results side by side,
  plus topic overlap, shared datasets and metrics, prior work both cite, and
  whether one paper cites the other.
- **Find papers.** Search arXiv, OpenAlex (every field, open-access PDFs
  only) and Europe PMC (biomedical) together or one at a time, choose how
  many results you want (up to 50), and analyze any result in one click.
- **Layout-aware extraction.** Two-column papers are read in the right
  order, and small-caps titles and headings are repaired.

Everything runs on the server. No API key is needed. An optional Gemini
report appears only if a key is configured.

## Project layout

```
avanti/      analysis engine (shared by both apps)
  extract     PDF -> layout-correct text, title, authors
  sections    section detection (Abstract, Method, Results, ...)
  analyze     structured findings, metrics, datasets
  summarize   section-aware summary
  qa          BM25 question answering with page citations
  references  bibliography parsing and citation counts
  compare     side-by-side comparison of two papers
  sources     arXiv / OpenAlex / Europe PMC search, safe PDF download
  textutil    sentences, citation stripping, prose filtering
  ocr         optical reading for scanned pages
  llm         optional Gemini report
  pipeline    all of the above in one call
server/      FastAPI app and the web UI (deployed on Render via Dockerfile)
app.py       Streamlit app (deployed on Streamlit Community Cloud)
tests/       pytest suite, run on every push by GitHub Actions
```

## Run locally

```
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest -q
.venv/Scripts/streamlit run app.py                         # Streamlit app
.venv/Scripts/python -m uvicorn server.main:app --reload   # web app
```

Dependencies: `requirements.txt` is for Streamlit Cloud,
`requirements-server.txt` is for the Docker image (adds the API and OCR),
and `requirements-dev.txt` is for development and CI.

## Security notes

- The server downloads PDFs only from public internet addresses. Private,
  loopback and cloud-metadata addresses are refused, and every redirect is
  re-checked.
- There is no endpoint that lists other users' analyses. A browser only
  knows the job ids it created.
- CI runs with a read-only token, uses no secrets, and pins each GitHub
  Action to a full commit hash.

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

## Features in detail

**Structured read-out.** Problem addressed, what the authors did, reported
results, data used, stated limitations, future work — each pulled separately
with cue-phrase plus TF-IDF scoring, and cited to a page. A sentence that
fits two categories is filed under the one it matches more strongly.

**Ask the paper.** A BM25 index over every sentence of the body (the
bibliography is left out), so you can ask questions and get back the
paper's own sentences with page and section citations. It's retrieval, not
generation: nothing is invented, and you can check every answer against
the PDF.

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

**References.** The bibliography is read from the small print that the
analysis text leaves out, split into entries (numbered and author-year
styles), and matched against the in-text citations, so the most-relied-on
prior work rises to the top.

**Search.** arXiv, OpenAlex and Europe PMC. Only results with an open PDF
are shown, since a paper has to be read to be analyzed, and where OpenAlex
lists several copies of a paper the arXiv copy is preferred because it
reliably downloads.

**Optional AI report.** If a Gemini key is configured, an extra panel writes
a critical report. The panel is hidden entirely when no key is set.

---

## More setup

On macOS or Linux use `.venv/bin/` instead of `.venv/Scripts/` in the
commands above.

### OCR for scanned PDFs

The Render web app includes OCR. The Streamlit app does not, and this is
deliberate: `rapidocr-onnxruntime` pulls in OpenCV and onnxruntime, about
100MB of wheels and 41% of the entire install, enough to push Streamlit
Community Cloud's free build past its limit. Without it the app detects a
scanned PDF and reports it rather than failing on it.

To enable it locally:

```bash
.venv/Scripts/pip install rapidocr-onnxruntime
```

On Linux it also needs two system libraries, or OpenCV fails to import with
`libGL.so.1: cannot open shared object file`:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

Budget about ten seconds a page on CPU. `AVANTI_ENABLE_OCR=0` switches it
off again without uninstalling, and `AVANTI_OCR_DPI` trades accuracy for
speed.

### Optional: the AI panel

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then paste a key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
into that file. It is gitignored and must never be committed. On Render,
set `GEMINI_API_KEY` in the service's environment settings instead.

---

## Limitations

Worth knowing before you trust the output:

- **OCR is slow and imperfect.** About ten seconds a page on CPU, and it
  recovers text without font information -- so on a scan the title is
  guessed from line length rather than type size, and column detection
  works from noisier boxes. It can be switched off with
  `AVANTI_ENABLE_OCR=0`, in which case a scanned PDF is detected and
  reported rather than silently coming back empty.
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
- **Reference parsing is heuristic.** It handles the common numbered and
  author-year styles; an unusual style can split or title an entry wrongly.
  Citation counts for author-year papers match on surname and year, so two
  works by one author in the same year can share counts.
- **Topic overlap is a rough measure.** With only two documents there is no
  corpus to say which words are distinctive, so two papers from the same
  field always share some vocabulary. Read the label, not the percentage.
- **Some open PDFs will not download.** Publisher sites sometimes block
  automated downloads or list a dead mirror. The app reports that and the
  PDF and source-page links still work in a browser.

---

## License

MIT
