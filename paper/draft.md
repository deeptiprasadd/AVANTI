# AVANTI: citation-grounded question answering, structured read-outs and reference parsing for research papers

Target: International Journal on Digital Libraries (Springer, journal 799).
Status: first draft. Every number below is copied from `evaluation/RESULTS.md`.
Text in [square brackets] is a note to the authors and must be resolved or
removed before submission.

## Abstract

Researchers often need to find what a paper says about one point without
reading all of it. We describe AVANTI, a web tool that reads a PDF and returns
a structured read-out (problem, contribution, datasets, metrics, limitations),
a parsed bibliography with in-text citation counts, and answers to questions
made of sentences from the paper, each cited to a page and section. Question
answering ranks sentences with BM25, adjusted for the type of question and the
section a sentence comes from, and uses no language model. On QASPER, a
benchmark of questions over NLP papers, this ranking finds the paragraphs that
annotators cited as evidence better than plain BM25 does (MRR 0.427 against
0.392 on 1,294 test questions; paired difference +0.035, 95% interval 0.025 to
0.045). A small dense retriever alone scores higher (0.487). Fusing it with
AVANTI's ranking scores highest (0.503) and beats fusing it with plain BM25
(0.483) by 0.020 (0.011 to 0.029). We also judged the read-out and the
reference parser by hand on two sets of ten papers. On the set not used while
developing the tool, metric names reached F1 0.64 and problem sentences
precision 0.78, while reference counts came within 10% of the true count for
five of ten papers. We report the reference parser as the main weakness.

Keywords: scholarly document processing, question answering, evidence
retrieval, reference parsing, digital libraries

## 1 Introduction

A reader who opens an unfamiliar paper usually wants a few things quickly: what
problem it addresses, what the authors built, which data and measures they
used, what they admit it cannot do, and what it cites. Search engines return
the paper but not these answers. Large language models can produce them, but
they can state things the paper never says, and a reader has no cheap way to
tell.

AVANTI takes a more modest route. It reads the PDF with attention to layout
(two-column papers, small-caps headings, scanned pages), finds the paper's
sections, and extracts sentences and names that match known patterns for each
of the items above. When a reader asks a question, AVANTI returns sentences
from the paper itself, each labelled with its page and section, so every answer
can be checked against the PDF. It needs no API key. The service is
available as a web application and the code is public [confirm the repository
is public and add the URL].

This paper makes three contributions.

1. A retrieval method for questions about a single paper that combines BM25
   with the question's intent and the section each sentence comes from. Its
   weights were chosen on the QASPER training split, and it is evaluated on the
   dev and test splits with an ablation of each component.
2. A comparison of that method against TF-IDF, plain BM25, a dense retriever
   and rank fusion of the two, which shows where lexical, section-aware ranking
   still adds value when a dense retriever is available.
3. A hand-judged evaluation of the structured read-out and the reference
   parser on twenty papers, including the failures.

Section 2 gives background, Section 3 describes the system, Section 4 the
retrieval method, Sections 5 and 6 the two evaluations, and Section 7 the
limits.

## 2 Related work

[AUTHORS: this section is a skeleton. Only the works below were checked
against their sources. A proper survey of scientific question answering,
scholarly document processing and citation parsing is still to be written.]

QASPER (Dasigi et al., 2021) contains 5,049 information-seeking questions over
1,585 NLP papers. The question writers saw only each paper's title and
abstract, and separate annotators answered and marked supporting evidence. We
use its evidence annotations to evaluate retrieval.

Tools that convert scholarly PDFs to structured text include GROBID, which
extracts header metadata and parses references with conditional random fields,
and ParsCit (Councill et al., 2008), a CRF reference-string parser. S2ORC (Lo et
al., 2020) provides structured full text and resolved references for millions
of open-access papers. AVANTI differs in aiming at a reader of one paper rather
than at corpus construction, and in reading the bibliography with rules rather
than a trained model.

BM25 (Robertson and Zaragoza, 2009) is the lexical baseline throughout.
Reciprocal rank fusion (Cormack et al., 2009) combines two rankings without
training. The dense retriever is a BGE model (Xiao et al., 2024).

## 3 System

AVANTI is a Python package behind a FastAPI web service. A PDF passes through
five stages.

Extraction reads text with layout awareness, so that two-column papers come out
in reading order, repairs titles and headings set in small caps, and falls back
to optical character recognition when a page has no text layer. Section
detection assigns lines to canonical sections (abstract, introduction, method,
experiments, results, discussion, conclusion, references) from their headings.

The read-out finds the problem, contribution and limitation sentences by cue
phrases scored with TF-IDF salience, searching the section where each is most
likely to appear. Dataset names come from a list of well-known names and from
capitalised names that precede "dataset", "corpus" or "benchmark". Metric names
are found where they appear in an evaluating context, with or without a number
beside them, and metric values where a number follows the name.

The reference parser splits the bibliography into entries in five styles
(numbered with brackets, numbered with a full stop, keyed by author and year,
surname-first and names-first), then separates authors, title, year and any
arXiv identifier or DOI, including the "Authors: Title." form that Springer
templates use. It counts how often each entry is cited in the body, so a reader can
see which works the paper depends on.

The question-answering index holds the paper's sentences. Section 4 describes
how it ranks them. The same service also compares two papers and searches
arXiv, OpenAlex and Europe PMC; these are outside the evaluation here.

## 4 Question answering over one paper

The index contains every sentence of at least six words outside the references
and acknowledgements. A question is scored against each sentence in four steps.

First, stop words and question words ("what", "does", "paper") are removed, and
the remaining words are stemmed. Second, each remaining word of a small
vocabulary brings in related words at a lower weight. "Problem" brings in
"challenge", "limited", "lack" and similar words, because a paper rarely
repeats the question's own wording. Own words keep weight 1 and related words
0.2. For a data question, the paper's own dataset names are also added.

Third, sentences are scored with BM25 (k1 = 1.5, b = 0.75) over the weighted
terms, and the scores are divided by the best score. A question may also match
one of six intents (problem, method, data, results, limitations, comparison).
Each intent has a set of cue phrases whose strength in a sentence is added at
weight 0.2, and a multiplicative boost for the sections that usually answer it.
Limitation questions boost the discussion by 1.7 and the conclusion by 1.5;
problem questions boost the abstract and introduction by 1.6; the other intents
have similar tables.

Fourth, sentences scoring below a quarter of the best score are dropped, and
repeated sentences are removed. The reader sees up to five sentences,
each with its page and section.

The two weights (cue strength and related words) were chosen by grid search on
the QASPER training split (cue weight in {0, 0.1, 0.2, 0.4, 0.8}, related-word
weight in {0.1, 0.2, 0.35, 0.5}; 813 papers, 2,058 questions). Mean reciprocal
rank on that split was 0.320 for plain BM25, 0.331 at the original settings
(0.8, 0.35) and 0.344 at the chosen settings (0.2, 0.2). The surface was flat
near its top, so the chosen values are not a sharp optimum. The dev split was
run after the weights were fixed, and the test split once, after dev.

## 5 Evaluation 1: evidence retrieval on QASPER

### 5.1 Protocol

For each answerable question with at least one cited text paragraph, a system
ranks the paper's paragraphs, and we check where the annotators' evidence
paragraphs fall. Figure and table evidence cannot be retrieved from text and is
ignored. Gold evidence is the union over annotators. AVANTI ranks sentences, so
a paragraph is ranked by its best sentence. This leaves 882 dev questions and
1,294 test questions. We report mean reciprocal rank (MRR), hit@k (any gold
paragraph in the top k) and recall@k, with 95% bootstrap intervals over
questions, and paired differences between systems.

Systems: TF-IDF cosine over paragraphs; plain BM25 (AVANTI with expansion,
section prior and cues switched off); AVANTI; AVANTI with each of the three
additions removed in turn; a dense retriever; dense plus plain BM25; and dense
plus AVANTI. The dense retriever embeds whole paragraphs with bge-small-en-v1.5
(a quantized ONNX build run through the fastembed library). Fusion is
reciprocal rank fusion with k = 60.

### 5.2 Results

Table 1 gives MRR on both splits and, for the test split, hit@3 and recall@10.

| System | dev MRR | test MRR | test hit@3 | test recall@10 |
|---|---|---|---|---|
| TF-IDF cosine | 0.355 | 0.384 | 0.454 | 0.597 |
| BM25 (plain) | 0.354 | 0.392 | 0.479 | 0.620 |
| AVANTI | 0.385 | 0.427 | 0.523 | 0.660 |
| AVANTI without expansion | 0.379 | 0.419 | 0.512 | 0.654 |
| AVANTI without section prior | 0.373 | 0.411 | 0.507 | 0.653 |
| AVANTI without intent cues | 0.377 | 0.423 | 0.524 | 0.650 |
| Dense | 0.431 | 0.487 | 0.571 | 0.660 |
| Dense + plain BM25 | 0.442 | 0.483 | 0.566 | 0.681 |
| Dense + AVANTI | 0.460 | 0.503 | 0.595 | 0.708 |

Table 1. Evidence retrieval on QASPER.

AVANTI improves on plain BM25 on both splits. The paired MRR difference is
+0.030 (95% interval 0.018 to 0.041) on dev and +0.035 (0.025 to 0.045) on
test. Removing related-word expansion or the section prior lowers MRR on both
splits. The intent cues add 0.008 on dev and 0.004 on test, so their
contribution is not established. The steering applies only to questions that
match one of the six intents, which was 577 of the 882 dev questions. On the
other 305, AVANTI ranks exactly as plain BM25 does.

The dense retriever alone is stronger than AVANTI alone (0.487 against 0.427 on
test). Fusion changes the picture. Dense plus AVANTI beats dense plus plain
BM25 by 0.018 MRR on dev (0.009 to 0.028) and 0.020 on test (0.011 to 0.029),
so the section-aware ranking still adds value next to a dense retriever.
Adding plain BM25 to the dense retriever helps on dev (0.431 to 0.442) and does
nothing on test (0.487 to 0.483).

## 6 Evaluation 2: the read-out and the reference parser

### 6.1 Protocol

QASPER says nothing about the read-out or the bibliography, so we judged those
by hand. We drew papers at random from the QASPER test split with a fixed seed,
in three groups of ten. The first ten were used to debug the reference parser.
The next ten (set 1) were judged and then used while improving the read-out.
The last ten (set 2) were downloaded after those improvements and not examined
until then. Set 2 is the fair measurement, and no change was made to the tool
after it was judged.

Each row the tool produced was judged against the PDF: whether the title is
right; whether each listed dataset or metric is one the paper uses; whether each
problem, contribution and limitation sentence states what it claims to; and
whether each of ten evenly spaced bibliography entries has the right authors,
title and year. Datasets and metrics the tool missed were added by the judge, so
recall is measured against what the judge found, not an exhaustive list. The
true reference count was checked on the rendered reference pages, not the
extracted text. Rules for edge cases (platforms are not datasets, accents the
PDF splits off a letter are tolerated, a year that is really an arXiv number is
wrong) are in `evaluation/LABELING.md`.

[AUTHORS, REQUIRED BEFORE SUBMISSION: state exactly who or what made these
row-by-row judgements. If any tool did, give its name, version, the dates and
the instructions it was given, as the journal's policy on automated tools
requires (see paper/submission-checklist.md). Do not describe the judgements as
made by independent human annotators unless that is true.]

### 6.2 Results

| Measure | Set 1, before fixes | Set 2, after fixes |
|---|---|---|
| Title correct | 1.00 [1.00, 1.00] | 0.90 [0.70, 1.00] |
| Contribution sentences correct | 0.82 [0.67, 1.00] | 0.85 [0.74, 0.96] |
| Problem sentences correct | 0.25 [0.00, 0.33] | 0.78 [0.43, 1.00] |
| Limitation sentences correct | 0.44 [0.13, 0.71] | none returned |
| Datasets, precision | 0.56 [0.29, 0.80] | 0.83 [0.33, 1.00] |
| Datasets, recall | 0.50 [0.22, 0.81] | 0.42 [0.11, 0.69] |
| Metrics, precision | 0.50 [0.00, 1.00] | 0.82 [0.60, 1.00] |
| Metrics, recall | 0.07 [0.00, 0.17] | 0.53 [0.26, 0.80] |
| Reference count within 10% | 0.90 [0.70, 1.00] | 0.50 [0.20, 0.80] |
| Sampled references fully correct | 0.77 [0.57, 0.91] | 0.76 [0.59, 0.89] |

Table 2. Hand-judged read-out and reference results, with 95% intervals that
resample papers.

The two sets contain different papers, so the change between them is
indicative, not a controlled comparison. Between the sets we made five changes:
metric names are found without a number beside them, and a verb ("Recall that")
is no longer read as a metric; a problem sentence must state a gap or a
question; a bare "However" no longer makes a limitation; platforms, analogies
and repeated names are dropped from the dataset list; and bibliography
entries in the "Authors: Title." form are split correctly.

Metric recall rose from 0.07 to 0.53 and problem-sentence precision from 0.25
to 0.78. Dataset precision rose to 0.83, but recall stayed low: names that do
not sit beside the word "dataset" or "corpus" (DUC 2002, MultiWOZ, IWSLT 2015)
are missed.

References did not generalise. Over both sets, 14 of 20 reference counts were
within 10% of the true count. On set 2 the count was too low for five papers
(0 of 2, 17 of 19, 21 of 28, 38 of 43 and 22 of 29). The causes we saw were an
"Authors (Year). Title." style that the parser does not read, two columns
running together, and entries cut short. Among sampled entries, errors were
truncated author lists, titles cut or merged with the next column, and an
arXiv number or a page number read as the year.

## 7 Limitations

Each set has ten papers from one field, all from arXiv, so the intervals in
Table 2 are wide and the results may not carry to other fields or to
bibliography styles common elsewhere. The first set was examined while the tool
was being improved, so only set 2 is a clean test of those improvements.

The row-by-row judgements come from a single source. A check on three papers
did not discriminate: every row the checker judged was marked correct, which
includes entries that are visibly wrong, such as an empty title. We therefore
report no agreement figure. Sentence-level judgements (is this the problem?)
are subjective, and some differences we saw between two readings of the same
sentence were reasonable on both sides.

The retrieval evaluation measures whether evidence paragraphs are ranked near
the top, not whether a reader finds the answer. It was run on NLP papers with
one small, quantized dense model, and a larger model would probably score
higher. The section boosts and intent cues were designed for six common
question types and apply to under two thirds of QASPER questions. The tool
returns existing sentences and does not combine them, so a question whose
answer spans several places gets several sentences and no synthesis.

Extraction depends on the PDF. Ligatures and accents are sometimes lost (one
title in set 2 came out as "ffi E cient"), and scanned papers rely on optical
character recognition.

## 8 Future work

The reference parser comes first. The failures in Section 6 point to specific
styles and layouts that a rule set can cover, and a trained parser such as
those cited in Section 2 could serve as a fallback. After that, a dense or
hybrid retrieval mode for the web service, kept optional because embedding a
paper costs time on the free hosting tier, and a user study that measures
whether readers find answers faster than with search alone.

## Declarations

Funding: [to complete]

Competing interests: [to complete]

Author contributions: [to complete]

Data and code availability: QASPER is public (Dasigi et al., 2021). The
evaluation code, the label sheets and the retrieval results are in the project
repository under `evaluation/` [add the repository URL and a release tag or
archive DOI]. The downloaded PDFs are not redistributed.

Use of automated tools: [REQUIRED. Complete as described in Section 6.1 and
paper/submission-checklist.md.]

## References

See `paper/references.md` for the verified list and what is still unverified.
