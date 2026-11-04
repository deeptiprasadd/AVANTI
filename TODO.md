# AVANTI: what is being done, and why

The goal: turn AVANTI from a working tool into a journal paper whose claims
are backed by measurements. Each item says what it is and why it matters.
Tick a box when the item is finished.

## A. Prove the question answering works

- [x] **Build an evaluation on QASPER** (`evaluation/qasper.py`).
      Why: a paper needs numbers against a public benchmark, not just a demo.
- [x] **Compare against plain baselines** (TF-IDF, plain BM25).
      Why: shows whether AVANTI's ranking is actually better than the simple way.
- [x] **Tune the ranking weights on the training split only.**
      Why: tuning on the data you report on would inflate the result.
- [x] **Add a modern dense-retrieval baseline and a hybrid.**
      Why: a reviewer will ask "what about embeddings?". The hybrid is the best system.
- [x] **Run once on the untouched test split.**
      Why: the final, fair number. Result: AVANTI + dense beats dense + plain BM25 by about 0.02 MRR, with intervals that exclude zero.
- [x] **Write the results down** (`evaluation/RESULTS.md`).

## B. Prove the other features work (read-out and references)

- [x] **Build hand-labelling sheets and a scorer** (`evaluation/labeling.py`).
      Why: QASPER only tests question answering. Title, datasets, metrics,
      problem/contribution/limitation sentences and references need their own check.
- [x] **Fix bibliography reading** (4 reference styles, split-off headings, accents).
      Why: half the papers had 0 or 1 references found.
- [x] **Set aside 10 unseen papers and label them.**
      Why: scoring papers the code was tuned on gives flattering numbers.
- [x] **Score them.** Weak spots found: metrics (recall 0.07), problem
      sentences (0.25), limitation sentences (0.44), datasets (F1 0.53).

## C. Fix the weak spots found in B  (done; effect not yet measured, see D)

- [x] **Metric extractor:** now reports metric names the paper uses (accuracy,
      F1, BLEU, EER, hits@k...), not only ones that sit next to a number;
      "Recall that..." is no longer a metric. Reads small-caps "B LEU" too.
- [x] **Problem sentences:** now prefers sentences that state a gap or ask a
      question; drops "we did X", results and background about the topic.
- [x] **Limitation sentences:** a bare "However..." no longer counts; criticism
      of earlier work and rebuttals are penalised.
- [x] **References in `Authors: Title. In: Venue. (Year)` style.**
      Why: it failed on every sampled entry.
- [x] **Dataset false positives:** platforms ("Twitter"), analogies
      ("like Penn Treebank"), duplicates and one-word adjectives are dropped.
- [x] **Tests for each fix**, built from the real sentences that failed.
- [x] **Check question answering did not change** (still 0.385 MRR on dev).

Known remaining weak spots, to be honest about in the paper:
objective sentences are still sometimes background; datasets can still list
class names ("Legitimate", "Malicious"); metric names in survey papers are
mentions, not results.

## D. Measure again, fairly  (done except your spot-check)

- [x] **Download 10 more unseen papers** (`evaluation/papers_final/`, not in git).
      Why: the first 10 had been looked at while fixing.
- [x] **Label them and score** (`evaluation/sheets_final/`).
      Result: metrics F1 0.13 -> 0.64, problem sentences 0.25 -> 0.78,
      dataset precision 0.56 -> 0.83. Reference counts did not hold up:
      0.50 within 10% on the new papers.
- [x] **Spot-check 3 papers** (`evaluation/spotcheck/`). Filled in, but every
      judged row was "y", so kappa is 0.000 and it cannot validate the labels.
      The paper reports no agreement figure and says so (draft, Section 7).
- [x] **Update `evaluation/RESULTS.md`** with both sets and the limits.

## D2. Problems the fair test found (decide whether to fix before writing)

- [ ] **Reference counts too low on 5 of 10 new papers.** Causes seen:
      `Authors (Year). Title.` style, two columns running together, entries
      cut short, a 2-entry list with no markers.
- [ ] **Dataset recall 0.42:** names without "dataset" beside them are missed.
- [ ] **Limitation sentences:** none returned on the new papers; cannot be assessed.
- [ ] **Titles with ligatures** extract as "ffi E cient".
- Note: fixing any of these means the 20 labelled papers are used up for that
  part; a third set would be needed for an honest number.

## E. Write and present  <- in progress (target: International Journal on Digital Libraries, journal 799)

- [x] **Decide the venue:** International Journal on Digital Libraries.
- [x] **First draft** in `paper/draft.md`: all numbers copied from RESULTS.md.
- [x] **Verified reference list** started in `paper/references.md`.
- [x] **Submission checklist** in `paper/submission-checklist.md`.
- [ ] **Read the journal's scope and submission guidelines in a browser**
      (the pages could not be read automatically): article type, length,
      template, open access fees.
- [ ] **Complete the disclosure placeholders** (draft Section 6.1 and
      Declarations). Required by the publisher's policy on automated tools.
- [ ] **Write the literature review** (Section 2) and confirm all author lists.
- [ ] **Decide about the reference parser** (D2) before submission.
- [ ] **Figures:** pipeline diagram, plot of Table 1.
- [ ] **Convert to the journal template**, add repository URL and archived release.
- [ ] **Cover letter.**
- [ ] **Optional: hybrid dense retrieval as an opt-in mode in the app.**
- [ ] **Slides and a demo script.**
