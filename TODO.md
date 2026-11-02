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

## D. Measure again, fairly

- [ ] **Download 10 more unseen papers.**
      Why: the 10 labelled papers were looked at while fixing, so they can no
      longer give an honest number for the parts fixed in C.
- [ ] **Label them and score.**
- [ ] **Spot-check 2 or 3 papers by hand yourself (about 1 hour).**
      Why: the labels so far were made by the assistant, not an independent person.
      Reporting agreement with your check makes the numbers defensible.
- [ ] **Update `evaluation/RESULTS.md` with the final read-out numbers.**

## E. Write and present

- [ ] **Decide the venue** (workshop or demo track first; journal needs more).
- [ ] **Draft the paper**: problem, method, experiments (tables above), limits.
- [ ] **State in the methods how labels were produced.**
- [ ] **Optional: hybrid dense retrieval as an opt-in mode in the app.**
      Why: more accurate answers, but slower on the free hosting tier.
- [ ] **Slides and a demo script.**
