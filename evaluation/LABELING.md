# How to label the sheets

Each paper gets one CSV file (open it in Excel). The tool's answers are
already filled in. You judge them. Budget about 20 minutes per paper, with
the PDF open next to the sheet.

Columns: `paper`, `task`, `question`, `prediction`, `verdict`, `entry`.
Only fill in `verdict` and `entry`. Do not edit the other columns, and do not
add, delete or reorder rows.

## What to do, row by row

| task | what you do |
|---|---|
| `title` | `verdict`: y if it is the paper's real title, else n |
| `dataset` | `verdict`: y if the paper really uses or evaluates on it, else n |
| `dataset_missed` | `entry`: datasets the paper uses that are not listed above, separated by `;`. Leave blank if none |
| `metric` | `verdict`: y if the paper reports this metric, else n |
| `metric_missed` | `entry`: metrics the paper reports that are not listed, separated by `;`. Blank if none |
| `finding:objective` | `verdict`: y if the sentence states the problem the paper addresses |
| `finding:contribution` | `verdict`: y if it states what the authors built or propose |
| `finding:limitation` | `verdict`: y if it states a limitation the authors admit |
| `reference_count` | `entry`: the real number of entries in the paper's reference list |
| `reference` | `verdict`: y only if authors, title and year are all correct |

A sentence that is about the right topic but quoted from a different
paper's work (for example a prior method) is an `n`.

Leave `verdict` blank only for rows you cannot judge. Blank rows are not
counted.

## Conventions

- A dataset is a named dataset or corpus. A platform ("Twitter"), a model
  ("Clarifai") or a phrase that merely contains a name is an `n`.
- The same dataset listed twice ("Mboshi5k", "Mboshi5K", or "SST" next to
  "SST-2") is `y` the first time and `n` for the repeats.
- A word that only looks like a metric ("Recall that ...") is an `n`.
- Accents that the PDF turned into loose marks ("Ku¨c¸u¨k") are tolerated in
  a reference. A cut-off name ("et a"), a missing first name, an arXiv number
  read as a year, or a title that runs into the venue is an `n`.
- A reference that really has no year or no authors is correct if the sheet
  also leaves them empty.
- Reference counts are checked against the rendered PDF pages, not the
  extracted text.

## Rules that keep the numbers honest

- Judge from the PDF, not from what seems plausible.
- Do not change the tool between labelling and scoring. If you fix a bug,
  label the new sheets again from scratch.
- Do not look at the score until all sheets are done.

## Scoring

```
python -m evaluation.labeling score --sheets path/to/sheets
```

Prints, with 95% intervals that resample papers: title accuracy; precision,
recall and F1 for datasets and metrics; precision for the problem,
contribution and limitation sentences; whether the reference count is within
10% of the real one; and how many sampled references were parsed fully
correctly.

Recall is measured against the datasets and metrics you found while
labelling, not against a complete list.

## Independent spot-check

`evaluation/spotcheck/` holds blank copies of three sheets (1905.05644,
1908.11664, 2002.06854). Fill them in yourself, from the PDFs in
`evaluation/papers_final/`, without looking at `evaluation/sheets_final/`.
Then compare:

```
python -m evaluation.labeling agree --a evaluation/sheets_final --b evaluation/spotcheck
```

It prints how often the two sets of answers match (y/n agreement, kappa,
missed-list and reference-count agreement). About an hour of work, and it is
what makes the numbers defensible.
