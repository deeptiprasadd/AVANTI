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
