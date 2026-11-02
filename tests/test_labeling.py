"""Tests for the hand-labelling sheets and their scoring."""

from types import SimpleNamespace as NS

from evaluation import labeling


def _analysis():
    brief = NS(
        datasets=["ImageNet", "CIFAR-10", "Wikipedia"],
        metrics=[NS(name="Accuracy"), NS(name="accuracy"), NS(name="F1")],
        objective=[NS(text="Models are slow.")],
        contribution=[NS(text="We propose X."), NS(text="We also propose Y.")],
        limitation=[],
    )
    refs = [NS(authors=f"A{i}", title=f"T{i}", year=2000 + i, year_label="")
            for i in range(25)]
    return NS(title="A Paper", brief=brief, references=refs)


def _filled(verdicts, missed_datasets="", true_refs="25"):
    rows = labeling.make_rows("p1", _analysis())
    for r in rows:
        if r["task"] == "dataset_missed":
            r["entry"] = missed_datasets
        elif r["task"] == "reference_count":
            r["entry"] = true_refs
        elif r["task"] in verdicts:
            r["verdict"] = verdicts[r["task"]]
    return rows


def test_sheet_has_one_row_per_prediction_and_the_missed_rows():
    rows = labeling.make_rows("p1", _analysis())
    tasks = [r["task"] for r in rows]
    assert tasks.count("title") == 1
    assert tasks.count("dataset") == 3 and tasks.count("dataset_missed") == 1
    # "Accuracy" and "accuracy" are the same metric.
    assert tasks.count("metric") == 2
    assert tasks.count("finding:contribution") == 2
    assert tasks.count("reference") == labeling.REFERENCE_SAMPLE
    assert tasks.count("reference_count") == 1


def test_the_reference_sample_is_spread_through_the_list():
    sample = labeling._sample(list(range(100)), 10)
    assert sample == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_precision_recall_and_f1_for_datasets():
    rows = labeling.make_rows("p1", _analysis())
    # 3 predicted: 2 right, 1 wrong; the tool missed 2 more.
    answers = iter(["y", "y", "n"])
    for r in rows:
        if r["task"] == "dataset":
            r["verdict"] = next(answers)
        elif r["task"] == "dataset_missed":
            r["entry"] = "SQuAD; MNLI"
    result, papers, _ = labeling.score(rows, n_boot=50)
    assert papers == 1
    assert abs(result["dataset precision"]["value"] - 2 / 3) < 1e-9
    assert abs(result["dataset recall"]["value"] - 2 / 4) < 1e-9
    p, r = 2 / 3, 2 / 4
    assert abs(result["dataset F1"]["value"] - 2 * p * r / (p + r)) < 1e-9


def test_reference_count_is_judged_within_ten_percent():
    rows = _filled({}, true_refs="27")           # 25 vs 27 is within 10%
    result, _, _ = labeling.score(rows, n_boot=20)
    assert result["reference count within 10%"]["value"] == 1.0
    rows = _filled({}, true_refs="40")
    result, _, _ = labeling.score(rows, n_boot=20)
    assert result["reference count within 10%"]["value"] == 0.0


def test_blank_rows_are_counted_not_scored():
    rows = labeling.make_rows("p1", _analysis())
    result, _, blank = labeling.score(rows, n_boot=20)
    assert "title accuracy" not in result
    assert blank > 0


def test_sheets_round_trip_through_csv(tmp_path):
    rows = _filled({"title": "y"})
    labeling.write_sheet(rows, str(tmp_path / "p1.csv"))
    back = labeling.read_sheets(str(tmp_path))
    assert back == rows
    assert labeling.score(back, n_boot=20)[0]["title accuracy"]["value"] == 1.0
