"""Tests for bibliography parsing and citation counting.

Entries are taken from real papers' reference lists (the Transformer paper,
Adam, DiNAT) with the shapes that broke earlier versions of the parser: an
author list closed by an initial, "et al.", a title ending in a question
mark, a title starting with "A", and CVPR's trailing back-reference pages.
"""

from types import SimpleNamespace

from avanti import references as R

NUMBERED = """\
[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint
arXiv:1607.06450, 2016.
[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly
learning to align and translate. CoRR , abs/1409.0473, 2014.
[3] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural
Information Processing Systems, (NIPS) , 2016.
[4] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, et al. Mmdetection:
Open mmlab detection toolbox and benchmark. arXiv:1906.07155, 2019. 10
[5] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through
ade20k dataset. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 10, 11
"""

AUTHOR_YEAR = """\
Amari, Shun-Ichi. Natural gradient works efficiently in learning. Neural computation, 10(2):251-276, 1998.
Duchi, John, Hazan, Elad, and Singer, Yoram. Adaptive subgradient methods for online learning and stochastic
optimization. The Journal of Machine Learning Research , 12:2121-2159, 2011.
Hinton, G.E. and Salakhutdinov, R.R. Reducing the dimensionality of data with neural networks. Science , 313
(5786):504-507, 2006.
Roux, Nicolas L and Fitzgibbon, Andrew W. A fast natural newton method. In Proceedings of the 27th
International Conference on Machine Learning (ICML-10), pp. 623-630, 2010.
Zeiler, Matthew D. Adadelta: An adaptive learning rate method. arXiv preprint arXiv:1212.5701 , 2012.
"""


def _parse(text):
    return [R.parse_entry(e) for e in R.split_entries(text)]


# ------------------------------------------------------------- splitting


def test_numbered_entries_split_on_their_markers():
    refs = _parse(NUMBERED)
    assert [r.number for r in refs] == [1, 2, 3, 4, 5]
    # Wrapped lines are joined back into their entry.
    assert refs[1].text.endswith("abs/1409.0473, 2014.")


def test_author_year_entries_split_where_a_line_ends_an_entry():
    refs = _parse(AUTHOR_YEAR)
    assert len(refs) == 5
    assert refs[1].authors == "Duchi, John, Hazan, Elad, and Singer, Yoram"


def test_back_reference_pages_are_stripped():
    """CVPR prints the citing pages after an entry: "2017. 10, 11"."""
    refs = _parse(NUMBERED)
    assert refs[4].text.endswith("(CVPR), 2017.")
    assert refs[4].year == 2017


# --------------------------------------------------------------- parsing


def test_titles_are_found():
    titles = [r.title for r in _parse(NUMBERED)]
    assert titles[0] == "Layer normalization"
    assert titles[1] == "Neural machine translation by jointly learning to align and translate"


def test_et_al_ends_the_author_list():
    assert _parse(NUMBERED)[3].title.startswith("Mmdetection")


def test_a_question_mark_ends_a_title():
    assert _parse(NUMBERED)[2].title == "Can active memory replace attention?"


def test_an_initial_can_close_the_author_list():
    refs = {r.authors.split(",")[0]: r for r in _parse(AUTHOR_YEAR)}
    assert refs["Zeiler"].title == "Adadelta: An adaptive learning rate method"
    assert refs["Hinton"].title.startswith("Reducing the dimensionality")


def test_a_title_starting_with_a_is_not_an_initial():
    refs = {r.authors.split(",")[0]: r for r in _parse(AUTHOR_YEAR)}
    assert refs["Roux"].title == "A fast natural newton method"


def test_identifiers_become_links():
    refs = _parse(NUMBERED)
    assert refs[0].arxiv_id == "1607.06450"
    assert refs[0].url == "https://arxiv.org/abs/1607.06450"
    # CoRR's "abs/1409.0473" is an arXiv id too.
    assert refs[1].arxiv_id == "1409.0473"
    doi = R.parse_entry("[9] A. Author. A title here. Nature, 2020. doi:10.1038/s41586-020-1234-5.")
    assert doi.doi == "10.1038/s41586-020-1234-5"
    assert doi.url == "https://doi.org/10.1038/s41586-020-1234-5"


def test_an_entry_without_an_identifier_gets_a_search_link():
    ref = _parse(AUTHOR_YEAR)[0]
    assert ref.url == ""
    assert ref.search_url.startswith("https://scholar.google.com/scholar?q=Natural+gradient")


def test_first_author_surname_handles_both_name_orders():
    numbered, author_year = _parse(NUMBERED)[1], _parse(AUTHOR_YEAR)[1]
    assert numbered.first_author_surname == "Bahdanau"
    assert author_year.first_author_surname == "Duchi"


# ------------------------------------------------------------- citations


def test_numbered_citations_expand_ranges_and_lists():
    refs = _parse(NUMBERED)
    R.count_citations(refs, "as in [2] and [1, 2]; see also [3-5] and [2].")
    assert [r.cited for r in refs] == [1, 3, 1, 1, 1]


def test_author_year_citations_match_surname_and_year():
    refs = _parse(AUTHOR_YEAR)
    R.count_citations(
        refs,
        "Adagrad (Duchi et al., 2011) works well. Duchi et al. (2011) show it. "
        "Natural gradients (Amari, 1998). Not a citation: Duchi in 2015.",
    )
    by = {r.first_author_surname: r.cited for r in refs}
    assert by["Duchi"] == 2
    assert by["Amari"] == 1


def test_a_lettered_year_only_matches_its_own_letter():
    refs = [
        R.parse_entry("Hinton, Geoffrey. Deep networks for speech. IEEE, 2012a."),
        R.parse_entry("Hinton, Geoffrey. Improving neural networks. arXiv, 2012b."),
    ]
    R.count_citations(refs, "(Hinton et al., 2012a) and (Hinton et al., 2012b; Hinton et al., 2012b)")
    assert [r.cited for r in refs] == [1, 2]


def test_most_cited_orders_by_count_and_skips_uncited():
    refs = _parse(NUMBERED)
    R.count_citations(refs, "[3] [3] [3] [1] [1]")
    assert [r.number for r in R.most_cited(refs)] == [3, 1]


# ------------------------------------------------------------ public API


def test_extract_references_reads_the_small_print():
    """The bibliography comes from full_text, which keeps small type."""
    doc = SimpleNamespace(
        full_text="1 Introduction\nWe build on [1] and [2].\nReferences\n" + NUMBERED,
        text="1 Introduction\nWe build on [1] and [2].",
    )
    refs = R.extract_references(doc, "We build on [1] and [2].")
    assert len(refs) == 5
    assert refs[0].cited == 1 and refs[1].cited == 1


def test_no_bibliography_means_no_references():
    doc = SimpleNamespace(full_text="1 Introduction\nNo references here.", text="")
    assert R.extract_references(doc) == []


def test_shared_matches_by_identifier_or_title():
    a = _parse(NUMBERED)
    b = [
        R.parse_entry("[7] Someone Else. Layer normalization. Preprint arXiv:1607.06450, 2016."),
        R.parse_entry("[8] X. Y. Neural machine translation by jointly learning to align and translate. ICLR, 2015."),
        R.parse_entry("[9] P. Q. Something unrelated entirely. Venue, 2020."),
    ]
    pairs = R.shared(a, b)
    assert {p["a"].number for p in pairs} == {1, 2}


# ------------------------------------------------- more bibliography styles
#
# The entries below are from real arXiv papers whose reference lists the
# parser used to read as empty or as a single entry.

DOT_NUMBERED = """\
1. Maxwell, K. Tamsin, and Burkhard Schafer. "Concept and Context in Legal Information Retrieval." JURIX. 2008.
2. Danilo S. Carvalho, Minh-Tien Nguyen, Chien-Xuan Tran, Minh-Le Nguyen.
"Lexical-Morphological Modeling for Legal Text Analysis". Ninth International
Workshop on Juris-informatics (JURISIN), 2015
3. Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. Learning to rank:
from pairwise approach to listwise approach. In ICML, 2007.
"""

KEYED = """\
[Finn and et al. 2017] Finn, C., and et al. 2017. Modelagnostic meta-learning for fast adaptation of deep networks.
In ICML .
[Snell, Swersky, and Zemel 2017] Snell, J.; Swersky, K.; and
Zemel, R. 2017. Prototypical networks for few-shot learning. In NIPS, 4077-4087.
[Sung and et al. 2018] Sung, F., and et al. 2018. Learning to
compare: Relation network for few-shot learning. In CVPR.
"""

# Names in reading order, the year straight after, and a PDF that has turned
# accents into loose marks ("Ku¨c¸u¨k"). The Ritter entry shares its line
# with the one before it.
ACL = """\
Bo Han and Timothy Baldwin. 2011. Lexical normalisation of short text messages: Makn sens a# twitter. In Proceedings
of the 49th Annual Meeting of the Association for Computational Linguistics, pages 368-378.
Dilek Ku¨c¸u¨k. 2017. Joint named entity recognition and stance detection in tweets. arXiv preprint arXiv:1707.09611 .
Dog˘an Ku¨c¸u¨k, Nursal Arıcı, and Dilek Ku¨c¸u¨k. 2017. Named entity recognition in Turkish. In
Proceedings of the International Conference, pages 176-
181.
Paul Newman and Mustapha Ahmad. 1992. Hypocoristic names in hausa. Anthropological Linguistics, pages 159-172. Alan Ritter, Sam Clark, Oren Etzioni, et al. 2011. Named entity recognition in tweets: an experimental study. In Proceedings, pages 1524-1534.
"""


def test_entries_numbered_with_a_full_stop_are_split_and_numbered():
    refs = _parse(DOT_NUMBERED)
    assert [r.number for r in refs] == [1, 2, 3]
    assert refs[1].text.startswith("2. Danilo S. Carvalho")
    assert refs[1].authors.startswith("Danilo S. Carvalho")
    assert refs[1].text.endswith("2015")


def test_a_year_is_not_taken_for_an_entry_number():
    """"2008." at the start of a line is a year, not entry number 8."""
    text = DOT_NUMBERED.replace("JURIX. 2008.", "JURIX.\n2008.")
    assert [r.number for r in _parse(text)] == [1, 2, 3]


def test_keyed_entries_split_on_their_labels_and_drop_them():
    refs = _parse(KEYED)
    assert len(refs) == 3
    assert refs[1].authors == "Snell, J.; Swersky, K.; and Zemel, R"
    assert refs[1].title == "Prototypical networks for few-shot learning"
    assert refs[1].year == 2017
    assert refs[0].authors == "Finn, C., and et al"


def test_names_first_entries_split_where_the_year_follows_the_names():
    refs = _parse(ACL)
    assert [r.year for r in refs] == [2011, 2017, 2017, 1992, 2011]
    assert refs[0].authors == "Bo Han and Timothy Baldwin"
    assert refs[0].title.startswith("Lexical normalisation")
    # Accents the PDF split off do not hide an entry.
    assert refs[2].authors.startswith("Dog˘an")
    # Two entries on one line are two entries.
    assert refs[3].text.endswith("pages 159-172.")
    assert refs[4].authors.startswith("Alan Ritter")


def test_a_reference_title_that_looks_like_a_heading_does_not_end_the_list():
    from avanti import sections

    text = "\n".join([
        "1 Introduction",
        "Earlier work studied the problem at length in many settings.",
        "References",
        "Samuel R. Bowman, Gabor Angeli, and Christopher Manning. 2015. A large",
        "annotated corpus for learning natural language inference.",
        "ACM Press",
        "Daniel Cer, Mona Diab, and Eneko Agirre. 2017. Semeval-2017 task 1. In",
        "Proceedings of SemEval.",
        "Appendix",
        "Extra tables.",
    ])
    body = R.bibliography_text(sections.analyze(text))
    assert "Daniel Cer" in body and "ACM Press" in body
    assert "Extra tables" not in body


def test_bibliographical_references_is_a_bibliography_heading():
    from avanti import sections

    bucket, _ = sections.classify_heading("7. Bibliographical References")
    assert bucket == "references"
