"""AVANTI -- research paper analyzer.

A single clean surface: drop in a PDF, get a structured read-out of what the
paper claims, what it measured, and what it admits it cannot do -- then ask
the paper questions and get answers cited to a page, see which prior work
it leans on, and compare it with another paper side by side.

Everything runs locally. The Gemini deep-analysis panel appears only when an
API key is configured; without one the app is fully functional offline.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import time

import requests
import streamlit as st

from avanti import compare, llm, ocr, pipeline, qa, references, sources
from avanti.pipeline import Analysis

st.set_page_config(
    page_title="AVANTI",
    page_icon="\U0001f52c",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Styling. Kept deliberately small: a brand header, one accent hue, and
# neutral surfaces. The top padding is generous on purpose -- Streamlit's
# toolbar is fixed to the top of the viewport and was clipping the wordmark
# clean off the page.
st.markdown(
    """
    <style>
      /* Theme handling, the hard-won version.
       *
       * Do NOT key colors off `prefers-color-scheme`. Streamlit picks its
       * own theme independently of the OS setting, and exposes no CSS
       * variable saying which it chose -- so a dark media query fired while
       * Streamlit rendered a white page, painting near-white text on white
       * and making the wordmark unreadable.
       *
       * Instead: all text inherits Streamlit's own text color, so it is
       * correct in either theme by construction. Surfaces and rules are
       * neutral translucent grey, which reads on any background. Only the
       * accent is a fixed hue, and it is used exclusively on bold text and
       * borders where its contrast holds up both ways.
       */
      :root {
        --accent: #4f46e5;
        --accent-soft: rgba(79, 70, 229, 0.09);
        --accent-line: rgba(99, 102, 241, 0.42);
        --rule: rgba(128, 128, 128, 0.26);
        --surface: rgba(128, 128, 128, 0.08);
      }

      .block-container { padding-top: 4rem !important; max-width: 47rem; }
      #MainMenu, footer { visibility: hidden; }

      /* ---------- brand header ---------- */
      .brand { display: flex; align-items: center; gap: 0.7rem; }
      .brand-mark {
        width: 2.3rem; height: 2.3rem; flex: 0 0 auto;
        border-radius: 9px; background: var(--accent);
        display: flex; align-items: center; justify-content: center;
        color: #fff; font-weight: 700; font-size: 1.15rem;
        font-family: ui-sans-serif, system-ui, sans-serif;
        box-shadow: 0 2px 10px rgba(79, 70, 229, 0.28);
      }
      .brand-name {
        font-size: 2rem !important; font-weight: 800 !important;
        letter-spacing: 0.15em !important; line-height: 1.05 !important;
        margin: 0 !important; color: inherit;
      }
      .brand-kicker {
        font-size: 0.72rem !important; font-weight: 700 !important;
        letter-spacing: 0.19em !important; text-transform: uppercase;
        color: var(--accent); margin: 0.34rem 0 0 0 !important;
        line-height: 1.2 !important;
      }
      .brand-rule {
        height: 2px; margin: 1.05rem 0 0.35rem 0; border-radius: 2px;
        background: linear-gradient(90deg, var(--accent) 0%,
                    var(--accent-line) 38%, transparent 100%);
      }
      .brand-tagline {
        color: inherit; opacity: 0.62; font-size: 0.9rem !important;
        margin: 0 0 1.5rem 0 !important; line-height: 1.5 !important;
      }

      /* ---------- analyzed paper ---------- */
      .paper-title {
        font-size: 1.32rem !important; font-weight: 700 !important;
        line-height: 1.35 !important; color: inherit;
        margin-bottom: 0.25rem !important;
      }
      .paper-authors { color: inherit; opacity: 0.62; font-size: 0.9rem; }
      .meta-row {
        display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.7rem;
      }
      .meta-pill {
        font-size: 0.76rem; color: inherit; opacity: 0.72;
        background: var(--surface); border: 1px solid var(--rule);
        border-radius: 6px; padding: 0.16rem 0.5rem;
      }
      .meta-pill.accent {
        color: var(--accent); opacity: 1; background: var(--accent-soft);
        border-color: var(--accent-line); font-weight: 600;
      }

      /* ---------- read-out ---------- */
      .group-label {
        font-size: 0.73rem !important; font-weight: 700 !important;
        letter-spacing: 0.13em !important; text-transform: uppercase;
        color: var(--accent); margin: 1.5rem 0 0.5rem 0 !important;
      }
      .finding {
        border-left: 3px solid var(--accent-line);
        background: var(--surface);
        border-radius: 0 7px 7px 0;
        padding: 0.6rem 0.85rem; margin: 0.45rem 0;
        font-size: 0.925rem; line-height: 1.6; color: inherit;
      }
      .cite {
        color: var(--accent); font-size: 0.74rem; font-weight: 600;
        letter-spacing: 0.03em;
      }

      .chip {
        display: inline-block; padding: 0.24rem 0.65rem;
        margin: 0.18rem 0.3rem 0.18rem 0;
        background: var(--surface); border: 1px solid var(--rule);
        border-radius: 999px; font-size: 0.82rem; color: inherit;
      }
      .metric-row {
        display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.35rem 0 0.4rem 0;
      }
      .metric-box {
        border: 1px solid var(--accent-line); background: var(--accent-soft);
        border-radius: 9px; padding: 0.45rem 0.8rem; min-width: 5.5rem;
      }
      .metric-name {
        font-size: 0.68rem !important; font-weight: 700 !important;
        letter-spacing: 0.1em !important; text-transform: uppercase;
        color: inherit; opacity: 0.65;
      }
      .metric-val {
        font-size: 1.18rem !important; font-weight: 700 !important;
        color: var(--accent); line-height: 1.25 !important;
      }
      .empty-note {
        color: inherit; opacity: 0.58; font-size: 0.87rem; font-style: italic;
        margin: 0.3rem 0 0.2rem 0;
      }

      /* ---------- references ---------- */
      .ref {
        border-bottom: 1px solid var(--rule); padding: 0.55rem 0;
        font-size: 0.9rem; line-height: 1.5; color: inherit;
      }
      .ref-num { opacity: 0.55; font-variant-numeric: tabular-nums; }
      .ref-title { font-weight: 600; }
      .ref-meta { opacity: 0.62; font-size: 0.8rem; }
      .ref a, .cmp a { color: var(--accent); text-decoration: none; }
      .count {
        display: inline-block; margin-left: 0.4rem; padding: 0.02rem 0.45rem;
        border-radius: 999px; background: var(--accent-soft);
        color: var(--accent); font-size: 0.72rem; font-weight: 700;
      }

      /* ---------- compare ---------- */
      .cmp-head {
        font-weight: 700; font-size: 1rem; line-height: 1.35; color: inherit;
      }
      .cmp-sub { opacity: 0.62; font-size: 0.82rem; margin-top: 0.15rem; }
      .chip.shared {
        border-color: var(--accent-line); background: var(--accent-soft);
        color: var(--accent); font-weight: 600;
      }
      .source-tag {
        display: inline-block; font-size: 0.68rem; font-weight: 700;
        letter-spacing: 0.06em; text-transform: uppercase;
        color: var(--accent); margin-right: 0.35rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Only promise scanned PDFs where OCR is actually installed -- it is not on
# Streamlit Community Cloud, whose free build cannot fit the OCR engine.
_SCANS = " Scanned PDFs included." if ocr.available() else ""

st.markdown(
    f"""
    <div class="brand">
      <div class="brand-mark">A</div>
      <div>
        <p class="brand-name">AVANTI</p>
        <p class="brand-kicker">Research Paper Analyzer</p>
      </div>
    </div>
    <div class="brand-rule"></div>
    <p class="brand-tagline">
      Reads a paper the way you would &mdash; structure first, then the
      claims, then the numbers.{_SCANS}
    </p>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------- pipeline

# Papers kept in the session at once. Each holds its text and question
# index; a handful is plenty to compare, and bounds memory on a small host.
MAX_LIBRARY = 4


def esc(value) -> str:
    """Escape text from a PDF before it goes into HTML.

    Paper text is untrusted input, and it is also just text: "p < 0.05" must
    render as characters, not as the start of a tag.
    """
    return html.escape(str(value), quote=True)


def library() -> dict:
    """Analyzed papers, oldest first: key -> {"name", "analysis", "elapsed"}."""
    return st.session_state.setdefault("library", {})


def current() -> Analysis | None:
    entry = library().get(st.session_state.get("key"))
    return entry["analysis"] if entry else None


def load(
    data: bytes,
    name: str,
    max_pages: int,
    target_words: int,
    ocr_mode: str = "auto",
) -> bool:
    """Analyze a PDF, add it to the library and make it the current paper.

    Returns False without doing anything if this exact PDF and these
    settings are already in the library -- the uploader re-submits its file
    on every rerun, and that must not re-analyze it or switch back to it.
    """
    key = (
        hashlib.sha256(data).hexdigest()[:16]
        + f"-{max_pages}-{target_words}-{ocr_mode}"
    )
    if key in library():
        return False

    t0 = time.time()
    status = st.empty()
    bar = st.progress(0.0)

    def on_progress(stage: str, fraction: float, detail: str) -> None:
        # OCR runs at roughly ten seconds a page, so silent progress would
        # look like a hang. Report every stage.
        status.caption(detail)
        bar.progress(min(1.0, fraction))

    with st.spinner("Reading the paper..."):
        analysis = pipeline.run(
            data,
            max_pages=max_pages,
            target_words=target_words,
            ocr_mode=ocr_mode,
            progress=on_progress,
        )
    status.empty()
    bar.empty()

    lib = library()
    lib[key] = {"name": name, "analysis": analysis, "elapsed": time.time() - t0}
    while len(lib) > MAX_LIBRARY:
        lib.pop(next(iter(lib)))
    st.session_state.key = key
    st.session_state.question = ""
    return True


def settings() -> tuple[int, int, str]:
    """The current Options values, remembered across reruns.

    The search tab needs these too, but they are rendered in the analyze
    tab, so they are kept in session state rather than read twice.
    """
    return (
        int(st.session_state.get("opt_pages", 30)),
        int(st.session_state.get("opt_words", 450)),
        st.session_state.get("opt_ocr", "auto"),
    )


def paper_name(key: str) -> str:
    entry = library()[key]
    title = entry["analysis"].doc.title or entry["name"]
    return title if len(title) <= 70 else title[:67] + "..."


# ------------------------------------------------------------- rendering


def label(text: str) -> None:
    """A small uppercase group heading."""
    st.markdown(f'<p class="group-label">{esc(text)}</p>', unsafe_allow_html=True)


def finding_html(text: str, page=None) -> str:
    cite = f' <span class="cite">p.{esc(page)}</span>' if page else ""
    return f'<div class="finding">{esc(text)}{cite}</div>'


def chips(items, shared=frozenset()) -> str:
    return " ".join(
        f'<span class="chip{" shared" if i.lower() in shared else ""}">{esc(i)}</span>'
        for i in items
    )


def render_findings(title: str, findings, empty: str = "") -> None:
    """Render one category of the structured read-out."""
    if not findings:
        if empty:
            label(title)
            st.markdown(f'<p class="empty-note">{esc(empty)}</p>',
                        unsafe_allow_html=True)
        return
    label(title)
    st.markdown(
        "".join(finding_html(f.text, f.page) for f in findings),
        unsafe_allow_html=True,
    )


def render_brief(a: Analysis) -> None:
    """The 'at a glance' read-out: claims, data, numbers, caveats."""
    b = a.brief

    render_findings("Problem addressed", b.objective,
                    "No explicit problem statement found.")
    render_findings("What the authors did", b.contribution)

    if b.metrics:
        label("Reported results")
        boxes = "".join(
            f'<div class="metric-box"><div class="metric-name">{esc(m.name)}</div>'
            f'<div class="metric-val">{esc(m.value)}{esc(m.unit)}</div></div>'
            for m in b.metrics[:8]
        )
        st.markdown(f'<div class="metric-row">{boxes}</div>',
                    unsafe_allow_html=True)
        with st.expander("Where these numbers come from"):
            st.markdown(
                "".join(finding_html(m.sentence, m.page) for m in b.metrics[:10]),
                unsafe_allow_html=True,
            )

    if b.datasets:
        label("Data used")
        st.markdown(chips(b.datasets), unsafe_allow_html=True)

    render_findings("Stated limitations", b.limitation,
                    "The paper states no explicit limitations.")
    render_findings("Future work", b.future)


def render_ask(a: Analysis) -> None:
    """Question answering over the paper, cited to pages."""
    st.caption(
        f"Answers are sentences from the paper itself, cited to a page "
        f"({len(a.index)} passages indexed). Nothing is generated, so nothing "
        f"is invented."
    )

    # Suggestions render before the input on purpose: writing to a keyed
    # widget's state is only allowed before that widget is created. Doing it
    # here means a click fills the box and answers in the same run.
    cols = st.columns(2)
    for i, suggestion in enumerate(qa.suggested_questions(a.paper)):
        if cols[i % 2].button(suggestion, key=f"sug{i}", width="stretch"):
            st.session_state.question = suggestion

    # A keyed widget owns its own state, so anything typed here survives the
    # reruns triggered by other controls on the page.
    question = st.text_input(
        "Ask anything about this paper",
        key="question",
        placeholder="Which baseline did they compare against?",
    )

    if question:
        answers = a.index.ask(question, k=5)
        if not answers:
            st.info("Nothing in the paper matched that. Try different wording.")
        for ans in answers:
            where = f"p.{ans.page}"
            if ans.section:
                where += f" · {ans.section}"
            st.markdown(
                f'<div class="finding">{esc(ans.text)}'
                f'<br><span class="cite">{esc(where)}</span></div>',
                unsafe_allow_html=True,
            )


def render_summary(a: Analysis) -> None:
    """Section-by-section summary."""
    s = a.summary

    if s.key_points:
        label("Key points")
        st.markdown(
            "".join(finding_html(point) for point in s.key_points),
            unsafe_allow_html=True,
        )
        st.markdown("")

    if s.abstract:
        with st.expander("Abstract (as published)"):
            st.text(s.abstract)

    if not s.sections:
        st.info("Not enough structured text to summarize by section.")
        return

    label("By section")
    for sec in s.sections:
        with st.expander(f"{sec.label}  ({sec.source_words:,} words → {sec.word_count})"):
            st.text(sec.text)


def reference_html(r: references.Reference, show_count: bool = False) -> str:
    num = f'<span class="ref-num">[{r.number}]</span> ' if r.number else ""
    title = r.title or r.text[:160]
    meta = ", ".join(x for x in (r.authors[:140], str(r.year or "")) if x)
    link = (
        f'<a href="{esc(r.url)}" target="_blank" rel="noopener">'
        f'{"arXiv" if r.arxiv_id and not r.doi else "DOI"}</a>'
        if r.url
        else f'<a href="{esc(r.search_url)}" target="_blank" rel="noopener">Find</a>'
    )
    count = (
        f'<span class="count">cited {r.cited}x</span>'
        if show_count and r.cited else ""
    )
    return (
        f'<div class="ref">{num}<span class="ref-title">{esc(title)}</span>{count}'
        f'<br><span class="ref-meta">{esc(meta)} &middot; {link}</span></div>'
    )


def references_csv(refs) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["number", "authors", "title", "year", "cited_in_text", "link"])
    for r in refs:
        writer.writerow([r.number or "", r.authors, r.title, r.year or "",
                         r.cited, r.url or r.search_url])
    return out.getvalue()


def render_references(a: Analysis) -> None:
    """The bibliography, and which of it the paper leans on most."""
    refs = a.references
    if not refs:
        st.info(
            "No reference list was found in the pages that were read. If the "
            "paper is long, raise *Pages to read* under Options -- the "
            "bibliography is usually at the end."
        )
        return

    top = references.most_cited(refs)
    if top:
        label("Most cited in the text")
        st.caption("The prior work this paper builds on, by how often it is cited.")
        st.markdown("".join(reference_html(r, True) for r in top),
                    unsafe_allow_html=True)

    label(f"All references ({len(refs)})")
    needle = st.text_input(
        "Filter references", key="ref_filter",
        placeholder="author, title or year", label_visibility="collapsed",
    ).strip().lower()
    shown = [r for r in refs if not needle or needle in r.text.lower()]
    if not shown:
        st.caption("No reference matches that filter.")
    st.markdown("".join(reference_html(r, True) for r in shown[:250]),
                unsafe_allow_html=True)
    st.download_button(
        "References (.csv)", references_csv(refs),
        file_name="avanti-references.csv", mime="text/csv",
    )


def render_more(a: Analysis) -> None:
    if a.brief.phrases:
        label("Key phrases")
        st.markdown(chips(a.brief.phrases), unsafe_allow_html=True)

    if a.doc.tables:
        label("Tables found")
        for i, table in enumerate(a.doc.tables[:6], start=1):
            with st.expander(f"Table {i} ({len(table)} rows)"):
                header, *rows = table
                if rows:
                    st.dataframe(
                        [dict(zip(header, r)) for r in rows],
                        width="stretch",
                    )
                else:
                    st.write(table)

    label("Downloads")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Summary (.md)",
        a.summary.full_text,
        file_name="avanti-summary.md",
        width="stretch",
    )
    d2.download_button(
        "Extracted text (.txt)",
        a.doc.text,
        file_name="avanti-text.txt",
        width="stretch",
    )

    with st.expander("Extracted text (what the analysis actually read)"):
        st.text(a.doc.text[:60000])

    # Optional AI panel: only shown when a key is configured, so the app
    # stays uncluttered for offline use.
    if llm.has_key():
        st.divider()
        label("AI deep analysis (Gemini)")
        tone = st.selectbox(
            "Style", ["Professional", "Academic", "Plain English"]
        )
        if st.button("Generate AI report"):
            with st.spinner("Asking Gemini..."):
                report = llm.deep_analysis(a.paper.body, a.doc.title, tone)
            st.markdown(report)
            st.download_button(
                "Download AI report (.md)", report,
                file_name="avanti-ai-report.md",
            )
    else:
        st.divider()
        st.caption(
            "Optional: add a Gemini key to `.streamlit/secrets.toml` "
            "for an AI-written report. Everything above works without one."
        )


def render_paper_header(a: Analysis) -> None:
    doc = a.doc
    entry = library()[st.session_state.key]
    title = doc.title or entry["name"] or "Untitled"
    st.markdown(f'<div class="paper-title">{esc(title)}</div>',
                unsafe_allow_html=True)

    if doc.author_lines:
        st.markdown(
            f'<div class="paper-authors">{esc(doc.author_lines[0][:160])}</div>',
            unsafe_allow_html=True,
        )

    pills = []
    # An OCR'd document is flagged first: it tells the reader why the text
    # may be imperfect, which changes how much to trust the read-out.
    if doc.used_ocr:
        pills.append(("Read by OCR", True))
    pills += [
        (f"{doc.pages_processed} of {doc.page_count} pages", False),
        (doc.layout, False),
        (f"{doc.word_count:,} words", False),
        (f"{len(a.paper.sections)} sections", False),
    ]
    if doc.tables:
        pills.append((f"{len(doc.tables)} tables", False))
    if a.references:
        pills.append((f"{len(a.references)} references", False))
    pills.append((f"{entry['elapsed']:.1f}s", False))

    st.markdown(
        '<div class="meta-row">'
        + "".join(
            f'<span class="meta-pill{" accent" if hot else ""}">{esc(text)}</span>'
            for text, hot in pills
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    for warning in doc.warnings:
        st.warning(warning)


def render_compare() -> None:
    """Two papers from the library, side by side."""
    lib = library()
    if len(lib) < 2:
        st.info(
            "Analyze two papers -- upload them, or use **Find papers** -- and "
            "compare them here: problem, method, data and results side by "
            "side, plus the prior work both build on."
        )
        if lib:
            st.caption(f"In this session so far: {paper_name(next(iter(lib)))}")
        return

    keys = list(lib)
    first = st.session_state.get("key") if st.session_state.get("key") in lib else keys[0]
    c1, c2 = st.columns(2)
    a_key = c1.selectbox("First paper", keys, index=keys.index(first),
                         format_func=paper_name)
    others = [k for k in keys if k != a_key]
    b_key = c2.selectbox("Second paper", others, format_func=paper_name)

    c = compare.compare(lib[a_key]["analysis"], lib[b_key]["analysis"])

    st.markdown(
        f'<div class="meta-row"><span class="meta-pill accent">{esc(c.relation)}'
        f'</span><span class="meta-pill">{c.similarity:.0%} topic overlap'
        f'</span></div>',
        unsafe_allow_html=True,
    )
    if c.a_cites_b:
        st.caption("The first paper cites the second.")
    if c.b_cites_a:
        st.caption("The second paper cites the first.")

    shared_data = {d.lower() for d in c.shared_datasets}
    shared_metric = {m[0].lower() for m in c.shared_metrics}

    def side(col, s) -> None:
        with col:
            st.markdown(
                f'<div class="cmp-head">{esc(s.title)}</div>'
                f'<div class="cmp-sub">{esc(s.authors[:120])}</div>'
                f'<div class="cmp-sub">{s.pages} pages &middot; {s.words:,} words'
                f' &middot; {s.reference_count} references</div>',
                unsafe_allow_html=True,
            )
            for title, items, empty in (
                ("Problem", s.problem, "No explicit problem statement found."),
                ("Method", s.method, "No explicit contribution statement found."),
                ("Limitations", s.limitations, "No limitations stated."),
            ):
                label(title)
                st.markdown(
                    "".join(finding_html(t) for t in items)
                    or f'<p class="empty-note">{esc(empty)}</p>',
                    unsafe_allow_html=True,
                )
            label("Data used")
            st.markdown(chips(s.datasets, shared_data)
                        or '<p class="empty-note">None detected.</p>',
                        unsafe_allow_html=True)
            label("Reported results")
            metrics = [f"{k} {v}" for k, v in list(s.metrics.items())[:8]]
            st.markdown(
                chips(metrics, {f"{k} {v}".lower() for k, v in s.metrics.items()
                                if k.lower() in shared_metric})
                or '<p class="empty-note">None detected.</p>',
                unsafe_allow_html=True,
            )

    left, right = st.columns(2, gap="large")
    side(left, c.a)
    side(right, c.b)

    st.divider()
    label("In common")
    if c.shared_metrics:
        st.markdown("Metrics both report -- the place to read their numbers "
                    "against each other:")
        st.dataframe(
            [{"Metric": n, "First paper": va, "Second paper": vb}
             for n, va, vb in c.shared_metrics],
            width="stretch", hide_index=True,
        )
    if c.shared_datasets:
        st.markdown(f"**Shared data:** {esc(', '.join(c.shared_datasets))}")
    if c.shared_phrases:
        st.markdown(f"**Shared key phrases:** {esc(', '.join(c.shared_phrases))}")
    if c.shared_references:
        st.markdown(f"**Prior work both cite ({len(c.shared_references)}):**")
        st.markdown("".join(finding_html(t) for t in c.shared_references[:15]),
                    unsafe_allow_html=True)
    if not (c.shared_metrics or c.shared_datasets or c.shared_phrases
            or c.shared_references):
        st.caption("No shared data, metrics or references were found.")


# ------------------------------------------------------------------- tabs

# A tab change requested by code (after "Analyze this", say) is applied here,
# before the tabs widget exists -- its state cannot be written once it does.
if goto := st.session_state.pop("goto_tab", None):
    st.session_state.main_tab = goto

# Stateful tabs: they keep their selection across reruns, and a hidden tab's
# body is skipped entirely rather than rendered off-screen.
tab_analyze, tab_find, tab_compare = st.tabs(
    ["Analyze a paper", "Find papers", "Compare"],
    key="main_tab", on_change="rerun",
)

if tab_analyze.open:
    with tab_analyze:
        with st.expander("Options"):
            c1, c2 = st.columns(2)
            max_pages = c1.number_input(
                "Pages to read", 1, 200, 30, step=5, key="opt_pages"
            )
            target_words = c2.number_input(
                "Summary length (words)", 150, 1500, 450, step=50, key="opt_words"
            )
            ocr_mode = st.radio(
                "Scanned PDFs",
                ["auto", "always", "never"],
                horizontal=True,
                key="opt_ocr",
                help=(
                    "auto: read a page optically only when it has no text layer. "
                    "always: ignore the text layer and OCR everything. "
                    "never: skip OCR entirely."
                ),
            )
            if ocr.available():
                st.caption(f"OCR engine: {ocr.backend_label()} -- about 10s per page.")
            else:
                st.caption(
                    "OCR engine: not installed on this deployment, so a scanned "
                    "PDF is reported rather than read."
                )

        uploaded = st.file_uploader("Drop a PDF here", type=["pdf"],
                                    label_visibility="collapsed")

        if uploaded is not None:
            load(uploaded.getvalue(), uploaded.name, int(max_pages),
                 int(target_words), ocr_mode)

        if notice := st.session_state.pop("notice", ""):
            st.success(notice)

        analysis = current()
        if analysis is None:
            st.caption(
                "Upload a paper to begin, or use **Find papers** to search "
                "arXiv, OpenAlex and Europe PMC."
            )
        else:
            keys = list(library())
            if len(keys) > 1:
                choice = st.selectbox(
                    "Paper", keys, index=keys.index(st.session_state.key),
                    format_func=paper_name,
                    help="Papers analyzed this session. Compare any two in the "
                         "Compare tab.",
                )
                if choice != st.session_state.key:
                    st.session_state.key = choice
                    st.session_state.question = ""
                    st.rerun()

            st.divider()
            render_paper_header(analysis)
            st.divider()

            panes = st.tabs(
                ["At a glance", "Ask the paper", "Summary", "References", "More"],
                key="pane", on_change="rerun",
            )
            renderers = (render_brief, render_ask, render_summary,
                         render_references, render_more)
            for pane, render in zip(panes, renderers):
                if pane.open:
                    with pane:
                        render(analysis)

if tab_find.open:
    with tab_find:
        query = st.text_input(
            "Search", placeholder="graph neural networks for drug discovery"
        )
        c1, c2 = st.columns([3, 2])
        source = c1.selectbox(
            "Source",
            ["all", *sources.SOURCES],
            format_func=lambda s: "All sources" if s == "all" else sources.SOURCES[s],
            help="arXiv: preprints. OpenAlex: every field, open-access PDFs only. "
                 "Europe PMC: biomedical and life sciences.",
        )
        limit = c2.number_input("Results", 1, sources.MAX_RESULTS, 10, step=5)

        if st.button("Search", type="primary") and query.strip():
            try:
                with st.spinner("Searching..."):
                    found = sources.search(query, int(limit), source)
                st.session_state.results = found.results
                st.session_state.search_failed = found.failed
            except requests.RequestException as e:
                st.error(f"Search failed: {e}")
                st.session_state.results = []
                st.session_state.search_failed = {}

        for name in st.session_state.get("search_failed", {}):
            st.warning(f"{name} could not be reached; showing the other sources.")

        results = st.session_state.get("results", [])
        if results:
            st.caption(f"{len(results)} results")

        for i, r in enumerate(results):
            with st.expander(r.title):
                meta = " · ".join(x for x in (r.author_line, r.published) if x)
                st.markdown(
                    f'<span class="source-tag">{esc(r.source)}</span>'
                    f'<span class="ref-meta">{esc(meta)}</span>',
                    unsafe_allow_html=True,
                )
                if r.summary:
                    st.write(r.summary[:700] + ("..." if len(r.summary) > 700 else ""))

                c1, c2, c3 = st.columns(3)
                if c1.button("Analyze this", key=f"an{i}", width="stretch"):
                    try:
                        with st.spinner("Downloading PDF..."):
                            data = sources.fetch_pdf(r.pdf_url)
                        if data is None:
                            st.error("That link did not return a usable PDF.")
                        else:
                            pages, words, ocr_mode = settings()
                            load(data, r.title, pages, words, ocr_mode)
                            st.session_state.notice = f"Analyzed: {r.title[:70]}"
                            st.session_state.goto_tab = "Analyze a paper"
                            st.rerun()
                    except (requests.RequestException, sources.UnsafeURL) as e:
                        st.error(f"Download failed: {e}")
                c2.link_button("PDF", r.pdf_url, width="stretch")
                c3.link_button("Source page", r.abstract_url, width="stretch")

if tab_compare.open:
    with tab_compare:
        render_compare()
