"""AVANTI -- research paper analyzer.

A single clean surface: drop in a PDF, get a structured read-out of what the
paper claims, what it measured, and what it admits it cannot do -- then ask
the paper questions and get answers cited to a page.

Everything runs locally. The Gemini deep-analysis panel appears only when an
API key is configured; without one the app is fully functional offline.
"""

from __future__ import annotations

import hashlib
import time

import requests
import streamlit as st

from avanti import analyze, extract, llm, ocr, qa, sections, sources, summarize

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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
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
      claims, then the numbers. Scanned PDFs included.
    </p>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------- pipeline


class Analysis:
    """Everything we derived from one PDF, held together for the session."""

    def __init__(self, doc, paper, brief, summary, index):
        self.doc = doc
        self.paper = paper
        self.brief = brief
        self.summary = summary
        self.index = index


def run_pipeline(
    data: bytes,
    max_pages: int,
    target_words: int,
    ocr_mode: str = "auto",
    ocr_progress=None,
) -> Analysis:
    """Extract, segment, summarize and index a PDF."""
    doc = extract.extract(
        data,
        max_pages=max_pages,
        ocr_mode=ocr_mode,
        ocr_progress=ocr_progress,
    )
    paper = sections.analyze(doc.text)
    section_map = {s.name: s.text for s in paper.sections}

    def page_lookup(sentence: str) -> int:
        return doc.page_of_offset(doc.text.find(sentence[:60]))

    brief = analyze.brief(paper.body, section_map, page_lookup)
    summary = summarize.summarize_paper(paper, target_words=target_words)
    index = qa.build_index(doc, paper)
    return Analysis(doc, paper, brief, summary, index)


def load(
    data: bytes,
    name: str,
    max_pages: int,
    target_words: int,
    ocr_mode: str = "auto",
) -> bool:
    """Analyze a PDF and store it in the session.

    Returns True if a new analysis was run, False if this exact PDF and
    settings were already loaded.
    """
    key = (
        hashlib.sha256(data).hexdigest()[:16]
        + f"-{max_pages}-{target_words}-{ocr_mode}"
    )
    if st.session_state.get("key") == key:
        return False

    t0 = time.time()
    status = st.empty()
    bar = st.empty()

    def on_ocr_page(done: int, total: int) -> None:
        # OCR runs at roughly ten seconds a page, so silent progress would
        # look like a hang. Report it.
        status.caption(
            f"No text layer found -- reading page {done} of {total} "
            f"optically ({ocr.backend_label()})."
        )
        bar.progress(done / max(1, total))

    with st.spinner("Reading the paper..."):
        st.session_state.analysis = run_pipeline(
            data, max_pages, target_words,
            ocr_mode=ocr_mode, ocr_progress=on_ocr_page,
        )
    status.empty()
    bar.empty()
    st.session_state.key = key
    st.session_state.name = name
    st.session_state.elapsed = time.time() - t0
    st.session_state.question = ""
    return True


def settings() -> tuple[int, int]:
    """The current Options values, remembered across reruns.

    The search tab needs these too, but they are rendered in the analyze
    tab, so they are kept in session state rather than read twice.
    """
    return (
        int(st.session_state.get("opt_pages", 30)),
        int(st.session_state.get("opt_words", 450)),
        st.session_state.get("opt_ocr", "auto"),
    )


# ------------------------------------------------------------- rendering


def label(text: str) -> None:
    """A small uppercase group heading."""
    st.markdown(f'<p class="group-label">{text}</p>', unsafe_allow_html=True)


def render_findings(title: str, findings, empty: str = "") -> None:
    """Render one category of the structured read-out."""
    if not findings:
        if empty:
            label(title)
            st.markdown(f'<p class="empty-note">{empty}</p>',
                        unsafe_allow_html=True)
        return
    label(title)
    for f in findings:
        cite = f' <span class="cite">p.{f.page}</span>' if f.page else ""
        st.markdown(
            f'<div class="finding">{f.text}{cite}</div>', unsafe_allow_html=True
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
            f'<div class="metric-box"><div class="metric-name">{m.name}</div>'
            f'<div class="metric-val">{m.value}{m.unit}</div></div>'
            for m in b.metrics[:8]
        )
        st.markdown(f'<div class="metric-row">{boxes}</div>',
                    unsafe_allow_html=True)
        with st.expander("Where these numbers come from"):
            for m in b.metrics[:10]:
                page = f" (p.{m.page})" if m.page else ""
                st.markdown(f"- **{m.name} {m.value}{m.unit}**{page} -- {m.sentence}")

    if b.datasets:
        label("Data used")
        st.markdown(
            " ".join(f'<span class="chip">{d}</span>' for d in b.datasets),
            unsafe_allow_html=True,
        )

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
    # here means a click fills the box and answers in the same run -- no
    # explicit rerun, which would otherwise reset the pane selection.
    cols = st.columns(2)
    for i, suggestion in enumerate(qa.suggested_questions(a.paper)):
        if cols[i % 2].button(suggestion, key=f"sug{i}", use_container_width=True):
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
                where += f" &middot; {ans.section}"
            st.markdown(
                f'<div class="finding">{ans.text}'
                f'<br><span class="cite">{where}</span></div>',
                unsafe_allow_html=True,
            )


def render_summary(a: Analysis) -> None:
    """Section-by-section summary."""
    s = a.summary

    if s.key_points:
        label("Key points")
        for point in s.key_points:
            st.markdown(f'<div class="finding">{point}</div>',
                        unsafe_allow_html=True)
        st.markdown("")

    if s.abstract:
        with st.expander("Abstract (as published)"):
            st.write(s.abstract)

    if not s.sections:
        st.info("Not enough structured text to summarize by section.")
        return

    label("By section")
    for sec in s.sections:
        with st.expander(f"{sec.label}  ({sec.source_words:,} words → {sec.word_count})"):
            st.write(sec.text)


def render_paper_header(a: Analysis) -> None:
    doc = a.doc
    title = doc.title or st.session_state.get("name", "Untitled")
    st.markdown(f'<div class="paper-title">{title}</div>', unsafe_allow_html=True)

    if doc.author_lines:
        st.markdown(
            f'<div class="paper-authors">{doc.author_lines[0][:160]}</div>',
            unsafe_allow_html=True,
        )

    pills = []
    # An OCR'd document is flagged first: it tells the reader why the text
    # may be imperfect, which changes how much to trust the read-out.
    if doc.used_ocr:
        pills.append(('Read by OCR', True))
    pills += [
        (f"{doc.pages_processed} of {doc.page_count} pages", False),
        (doc.layout, False),
        (f"{doc.word_count:,} words", False),
        (f"{len(a.paper.sections)} sections", False),
    ]
    if doc.tables:
        pills.append((f"{len(doc.tables)} tables", False))
    pills.append((f"{st.session_state.get('elapsed', 0):.1f}s", False))

    st.markdown(
        '<div class="meta-row">'
        + "".join(
            f'<span class="meta-pill{" accent" if hot else ""}">{text}</span>'
            for text, hot in pills
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    for warning in doc.warnings:
        st.warning(warning)


# ------------------------------------------------------------------- tabs

tab_analyze, tab_find = st.tabs(["Analyze a paper", "Find papers"])

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
            st.caption("OCR engine: not installed. See requirements-ocr.txt.")

    uploaded = st.file_uploader("Drop a PDF here", type=["pdf"],
                                label_visibility="collapsed")

    if uploaded is not None:
        load(uploaded.getvalue(), uploaded.name, int(max_pages),
             int(target_words), ocr_mode)

    analysis: Analysis | None = st.session_state.get("analysis")

    if notice := st.session_state.pop("notice", ""):
        st.success(notice)

    if analysis is None:
        st.caption(
            "Upload a paper to begin, or use **Find papers** to pull one "
            "straight from arXiv."
        )
    else:
        st.divider()
        render_paper_header(analysis)
        st.divider()

        # A radio rather than st.tabs: any button inside a tab triggers a
        # rerun, and st.tabs resets to its first tab on rerun -- which would
        # bounce you off the Ask pane the moment you clicked a suggestion.
        # A keyed radio remembers the selection instead.
        pane = st.radio(
            "View",
            ["At a glance", "Ask the paper", "Summary", "More"],
            horizontal=True,
            key="pane",
            label_visibility="collapsed",
        )
        st.markdown("")

        if pane == "At a glance":
            render_brief(analysis)

        elif pane == "Ask the paper":
            render_ask(analysis)

        elif pane == "Summary":
            render_summary(analysis)

        else:
            if analysis.brief.phrases:
                label("Key phrases")
                st.markdown(
                    " ".join(
                        f'<span class="chip">{p}</span>'
                        for p in analysis.brief.phrases
                    ),
                    unsafe_allow_html=True,
                )

            if analysis.doc.tables:
                label("Tables found")
                for i, table in enumerate(analysis.doc.tables[:6], start=1):
                    with st.expander(f"Table {i} ({len(table)} rows)"):
                        header, *rows = table
                        if rows:
                            st.dataframe(
                                [dict(zip(header, r)) for r in rows],
                                use_container_width=True,
                            )
                        else:
                            st.write(table)

            label("Downloads")
            d1, d2 = st.columns(2)
            d1.download_button(
                "Summary (.md)",
                analysis.summary.full_text,
                file_name="avanti-summary.md",
                use_container_width=True,
            )
            d2.download_button(
                "Extracted text (.txt)",
                analysis.doc.text,
                file_name="avanti-text.txt",
                use_container_width=True,
            )

            with st.expander("Extracted text (what the analysis actually read)"):
                st.text(analysis.doc.text[:60000])

            # Optional AI panel: only shown when a key is configured, so the
            # app stays uncluttered for offline use.
            if llm.has_key():
                st.divider()
                label("AI deep analysis (Gemini)")
                tone = st.selectbox(
                    "Style", ["Professional", "Academic", "Plain English"]
                )
                if st.button("Generate AI report"):
                    with st.spinner("Asking Gemini..."):
                        report = llm.deep_analysis(
                            analysis.paper.body, analysis.doc.title, tone
                        )
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

with tab_find:
    query = st.text_input(
        "Search arXiv", placeholder="graph neural networks for drug discovery"
    )
    limit = st.slider("Results", 3, 20, 8)

    if st.button("Search", type="primary") and query.strip():
        try:
            with st.spinner("Searching arXiv..."):
                st.session_state.results = sources.search_arxiv(query, limit)
        except requests.RequestException as e:
            st.error(f"arXiv search failed: {e}")
            st.session_state.results = []

    results = st.session_state.get("results", [])
    if results:
        st.caption(f"{len(results)} results")

    for i, r in enumerate(results):
        with st.expander(r.title):
            st.caption(f"{r.author_line}  &middot;  {r.published}")
            st.write(r.summary[:700] + ("..." if len(r.summary) > 700 else ""))

            c1, c2, c3 = st.columns(3)
            if c1.button("Analyze this", key=f"an{i}", use_container_width=True):
                try:
                    with st.spinner("Downloading PDF..."):
                        data = sources.fetch_pdf(r.pdf_url)
                    if data is None:
                        st.error("That link did not return a usable PDF.")
                    else:
                        pages, words, ocr_mode = settings()
                        load(data, r.title, pages, words, ocr_mode)
                        # Tab bodies execute in written order, and switching
                        # tabs is client-side only -- so without an explicit
                        # rerun the analyze tab would still be showing its
                        # empty state from earlier in this same run.
                        st.session_state.notice = (
                            f"Analyzed: {r.title[:70]}"
                        )
                        st.rerun()
                except requests.RequestException as e:
                    st.error(f"Download failed: {e}")
            c2.link_button("PDF", r.pdf_url, use_container_width=True)
            c3.link_button("arXiv page", r.abstract_url, use_container_width=True)
