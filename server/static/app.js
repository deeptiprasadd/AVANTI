/* AVANTI frontend.
 *
 * No framework and no build step: the UI is three files served straight
 * from the API container. The app is a small state machine over three
 * views (intro, progress, result) driven by the job API -- upload returns
 * a job id, we poll it, then render the result.
 */
"use strict";

/* ─────────────────────────── utilities ─────────────────────────── */

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Escape anything that came from a PDF before it reaches innerHTML.
 *  Paper text is untrusted input: a title containing "<img onerror=...>"
 *  must render as characters, not markup. */
function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function toast(message, kind = "err") {
  const el = document.createElement("div");
  el.className = `toast ${kind === "ok" ? "ok" : ""}`;
  el.textContent = message;
  $("#toasts").append(el);
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 300);
  }, kind === "ok" ? 3200 : 6500);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) {
        detail = typeof body.detail === "string"
          ? body.detail
          : (body.detail[0]?.msg ?? detail);
      }
    } catch { /* non-JSON error body; keep the status line */ }
    throw new Error(detail);
  }
  return response.json();
}

/* ──────────────────────────── theme ────────────────────────────── */

const Theme = {
  init() {
    let saved = null;
    try { saved = localStorage.getItem("avanti-theme"); } catch { /* blocked */ }
    const system = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    this.set(saved || system, false);
    $("#theme-toggle").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      this.set(next, true);
    });
  },
  set(theme, persist) {
    document.documentElement.dataset.theme = theme;
    if (persist) {
      try { localStorage.setItem("avanti-theme", theme); } catch { /* blocked */ }
    }
  },
};

/* ──────────────────────────── state ────────────────────────────── */

const state = {
  capabilities: null,
  ocrMode: "auto",
  jobId: null,
  result: null,
  polling: null,
  cancelled: false,
};

/* Jobs this browser analyzed, so Compare can offer them. A job id is the
 * key to an analysis, so the server deliberately has no endpoint listing
 * everyone's -- each tab remembers its own, for its session only. */
const MyJobs = {
  KEY: "avanti-jobs",
  all() {
    try { return JSON.parse(sessionStorage.getItem(this.KEY)) || []; }
    catch { return []; }
  },
  add(id, title) {
    const list = this.all().filter(j => j.id !== id);
    list.push({ id, title });
    this.save(list.slice(-6));
  },
  remove(id) { this.save(this.all().filter(j => j.id !== id)); },
  save(list) {
    try { sessionStorage.setItem(this.KEY, JSON.stringify(list)); }
    catch { /* storage blocked: Compare just offers fewer papers */ }
  },
};

const spinnerNote = text => `<p class="empty-note"><span class="spinner"
  style="display:inline-block;vertical-align:-2px"></span> ${esc(text)}</p>`;

function showView(name) {
  for (const view of ["intro", "progress", "result"]) {
    $(`#view-${view}`).hidden = view !== name;
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ────────────────────────── capabilities ───────────────────────── */

async function loadCapabilities() {
  try {
    state.capabilities = await api("/api/health");
  } catch {
    toast("Could not reach the server. Is it running?");
    return;
  }
  const caps = state.capabilities;

  $("#dz-sub").textContent = `PDF, up to ${caps.limits.maxUploadMb} MB`;

  const pill = $("#cap-ocr");
  if (caps.ocr.available) {
    pill.textContent = "OCR ready";
    pill.hidden = false;
    $("#ocr-note").textContent =
      `Scanned pages are read with ${caps.ocr.backend}, roughly ` +
      `${caps.ocr.secondsPerPage}s per page.`;
  } else {
    $("#ocr-note").textContent =
      "OCR is not installed on this server, so a scanned PDF will be " +
      "reported rather than read.";
  }
}

/* ──────────────────────────── upload ───────────────────────────── */

function readOptions() {
  const clamp = (v, lo, hi, dflt) => {
    const n = parseInt(v, 10);
    return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt;
  };
  return {
    maxPages: clamp($("#opt-pages").value, 1, 60, 30),
    targetWords: clamp($("#opt-words").value, 150, 1500, 450),
    ocrMode: state.ocrMode,
  };
}

async function analyzeFile(file) {
  if (!file) return;
  if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") {
    toast("That is not a PDF.");
    return;
  }
  const opts = readOptions();
  const form = new FormData();
  form.append("file", file);
  form.append("max_pages", opts.maxPages);
  form.append("target_words", opts.targetWords);
  form.append("ocr_mode", opts.ocrMode);

  startProgress(file.name);
  try {
    const job = await api("/api/analyze", { method: "POST", body: form });
    pollJob(job.id);
  } catch (err) {
    failProgress(err.message);
  }
}

async function analyzeUrl(url, title) {
  const opts = readOptions();
  startProgress(title || url);
  try {
    const job = await api("/api/analyze-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url, title: title || "",
        max_pages: opts.maxPages,
        target_words: opts.targetWords,
        ocr_mode: opts.ocrMode,
      }),
    });
    pollJob(job.id);
  } catch (err) {
    failProgress(err.message);
  }
}

/* ─────────────────────────── progress ──────────────────────────── */

function startProgress(name) {
  state.cancelled = false;
  $("#progress-file").textContent = name;
  $("#progress-stage").textContent = "Reading the paper…";
  $("#progress-detail").textContent = "";
  $("#progress-hint").textContent = "";
  setBar(0.02);
  showView("progress");
}

function setBar(fraction) {
  const pct = Math.round(fraction * 100);
  $("#progress-bar").style.width = `${pct}%`;
  $("#progress-bar-outer").setAttribute("aria-valuenow", String(pct));
}

const STAGE_TEXT = {
  queued: "Queued…",
  starting: "Starting…",
  extracting: "Reading the PDF",
  sectioning: "Finding sections",
  analyzing: "Extracting claims and metrics",
  summarizing: "Summarizing by section",
  indexing: "Indexing for questions",
  done: "Done",
};

function pollJob(jobId) {
  state.jobId = jobId;
  let delay = 350;

  const tick = async () => {
    if (state.cancelled) return;
    let status;
    try {
      status = await api(`/api/jobs/${jobId}`);
    } catch (err) {
      failProgress(err.message);
      return;
    }

    setBar(status.progress);
    $("#progress-stage").textContent = STAGE_TEXT[status.stage] ?? status.stage;
    $("#progress-detail").textContent = status.detail || "";

    // OCR is the slow path, so say so rather than looking stuck.
    if (status.detail.includes("optically")) {
      $("#progress-hint").textContent =
        "This PDF has no text layer, so each page is being read optically. " +
        "That takes about ten seconds a page.";
    }

    if (status.status === "done") {
      loadResult(jobId);
      return;
    }
    if (status.status === "error") {
      failProgress(status.error || "Analysis failed.");
      return;
    }
    delay = Math.min(delay * 1.15, 1500);   // back off while work continues
    state.polling = setTimeout(tick, delay);
  };

  tick();
}

function failProgress(message) {
  toast(message);
  showView("intro");
}

async function loadResult(jobId) {
  try {
    state.result = await api(`/api/jobs/${jobId}/result`);
  } catch (err) {
    failProgress(err.message);
    return;
  }
  MyJobs.add(jobId, state.result.document.title || state.result.job?.name
                    || "Untitled paper");
  renderResult(state.result);
  showView("result");
}

/* ──────────────────────────── render ───────────────────────────── */

function findingHtml(f) {
  const cite = f.page ? `<span class="cite">p.${esc(f.page)}</span>` : "";
  return `<div class="finding">${esc(f.text)}${cite}</div>`;
}

function group(label, inner) {
  return `<div class="group"><p class="group-label">${esc(label)}</p>${inner}</div>`;
}

function renderResult(data) {
  const doc = data.document;

  $("#paper-title").textContent = doc.title || "Untitled paper";
  $("#paper-authors").textContent = (doc.authors || [])[0] || "";

  const pills = [];
  if (doc.usedOcr) pills.push(["Read by OCR", true]);
  pills.push([`${doc.pagesProcessed} of ${doc.pageCount} pages`, false]);
  if (doc.layout) pills.push([doc.layout, false]);
  pills.push([`${doc.wordCount.toLocaleString()} words`, false]);
  pills.push([`${data.sections.length} sections`, false]);
  if (doc.tableCount) pills.push([`${doc.tableCount} tables`, false]);
  if (data.job?.elapsed) pills.push([`${data.job.elapsed}s`, false]);

  $("#paper-meta").innerHTML = pills
    .map(([text, hot]) =>
      `<span class="meta-pill ${hot ? "hot" : ""}">${esc(text)}</span>`)
    .join("");

  $("#paper-warnings").innerHTML = (doc.warnings || [])
    .map(w => `<div class="notice notice-warn">${esc(w)}</div>`).join("");

  renderGlance(data);
  renderAsk(data);
  renderSummary(data);
  renderReferences(data);
  renderCompare();
  renderDetails(data);
  selectTab("glance");
}

function renderGlance(data) {
  const b = data.brief;
  const parts = [];

  parts.push(group("Problem addressed", b.objective.length
    ? b.objective.map(findingHtml).join("")
    : `<p class="empty-note">No explicit problem statement found.</p>`));

  if (b.contribution.length) {
    parts.push(group("What the authors did", b.contribution.map(findingHtml).join("")));
  }

  if (b.metrics.length) {
    const boxes = b.metrics.slice(0, 8).map(m => `
      <div class="metric-box">
        <div class="metric-name">${esc(m.name)}</div>
        <div class="metric-val">${esc(m.display)}</div>
      </div>`).join("");
    const sources = b.metrics.slice(0, 10).map(m =>
      `<div class="finding">${esc(m.sentence)}${m.page ? `<span class="cite">p.${esc(m.page)}</span>` : ""}</div>`
    ).join("");
    parts.push(group("Reported results",
      `<div class="metric-row">${boxes}</div>
       <details class="acc" style="margin-top:.8rem">
         <summary>Where these numbers come from</summary>
         <div class="acc-body">${sources}</div>
       </details>`));
  }

  if (b.datasets.length) {
    parts.push(group("Data used",
      `<div class="chips">${b.datasets.map(d => `<span class="chip">${esc(d)}</span>`).join("")}</div>`));
  }

  parts.push(group("Stated limitations", b.limitation.length
    ? b.limitation.map(findingHtml).join("")
    : `<p class="empty-note">The paper states no explicit limitations.</p>`));

  if (b.future.length) {
    parts.push(group("Future work", b.future.map(findingHtml).join("")));
  }

  $("#pane-glance").innerHTML = parts.join("");
}

function renderAsk(data) {
  const suggestions = (data.suggestedQuestions || [])
    .map(q => `<button class="suggestion" type="button">${esc(q)}</button>`).join("");

  $("#pane-ask").innerHTML = `
    <p class="ask-intro">
      Answers are sentences from the paper itself, cited to a page
      (${esc(data.passageCount)} passages indexed). Nothing is generated,
      so nothing is invented.
    </p>
    <div class="suggestions">${suggestions}</div>
    <form class="ask-row" id="ask-form">
      <input type="text" id="ask-input" autocomplete="off"
             placeholder="Which baseline did they compare against?"
             aria-label="Ask a question about this paper">
      <button class="btn btn-primary" type="submit">Ask</button>
    </form>
    <div id="ask-answers"></div>`;

  $("#ask-form").addEventListener("submit", e => {
    e.preventDefault();
    askQuestion($("#ask-input").value);
  });
  $$(".suggestion", $("#pane-ask")).forEach(btn =>
    btn.addEventListener("click", () => {
      $("#ask-input").value = btn.textContent;
      askQuestion(btn.textContent);
    }));
}

async function askQuestion(question) {
  question = (question || "").trim();
  if (question.length < 2) return;

  const out = $("#ask-answers");
  out.innerHTML = `<p class="empty-note"><span class="spinner"
     style="display:inline-block;vertical-align:-2px"></span> Searching…</p>`;

  try {
    const data = await api(`/api/jobs/${state.jobId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: 5 }),
    });
    if (!data.answers.length) {
      out.innerHTML = `<p class="empty-note">Nothing in the paper matched
        that. Try different wording.</p>`;
      return;
    }
    out.innerHTML = data.answers.map(a => `
      <div class="finding">${esc(a.text)}
        <span class="answer-src">page ${esc(a.page)}${
          a.section ? " · " + esc(a.section) : ""}</span>
      </div>`).join("");
  } catch (err) {
    out.innerHTML = "";
    toast(err.message);
  }
}

function renderSummary(data) {
  const s = data.summary;
  const parts = [];

  if (s.keyPoints.length) {
    parts.push(group("Key points",
      s.keyPoints.map(p => `<div class="finding">${esc(p)}</div>`).join("")));
  }
  if (s.abstract) {
    parts.push(`<details class="acc"><summary>Abstract (as published)</summary>
      <div class="acc-body">${esc(s.abstract)}</div></details>`);
  }
  if (s.sections.length) {
    parts.push(group("By section", s.sections.map(sec => `
      <details class="acc">
        <summary>${esc(sec.label)}
          <span class="acc-count">${esc(sec.sourceWords.toLocaleString())} words
            → ${esc(sec.words)}</span>
        </summary>
        <div class="acc-body">${esc(sec.text)}</div>
      </details>`).join("")));
  } else {
    parts.push(`<p class="empty-note">Not enough structured text to
      summarize by section.</p>`);
  }
  $("#pane-summary").innerHTML = parts.join("");
}

function renderDetails(data) {
  const parts = [];

  if (data.brief.phrases.length) {
    parts.push(group("Key phrases",
      `<div class="chips">${data.brief.phrases
        .map(p => `<span class="chip">${esc(p)}</span>`).join("")}</div>`));
  }

  if (data.sections.length) {
    parts.push(group("Sections detected",
      `<div class="chips">${data.sections
        .map(s => `<span class="chip">${esc(s.label)} · ${esc(s.words.toLocaleString())}w</span>`)
        .join("")}</div>`));
  }

  if (data.tables.length) {
    const tables = data.tables.map((t, i) => {
      const [header, ...rows] = t.rows;
      return `<details class="acc">
        <summary>Table ${i + 1}<span class="acc-count">${esc(t.rowCount)} rows</span></summary>
        <div class="acc-body"><div class="table-wrap"><table class="data">
          <thead><tr>${(header || []).map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead>
          <tbody>${rows.slice(0, 15).map(r =>
            `<tr>${r.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody>
        </table></div></div></details>`;
    }).join("");
    parts.push(group("Tables found", tables));
  }

  parts.push(group("Downloads", `
    <div class="dl-row">
      <button class="btn btn-ghost btn-sm" id="dl-summary">Summary (.md)</button>
      <button class="btn btn-ghost btn-sm" id="dl-text">Extracted text (.txt)</button>
      <button class="btn btn-ghost btn-sm" id="dl-json">Full analysis (.json)</button>
    </div>`));

  if (state.capabilities?.aiReport) {
    parts.push(group("AI deep analysis", `
      <p class="ask-intro">An optional Gemini write-up. Everything above is
        computed locally and does not use it.</p>
      <button class="btn btn-ghost btn-sm" id="ai-btn">Generate report</button>
      <div id="ai-out" style="margin-top:.8rem"></div>`));
  }

  parts.push(group("Extracted text",
    `<p class="ask-intro">Exactly what the analysis read.</p>
     <pre class="raw" id="raw-text">Loading…</pre>`));

  $("#pane-details").innerHTML = parts.join("");
  wireDetails(data);
}

function download(filename, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type: `${type};charset=utf-8` }));
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function slugOf(data) {
  return (data.document.title || "paper")
    .toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60)
    || "paper";
}

function wireDetails(data) {
  const slug = slugOf(data);

  $("#dl-summary").addEventListener("click", () =>
    download(`${slug}-summary.md`, data.summary.markdown, "text/markdown"));
  $("#dl-json").addEventListener("click", () =>
    download(`${slug}-analysis.json`, JSON.stringify(data, null, 2), "application/json"));

  // The extracted text can be large, so it is fetched on demand.
  let rawText = null;
  const loadRaw = async () => {
    if (rawText !== null) return rawText;
    const body = await api(`/api/jobs/${state.jobId}/text`);
    rawText = body.text;
    return rawText;
  };

  loadRaw()
    .then(text => { $("#raw-text").textContent = text.slice(0, 60000); })
    .catch(() => { $("#raw-text").textContent = "Could not load the extracted text."; });

  $("#dl-text").addEventListener("click", async () => {
    try { download(`${slug}-text.txt`, await loadRaw()); }
    catch (err) { toast(err.message); }
  });

  const aiBtn = $("#ai-btn");
  if (aiBtn) {
    aiBtn.addEventListener("click", async () => {
      aiBtn.disabled = true;
      const original = aiBtn.textContent;
      aiBtn.innerHTML = `<span class="spinner"></span> Asking Gemini…`;
      try {
        const body = await api(`/api/jobs/${state.jobId}/ai-report`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ tone: "Professional" }),
        });
        $("#ai-out").innerHTML = `<pre class="raw">${esc(body.report)}</pre>`;
      } catch (err) {
        toast(err.message);
      } finally {
        aiBtn.disabled = false;
        aiBtn.textContent = original;
      }
    });
  }
}

/* ────────────────────────── references ─────────────────────────── */

function referenceHtml(r) {
  const num = r.number ? `<span class="ref-num">[${esc(r.number)}]</span> ` : "";
  const meta = [(r.authors || "").slice(0, 140), r.year || ""].filter(Boolean).join(", ");
  const link = r.url
    ? `<a href="${esc(r.url)}" target="_blank" rel="noopener">${r.doi ? "DOI" : "arXiv"}</a>`
    : `<a href="${esc(r.searchUrl)}" target="_blank" rel="noopener">Find</a>`;
  const count = r.cited ? `<span class="count">cited ${esc(r.cited)}×</span>` : "";
  return `<div class="ref">${num}<span class="ref-title">${esc(r.title || r.text.slice(0, 160))}</span>${count}
    <span class="ref-meta">${esc(meta)}${meta ? " · " : ""}${link}</span></div>`;
}

function referencesCsv(refs) {
  const cell = value => {
    let s = String(value ?? "");
    // A cell opening with = + - @ is run as a formula by spreadsheet apps.
    if (/^[=+\-@]/.test(s)) s = "'" + s;
    return `"${s.replace(/"/g, '""')}"`;
  };
  const rows = [["number", "authors", "title", "year", "cited_in_text", "link"]];
  refs.forEach(r => rows.push(
    [r.number ?? "", r.authors, r.title, r.year ?? "", r.cited, r.url || r.searchUrl]));
  return rows.map(row => row.map(cell).join(",")).join("\r\n");
}

function renderReferences(data) {
  const pane = $("#pane-refs");
  const refs = data.references?.items || [];
  if (!refs.length) {
    pane.innerHTML = `<p class="empty-note">No reference list was found in the
      pages that were read. For a long paper, raise <b>Pages to read</b> under
      Options -- the bibliography is usually at the end.</p>`;
    return;
  }
  const top = data.references.mostCited || [];
  pane.innerHTML = (top.length ? group("Most cited in the text",
      `<p class="ask-intro">The prior work this paper builds on, by how often it is cited.</p>
       ${top.map(referenceHtml).join("")}`) : "") +
    group(`All references (${refs.length})`, `
      <div class="ref-tools">
        <input type="search" id="ref-filter" placeholder="Filter by author, title or year"
               aria-label="Filter references" autocomplete="off">
        <button class="btn btn-ghost btn-sm" id="dl-refs">References (.csv)</button>
      </div>
      <div id="ref-list"></div>`);

  const list = $("#ref-list");
  const draw = needle => {
    const shown = refs.filter(r => !needle || r.text.toLowerCase().includes(needle));
    list.innerHTML = shown.length
      ? shown.slice(0, 250).map(referenceHtml).join("")
      : `<p class="empty-note">No reference matches that filter.</p>`;
  };
  draw("");
  $("#ref-filter").addEventListener("input", e => draw(e.target.value.trim().toLowerCase()));
  $("#dl-refs").addEventListener("click", () =>
    download(`${slugOf(data)}-references.csv`, referencesCsv(refs), "text/csv"));
}

/* ─────────────────────────── compare ───────────────────────────── */

function renderCompare() {
  const pane = $("#pane-compare");
  const others = MyJobs.all().filter(j => j.id !== state.jobId).reverse();
  if (!others.length) {
    pane.innerHTML = `
      <p class="ask-intro">Analyze a second paper, then come back here to see
        the two side by side: problem, method, data and results, plus the prior
        work both build on.</p>
      <button class="btn btn-primary btn-sm" id="cmp-another">Analyze another paper</button>`;
    $("#cmp-another").addEventListener("click", () => $("#new-paper-btn").click());
    return;
  }
  pane.innerHTML = `
    <div class="cmp-pick">
      <label class="field field-inline">
        <span class="field-label">Compare with</span>
        <select id="cmp-other">${others.map(j =>
          `<option value="${esc(j.id)}">${esc(j.title)}</option>`).join("")}</select>
      </label>
      <button class="btn btn-primary btn-sm" id="cmp-go">Compare</button>
    </div>
    <div id="cmp-out"></div>`;
  $("#cmp-go").addEventListener("click", () => runCompare($("#cmp-other").value));
}

async function runCompare(otherId) {
  const out = $("#cmp-out");
  out.innerHTML = spinnerNote("Comparing…");
  try {
    const c = await api(`/api/compare?a=${encodeURIComponent(state.jobId)}` +
                        `&b=${encodeURIComponent(otherId)}`);
    out.innerHTML = comparisonHtml(c);
  } catch (err) {
    out.innerHTML = "";
    // Finished jobs expire, and a free host restarts when it wakes up.
    if (/No such job/.test(err.message)) {
      MyJobs.remove(otherId);
      renderCompare();
      toast("That analysis has expired on the server. Analyze it again to compare.");
    } else {
      toast(err.message);
    }
  }
}

function comparisonHtml(c) {
  const sharedData = new Set(c.sharedDatasets.map(d => d.toLowerCase()));
  const sharedMetric = new Set(c.sharedMetrics.map(m => m.name.toLowerCase()));
  const list = (items, empty) => items.length
    ? items.map(t => `<div class="finding">${esc(t)}</div>`).join("")
    : `<p class="empty-note">${esc(empty)}</p>`;
  const chipRow = items => items.length
    ? `<div class="chips">${items.map(([text, shared]) =>
        `<span class="chip${shared ? " chip-shared" : ""}">${esc(text)}</span>`).join("")}</div>`
    : `<p class="empty-note">None detected.</p>`;

  const side = (s, who) => `
    <div class="cmp-col">
      <p class="cmp-who">${esc(who)}</p>
      <h3 class="cmp-title">${esc(s.title)}</h3>
      <p class="cmp-sub">${esc((s.authors || "").slice(0, 120))}</p>
      <p class="cmp-sub">${esc(s.pages)} pages · ${esc(s.words.toLocaleString())} words ·
        ${esc(s.referenceCount)} references</p>
      ${group("Problem", list(s.problem, "No explicit problem statement found."))}
      ${group("Method", list(s.method, "No explicit contribution statement found."))}
      ${group("Limitations", list(s.limitations, "No limitations stated."))}
      ${group("Data used", chipRow(s.datasets.map(d => [d, sharedData.has(d.toLowerCase())])))}
      ${group("Reported results", chipRow(Object.entries(s.metrics).slice(0, 8)
          .map(([k, v]) => [`${k} ${v}`, sharedMetric.has(k.toLowerCase())])))}
    </div>`;

  const notes = [];
  if (c.aCitesB) notes.push("This paper cites the other one.");
  if (c.bCitesA) notes.push("The other paper cites this one.");

  const common = [];
  if (c.sharedMetrics.length) {
    common.push(`<p class="ask-intro">Metrics both report -- the place to read
      their numbers against each other:</p>
      <div class="table-wrap"><table class="data">
        <thead><tr><th>Metric</th><th>This paper</th><th>Other paper</th></tr></thead>
        <tbody>${c.sharedMetrics.map(m =>
          `<tr><td>${esc(m.name)}</td><td>${esc(m.a)}</td><td>${esc(m.b)}</td></tr>`).join("")}</tbody>
      </table></div>`);
  }
  if (c.sharedDatasets.length) {
    common.push(`<p class="cmp-line"><b>Shared data:</b> ${esc(c.sharedDatasets.join(", "))}</p>`);
  }
  if (c.sharedPhrases.length) {
    common.push(`<p class="cmp-line"><b>Shared key phrases:</b> ${esc(c.sharedPhrases.join(", "))}</p>`);
  }
  if (c.sharedReferences.length) {
    common.push(`<p class="cmp-line"><b>Prior work both cite (${c.sharedReferences.length}):</b></p>
      ${list(c.sharedReferences.slice(0, 15), "")}`);
  }

  return `
    <div class="meta-row cmp-meta">
      <span class="meta-pill hot">${esc(c.relation)}</span>
      <span class="meta-pill">${esc(Math.round(c.similarity * 100))}% topic overlap</span>
    </div>
    ${notes.map(n => `<p class="cmp-note">${esc(n)}</p>`).join("")}
    <div class="cmp-grid">${side(c.a, "This paper")}${side(c.b, "Other paper")}</div>
    ${group("In common", common.length ? common.join("")
      : `<p class="empty-note">No shared data, metrics or references were found.</p>`)}`;
}

/* ───────────────────────────── tabs ────────────────────────────── */

function selectTab(name) {
  $$("#tabs button").forEach(btn =>
    btn.setAttribute("aria-selected", String(btn.dataset.pane === name)));
  ["glance", "ask", "summary", "refs", "compare", "details"].forEach(pane => {
    $(`#pane-${pane}`).hidden = pane !== name;
  });
}

/* ──────────────────────────── search ───────────────────────────── */

async function runSearch(query) {
  const out = $("#search-results");
  const btn = $("#search-btn");
  btn.disabled = true;
  const source = $("#search-source").value;
  const max = state.capabilities?.search?.maxResults ?? 50;
  const limit = Math.min(max, Math.max(1, parseInt($("#search-limit").value, 10) || 10));
  out.innerHTML = spinnerNote("Searching…");

  try {
    const data = await api(`/api/search?q=${encodeURIComponent(query)}` +
                           `&limit=${limit}&source=${encodeURIComponent(source)}`);
    // One source being down should not hide the others' results.
    const warn = Object.keys(data.failed || {}).map(name =>
      `<div class="notice notice-warn">${esc(name)} could not be reached; ` +
      `showing the other sources.</div>`).join("");
    if (!data.results.length) {
      out.innerHTML = warn + `<p class="empty-note">No results for that query.</p>`;
      return;
    }
    out.innerHTML = warn + `<p class="rc-count">${data.results.length} results</p>` +
      data.results.map((r, i) => `
      <article class="result-card">
        <h3>${esc(r.title)}</h3>
        <p class="rc-meta"><span class="src-tag">${esc(r.source)}</span>${esc(
          [r.authorLine, r.published].filter(Boolean).join(" · "))}</p>
        ${r.summary ? `<p class="rc-abs">${esc(r.summary)}</p>` : ""}
        <div class="rc-actions">
          <button class="btn btn-primary btn-sm" data-analyze="${i}">Analyze this</button>
          <a class="btn btn-ghost btn-sm" href="${esc(r.pdfUrl)}" target="_blank" rel="noopener">PDF</a>
          <a class="btn btn-ghost btn-sm" href="${esc(r.abstractUrl)}" target="_blank" rel="noopener">Source page</a>
        </div>
      </article>`).join("");

    $$("[data-analyze]", out).forEach(btn =>
      btn.addEventListener("click", () => {
        const r = data.results[Number(btn.dataset.analyze)];
        analyzeUrl(r.pdfUrl, r.title);
      }));
  } catch (err) {
    out.innerHTML = "";
    toast(err.message);
  } finally {
    btn.disabled = false;
  }
}

/* ───────────────────────────── wiring ──────────────────────────── */

function init() {
  Theme.init();
  loadCapabilities();

  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => analyzeFile(fileInput.files[0]));

  // Without preventDefault on dragover the browser just opens the file.
  ["dragenter", "dragover"].forEach(evt =>
    dropzone.addEventListener(evt, e => {
      e.preventDefault(); dropzone.classList.add("is-over");
    }));
  ["dragleave", "drop"].forEach(evt =>
    dropzone.addEventListener(evt, e => {
      e.preventDefault(); dropzone.classList.remove("is-over");
    }));
  dropzone.addEventListener("drop", e => analyzeFile(e.dataTransfer?.files?.[0]));

  // Dropping anywhere else must not navigate away from the app.
  window.addEventListener("dragover", e => e.preventDefault());
  window.addEventListener("drop", e => e.preventDefault());

  $$(".segmented button").forEach(btn =>
    btn.addEventListener("click", () => {
      $$(".segmented button").forEach(b => b.setAttribute("aria-checked", "false"));
      btn.setAttribute("aria-checked", "true");
      state.ocrMode = btn.dataset.ocr;
    }));

  $("#search-form").addEventListener("submit", e => {
    e.preventDefault();
    const q = $("#search-input").value.trim();
    if (q) runSearch(q);
  });

  $$("#tabs button").forEach(btn =>
    btn.addEventListener("click", () => selectTab(btn.dataset.pane)));

  $("#cancel-btn").addEventListener("click", () => {
    state.cancelled = true;
    clearTimeout(state.polling);
    showView("intro");
  });

  $("#new-paper-btn").addEventListener("click", () => {
    state.jobId = null; state.result = null;
    fileInput.value = "";
    showView("intro");
  });
}

document.addEventListener("DOMContentLoaded", init);
