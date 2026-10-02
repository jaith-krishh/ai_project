// Acoustic Sound Analyzer front end: upload a file, call /api/analyze, render the report.

const $ = (id) => document.getElementById(id);
const fileInput = $("file-input");
const dropzone = $("dropzone");
const analyzeBtn = $("analyze-btn");
const statusEl = $("status");
const errorEl = $("error");
const player = $("player");

let selectedFile = null;
let current = null;          // { result, audioUrl }
const history = [];          // previous analyses this session

// ---------- theme ----------

const themeBtn = $("theme-toggle");
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

function effectiveTheme() {
  return document.documentElement.dataset.theme || (systemDark.matches ? "dark" : "light");
}

function updateThemeButton() {
  themeBtn.textContent = effectiveTheme() === "dark" ? "Light mode" : "Dark mode";
}

themeBtn.addEventListener("click", () => {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) {}
  updateThemeButton();
});
systemDark.addEventListener("change", updateThemeButton);
updateThemeButton();

// ---------- helpers ----------

function fmtTime(sec, mode = "round") {
  const r = mode === "floor" ? Math.floor : mode === "ceil" ? (x) => Math.ceil(x - 1e-9) : Math.round;
  const s = Math.max(0, r(sec));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "style") node.setAttribute("style", v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

function categoryOf(label, result) {
  const ev = result.events.find((e) => e.label === label);
  return ev ? ev.category : "other";
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = !msg;
}

// ---------- file selection ----------

function selectFile(file) {
  if (!file) return;
  selectedFile = file;
  $("drop-text").textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
  analyzeBtn.disabled = false;
  showError("");
}

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((t) =>
  dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
["dragleave", "drop"].forEach((t) =>
  dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));
dropzone.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));

// ---------- analysis ----------

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  const file = selectedFile;
  analyzeBtn.disabled = true;
  showError("");
  statusEl.textContent = "Analyzing… this can take a while for long recordings.";
  try {
    const res = await fetch(`/api/analyze?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    const data = await res.json().catch(() => ({ error: `Server error (${res.status})` }));
    if (!res.ok || data.error) throw new Error(data.error || `Server error (${res.status})`);
    const entry = { result: data, audioUrl: URL.createObjectURL(file) };
    history.unshift(entry);
    render(entry);
    renderHistory();
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = "";
    showError(err.message);
  } finally {
    analyzeBtn.disabled = false;
  }
});

// ---------- rendering ----------

function render(entry) {
  current = entry;
  const r = entry.result;
  const agg = r.aggregation;
  const unknownCount = r.events.filter((e) => e.category === "unknown").length;

  $("results").hidden = false;
  $("res-file").textContent = r.filename;
  const note = $("res-note");
  note.hidden = r.unknown_detection;
  note.textContent = "Unknown-sound detection is off: outputs/centroids.pt or outputs/threshold.pt is missing.";

  $("stat-location").textContent = agg.location;
  $("stat-duration").textContent = fmtTime(r.duration);
  $("stat-events").textContent = r.events.length;
  $("stat-unknown").textContent = unknownCount;

  const reportLines = r.report.split("\n");
  const locLine = reportLines.find((l) => l.startsWith("2. Location classification:")) || "";
  const reason = locLine.match(/\((.*)\)\s*$/);
  $("location-reason").textContent = reason ? `Based on: ${reason[1]}` : "";

  player.src = entry.audioUrl;

  renderTimeline(r);
  renderEvents(r);
  renderBars($("label-bars"),
    Object.entries(agg.label_percentages)
      .sort((a, b) => b[1] - a[1])
      .map(([lbl, pct]) => [agg.label_display[lbl] || lbl, pct, categoryOf(lbl, r)]));
  renderBars($("category-bars"),
    Object.entries(agg.category_percentages)
      .map(([cat, pct]) => [cat[0].toUpperCase() + cat.slice(1), pct, cat === "other" ? "other" : cat]));
  $("report").textContent = r.report;

  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function seek(sec) {
  player.currentTime = sec;
  player.play().catch(() => {});
}

function renderTimeline(r) {
  const tl = $("timeline");
  tl.replaceChildren();
  const dur = r.duration || 1;
  const rows = new Map();
  for (const ev of r.events) {
    if (!rows.has(ev.display)) rows.set(ev.display, []);
    rows.get(ev.display).push(ev);
  }
  if (rows.size === 0) {
    tl.append(el("p", { class: "muted" }, "No sounds detected."));
  }
  for (const [name, evs] of rows) {
    const track = el("div", { class: "tl-track" });
    for (const ev of evs) {
      const left = (ev.start / dur) * 100;
      const width = Math.max(((ev.end - ev.start) / dur) * 100, 0.6);
      track.append(el("div", {
        class: `tl-seg cat-${ev.category}`,
        style: `left:${left}%;width:${width}%`,
        title: `${name}  ${fmtTime(ev.start, "floor")}–${fmtTime(ev.end, "ceil")}`,
        onclick: () => seek(ev.start),
      }));
    }
    tl.append(el("div", { class: "tl-row" }, el("span", { class: "tl-label", title: name }, name), track));
  }
  const axis = $("timeline-axis");
  axis.replaceChildren(...[0, 0.25, 0.5, 0.75, 1].map((f) => el("span", {}, fmtTime(dur * f))));
}

function renderEvents(r) {
  const body = $("events-body");
  body.replaceChildren();
  if (r.events.length === 0) {
    body.append(el("tr", {}, el("td", { colspan: "3", class: "muted" }, "No sounds detected.")));
    return;
  }
  for (const ev of r.events) {
    let details;
    if (ev.category === "unknown") {
      const matches = (ev.similar_to || [])
        .map((m) => `${Math.round(m.similarity * 100)}% ${m.display.toLowerCase()}`).join(", ");
      details = el("span", {}, "Closest matches: ", matches || "none");
    } else {
      details = el("span", {}, `Confidence ${ev.confidence.toFixed(2)}`);
    }
    const name = el("span", { class: ev.category === "unknown" ? "tag-unknown" : "" }, ev.display);
    body.append(el("tr", { onclick: () => seek(ev.start), title: "Play from here" },
      el("td", {}, el("span", { class: `dot cat-${ev.category}` }), name),
      el("td", {}, `${fmtTime(ev.start, "floor")}–${fmtTime(ev.end, "ceil")}`),
      el("td", {}, details)));
  }
}

function renderBars(container, rows) {
  container.replaceChildren();
  if (rows.length === 0) {
    container.append(el("p", { class: "muted" }, "Nothing to show."));
    return;
  }
  for (const [name, pct, cat] of rows) {
    container.append(el("div", { class: "bar" },
      el("div", { class: "bar-top" }, el("span", {}, name), el("span", { class: "muted" }, `${pct.toFixed(1)}%`)),
      el("div", { class: "bar-track" },
        el("div", { class: `bar-fill cat-${cat}`, style: `width:${Math.min(pct, 100)}%` }))));
  }
}

function renderHistory() {
  $("history-card").hidden = history.length < 2;
  const list = $("history");
  list.replaceChildren(...history.map((entry) => el("li", { onclick: () => render(entry) },
    el("span", {}, entry.result.filename),
    el("span", { class: "muted small" }, `${entry.result.aggregation.location} · ${fmtTime(entry.result.duration)}`))));
}

// ---------- export ----------

$("copy-btn").addEventListener("click", async () => {
  if (!current) return;
  try {
    await navigator.clipboard.writeText(current.result.report);
    statusEl.textContent = "Report copied.";
  } catch {
    statusEl.textContent = "Copy failed; select the text report below instead.";
  }
  setTimeout(() => { statusEl.textContent = ""; }, 2000);
});

$("json-btn").addEventListener("click", () => {
  if (!current) return;
  const { report, ...data } = current.result;
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: current.result.filename.replace(/\.[^.]+$/, "") + "_analysis.json" });
  document.body.append(a);
  a.click();
  a.remove();
});
