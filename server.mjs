import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 3000);
const EPISODES_DIR = path.join(__dirname, "episodes");
const ROADMAP_PATH = path.join(__dirname, "VIRAL_SHORTS_ROADMAP.md");

function safeReadJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function safeReadText(filePath) {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch {
    return "";
  }
}

function listEpisodes() {
  if (!fs.existsSync(EPISODES_DIR)) return [];
  const entries = fs.readdirSync(EPISODES_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .sort();

  const episodes = [];
  for (const slug of entries) {
    const epDir = path.join(EPISODES_DIR, slug);
    const timeline = safeReadJson(path.join(epDir, "config", "timeline.json"));
    const srcDir = path.join(epDir, "src");
    const audioDir = path.join(epDir, "audio");
    const srcFiles = fs.existsSync(srcDir)
      ? fs.readdirSync(srcDir).filter((f) => f.endsWith(".py")).sort()
      : [];
    const audioFiles = fs.existsSync(audioDir)
      ? fs.readdirSync(audioDir).filter((f) => f.endsWith(".py")).sort()
      : [];

    const hasMasterWav = fs.existsSync(path.join(epDir, "audio", "output", "master_audio.wav"));
    const hasSubtitles = fs.existsSync(path.join(epDir, "subtitles", "subtitles.ass"));
    const hasModal = fs.existsSync(path.join(epDir, "render_on_modal.py"));
    const hasVerifyFraming = fs.existsSync(path.join(srcDir, "verify_framing.py"));
    const hasLabelsQa = fs.existsSync(path.join(srcDir, "labels_3d.py"));

    const narrationLines = timeline?.narration_lines || timeline?.dialogue || [];
    const wordCount = narrationLines.reduce(
      (sum, ln) => sum + String(ln.text || "").trim().split(/\s+/).filter(Boolean).length,
      0
    );

    episodes.push({
      slug,
      title: timeline?.title || slug.replace(/-/g, " "),
      duration: timeline?.duration ?? 30.0,
      fps: timeline?.fps ?? 60,
      totalFrames: timeline?.totalFrames ?? 1800,
      resolution: timeline?.resolution || [1080, 1920],
      beats: timeline?.beats || [],
      narrationLines,
      sfxCues: timeline?.sfx_cues || [],
      wordCount,
      srcFiles,
      audioFiles,
      readiness: {
        masterWav: hasMasterWav,
        subtitles: hasSubtitles,
        modalScript: hasModal,
        verifyFraming: hasVerifyFraming,
        labelsQa: hasLabelsQa,
      },
      renderCommand: `gh workflow run render-episode-4k.yml -f episode=${slug} -f workers=10 -f samples=16 -f upscale_4k=true`,
      audioCommand: `python3 episodes/${slug}/audio/generate_audio.py --rebuild-all`,
    });
  }
  return episodes;
}

function parseRoadmap() {
  const md = safeReadText(ROADMAP_PATH);
  const rows = [];
  for (const line of md.split(/\r?\n/)) {
    const m = line.match(/^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$/);
    if (m) {
      const rawStatus = m[4].trim();
      let state = "Planned";
      if (rawStatus.includes("4K") || rawStatus.includes("✅")) state = "Shipped 4K";
      else if (rawStatus.includes("IN PROGRESS") || rawStatus.includes("🟡")) state = "In Progress";
      rows.push({
        index: Number(m[1]),
        slug: m[2].trim(),
        title: m[3].trim(),
        statusText: rawStatus.replace(/[✅🟡⬜]/g, "").trim(),
        state,
      });
    }
  }
  return rows;
}

function runPythonScript(scriptPath) {
  return new Promise((resolve) => {
    execFile("python3", [scriptPath], { cwd: __dirname, timeout: 15000 }, (err, stdout, stderr) => {
      resolve({
        ok: !err,
        stdout: (stdout || "").trim(),
        stderr: (stderr || "").trim(),
      });
    });
  });
}

function sendJson(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(body);
}

const INDEX_HTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>3D Science Shorts Studio — Blender 5.3.0 &amp; Modal 4K Pipeline</title>
  <meta name="description" content="Production console for 3D Science &amp; Action YouTube Shorts (60 fps, True 4K, Blender 5.3.0 Cycles OPTIX)." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #0b0f17;
      --surface: #111827;
      --surface-alt: #161f31;
      --border: rgba(148, 163, 184, 0.16);
      --border-strong: rgba(148, 163, 184, 0.28);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-ink: #082f49;
      --green: #22c55e;
      --amber: #f59e0b;
      --red: #ef4444;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 14px;
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
    }
    .mono { font-family: "JetBrains Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }
    .tabular { font-variant-numeric: tabular-nums; }

    .layout {
      display: grid;
      grid-template-columns: 268px minmax(0, 1fr);
      min-height: 100vh;
    }

    .sidebar {
      background: var(--surface);
      border-right: 1px solid var(--border);
      padding: 24px 16px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .brand {
      padding: 0 8px;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: var(--text);
      white-space: nowrap;
    }
    .nav-group-label {
      padding: 0 8px 8px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
    }
    .nav-list {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .nav-btn {
      width: 100%;
      text-align: left;
      background: transparent;
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 10px 12px;
      color: var(--text-muted);
      font: inherit;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      transition: background 150ms ease, color 150ms ease, border-color 150ms ease;
    }
    .nav-btn:hover {
      background: rgba(148, 163, 184, 0.08);
      color: var(--text);
    }
    .nav-btn.active {
      background: var(--surface-alt);
      border-color: var(--border-strong);
      color: var(--text);
      font-weight: 600;
    }
    .nav-btn .ep-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .nav-btn .ep-fps {
      font-size: 12px;
      color: var(--text-muted);
      flex-shrink: 0;
    }

    .main-col {
      display: flex;
      flex-direction: column;
      min-width: 0;
    }

    .topbar {
      height: 64px;
      padding: 0 32px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 32px;
      background: var(--bg);
    }
    .breadcrumb {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      color: var(--text-muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .breadcrumb strong {
      color: var(--text);
      font-weight: 600;
    }
    .top-actions {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-shrink: 0;
    }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      height: 38px;
      padding: 0 16px;
      border-radius: 8px;
      border: 1px solid var(--border-strong);
      background: var(--surface);
      color: var(--text);
      font: inherit;
      font-size: 13px;
      font-weight: 600;
      line-height: 1;
      cursor: pointer;
      white-space: nowrap;
      flex-shrink: 0;
      transition: background 150ms ease, border-color 150ms ease, transform 150ms ease;
    }
    .btn:hover { background: var(--surface-alt); border-color: rgba(148, 163, 184, 0.45); }
    .btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    .btn:disabled { opacity: 0.55; cursor: not-allowed; }
    .btn-primary {
      background: var(--accent);
      border-color: var(--accent);
      color: var(--accent-ink);
    }
    .btn-primary:hover { background: #7dd3fc; border-color: #7dd3fc; }

    .content {
      padding: 32px;
      max-width: 1220px;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 32px;
    }

    .hero-head h1 {
      margin: 0 0 8px;
      font-size: clamp(1.5rem, 2.2vw, 2.1rem);
      line-height: 1.2;
      font-weight: 700;
      letter-spacing: -0.02em;
      text-wrap: balance;
    }
    .meta-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      color: var(--text-muted);
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--surface);
      overflow: hidden;
    }
    .metric-cell {
      padding: 18px 20px;
      border-right: 1px solid var(--border);
    }
    .metric-cell:last-child { border-right: none; }
    .metric-label {
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .metric-val {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.01em;
    }
    .metric-sub {
      margin-top: 4px;
      font-size: 12px;
      color: var(--text-muted);
    }

    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }
    .section-head h2 {
      margin: 0;
      font-size: 16px;
      font-weight: 600;
    }

    .tabs {
      display: inline-flex;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 4px;
      gap: 4px;
    }
    .tab-btn {
      height: 32px;
      padding: 0 14px;
      border: none;
      border-radius: 6px;
      background: transparent;
      color: var(--text-muted);
      font: inherit;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
      transition: background 150ms ease, color 150ms ease;
    }
    .tab-btn.active {
      background: var(--surface-alt);
      color: var(--text);
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }
    tr:last-child td { border-bottom: none; }
    th {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      background: rgba(15, 23, 42, 0.45);
    }
    td.num, th.num { text-align: right; white-space: nowrap; }
    tbody tr:hover { background: rgba(148, 163, 184, 0.04); }

    .status-ok { color: var(--green); font-weight: 600; }
    .status-warn { color: var(--amber); font-weight: 600; }
    .status-err { color: var(--red); font-weight: 600; }

    .cmd-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
    }
    .cmd-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .cmd-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .cmd-title { font-weight: 600; font-size: 14px; }
    pre.code {
      margin: 0;
      padding: 12px 14px;
      border-radius: 8px;
      background: var(--bg);
      border: 1px solid var(--border);
      font-size: 12.5px;
      line-height: 1.5;
      color: #cbd5e1;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .qa-out {
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .search-input {
      height: 36px;
      width: 260px;
      padding: 0 12px;
      border-radius: 8px;
      border: 1px solid var(--border-strong);
      background: var(--surface);
      color: var(--text);
      font: inherit;
      font-size: 13px;
    }
    .search-input:focus { outline: 2px solid var(--accent); outline-offset: 1px; }

    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { border-right: none; border-bottom: 1px solid var(--border); }
      .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric-cell { border-bottom: 1px solid var(--border); }
      .cmd-grid { grid-template-columns: 1fr; }
      .topbar { padding: 0 16px; }
      .content { padding: 20px 16px; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">Science Shorts Studio</div>
      <div>
        <div class="nav-group-label">Episodes</div>
        <div id="episode-nav" class="nav-list"></div>
      </div>
      <div>
        <div class="nav-group-label">Planning</div>
        <div class="nav-list">
          <button id="nav-roadmap" class="nav-btn" type="button">
            <span class="ep-name">30-Episode Roadmap</span>
            <span id="roadmap-count" class="ep-fps mono">30</span>
          </button>
        </div>
      </div>
    </aside>

    <div class="main-col">
      <header class="topbar">
        <div id="breadcrumb" class="breadcrumb">Studio / <strong>Loading...</strong></div>
        <div class="top-actions">
          <button id="btn-run-qa" class="btn btn-primary" type="button">Run Framing &amp; Label QA</button>
        </div>
      </header>

      <main id="main-content" class="content">
        <div>Loading studio workspace...</div>
      </main>
    </div>
  </div>

  <script>
    const state = {
      episodes: [],
      roadmap: [],
      view: "episode",
      selectedSlug: "bulletproof-vest-myth",
      subTab: "beats",
      roadmapFilter: "all",
      roadmapQuery: "",
      qaBySlug: {},
      qaLoading: false,
      copiedKey: "",
    };

    function esc(str) {
      return String(str ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    async function loadStudio() {
      const [epRes, rmRes] = await Promise.all([
        fetch("/api/episodes").then((r) => r.json()),
        fetch("/api/roadmap").then((r) => r.json()),
      ]);
      state.episodes = epRes.episodes || [];
      state.roadmap = rmRes.roadmap || [];
      if (!state.episodes.some((e) => e.slug === state.selectedSlug) && state.episodes.length) {
        state.selectedSlug = state.episodes[0].slug;
      }
      document.getElementById("roadmap-count").textContent = String(state.roadmap.length);
      render();
    }

    async function runQaForCurrent() {
      const ep = state.episodes.find((e) => e.slug === state.selectedSlug);
      if (!ep || state.qaLoading) return;
      state.qaLoading = true;
      render();
      try {
        const res = await fetch("/api/qa?episode=" + encodeURIComponent(ep.slug));
        state.qaBySlug[ep.slug] = await res.json();
      } catch (err) {
        state.qaBySlug[ep.slug] = { error: String(err) };
      } finally {
        state.qaLoading = false;
        render();
      }
    }

    async function copyCommand(key, text) {
      try {
        await navigator.clipboard.writeText(text);
        state.copiedKey = key;
        render();
        setTimeout(() => {
          if (state.copiedKey === key) {
            state.copiedKey = "";
            render();
          }
        }, 1600);
      } catch {
        // Fallback ignored cleanly
      }
    }

    function renderSidebar() {
      const nav = document.getElementById("episode-nav");
      nav.innerHTML = state.episodes.map((ep) => {
        const active = state.view === "episode" && state.selectedSlug === ep.slug ? "active" : "";
        return '<button class="nav-btn ' + active + '" type="button" data-slug="' + esc(ep.slug) + '">' +
          '<span class="ep-name">' + esc(ep.slug) + '</span>' +
          '<span class="ep-fps mono">' + esc(ep.fps) + ' fps</span>' +
          '</button>';
      }).join("");

      nav.querySelectorAll("button[data-slug]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.view = "episode";
          state.selectedSlug = btn.dataset.slug;
          render();
        });
      });

      const rmBtn = document.getElementById("nav-roadmap");
      rmBtn.classList.toggle("active", state.view === "roadmap");
    }

    function renderEpisodeView() {
      const ep = state.episodes.find((e) => e.slug === state.selectedSlug);
      const main = document.getElementById("main-content");
      const bc = document.getElementById("breadcrumb");
      const qaBtn = document.getElementById("btn-run-qa");

      if (!ep) {
        bc.innerHTML = "Studio / <strong>No episodes found</strong>";
        qaBtn.style.display = "none";
        main.innerHTML = "<div class='panel qa-out'>No episode folders detected in episodes/.</div>";
        return;
      }

      bc.innerHTML = "Studio / Episodes / <strong>" + esc(ep.slug) + "</strong>";
      qaBtn.style.display = ep.readiness.verifyFraming ? "inline-flex" : "none";
      qaBtn.disabled = state.qaLoading;
      qaBtn.textContent = state.qaLoading ? "Running QA..." : "Run Framing & Label QA";

      const qa = state.qaBySlug[ep.slug];
      const qaHtml = qa ? renderQaPanel(qa) : "";

      let tabTableHtml = "";
      if (state.subTab === "beats") {
        tabTableHtml = '<div class="panel"><table><thead><tr>' +
          '<th>Beat</th><th>Scene Action</th><th>3D Callout</th><th class="num">Frames</th><th class="num">Time</th>' +
          '</tr></thead><tbody>' +
          (ep.beats.length ? ep.beats.map((b) =>
            '<tr>' +
              '<td class="mono">' + esc(b.id) + '</td>' +
              '<td>' + esc(b.name || b.description || "") + '</td>' +
              '<td class="mono">' + esc(b.callout || "—") + '</td>' +
              '<td class="num mono">' + esc(b.startFrame ?? "—") + '..' + esc(b.endFrame ?? "—") + '</td>' +
              '<td class="num mono">' + Number(b.start ?? 0).toFixed(2) + 's – ' + Number(b.end ?? 0).toFixed(2) + 's</td>' +
            '</tr>'
          ).join("") : '<tr><td colspan="5">No beat entries in timeline.json.</td></tr>') +
          '</tbody></table></div>';
      } else if (state.subTab === "script") {
        tabTableHtml = '<div class="panel"><table><thead><tr>' +
          '<th>Line</th><th>Beat</th><th>Narration Copy</th><th class="num">Window</th>' +
          '</tr></thead><tbody>' +
          (ep.narrationLines.length ? ep.narrationLines.map((ln) =>
            '<tr>' +
              '<td class="mono">' + esc(ln.id) + '</td>' +
              '<td class="mono">' + esc(ln.beat || "—") + '</td>' +
              '<td>' + esc(ln.text) + '</td>' +
              '<td class="num mono">' + Number(ln.t0 ?? ln.start ?? 0).toFixed(2) + 's – ' + Number(ln.t1 ?? ln.end ?? 0).toFixed(2) + 's</td>' +
            '</tr>'
          ).join("") : '<tr><td colspan="4">No narration lines in timeline.json.</td></tr>') +
          '</tbody></table></div>';
      } else {
        tabTableHtml = '<div class="panel"><table><thead><tr>' +
          '<th>Event Key</th><th>Beat</th><th>SFX Cue Type</th><th class="num">Frame</th><th class="num">Time</th>' +
          '</tr></thead><tbody>' +
          (ep.sfxCues.length ? ep.sfxCues.map((c) =>
            '<tr>' +
              '<td class="mono">' + esc(c.ev || c.id || "—") + '</td>' +
              '<td class="mono">' + esc(c.beat || "—") + '</td>' +
              '<td>' + esc(c.type || c.desc || "—") + '</td>' +
              '<td class="num mono">' + esc(c.frame ?? "—") + '</td>' +
              '<td class="num mono">' + Number(c.t ?? 0).toFixed(2) + 's</td>' +
            '</tr>'
          ).join("") : '<tr><td colspan="5">No SFX cues listed in timeline.json.</td></tr>') +
          '</tbody></table></div>';
      }

      const audioStatus = ep.readiness.masterWav
        ? '<span class="status-ok">Master WAV ready</span>'
        : '<span class="status-warn">Auto-builds on Modal or via CLI</span>';

      main.innerHTML =
        '<section class="hero-head">' +
          '<h1>' + esc(ep.title) + '</h1>' +
          '<div class="meta-row">' +
            '<span class="mono">episodes/' + esc(ep.slug) + '</span>' +
            '<span>·</span>' +
            '<span class="tabular">' + esc(ep.resolution[0]) + '×' + esc(ep.resolution[1]) + ' → 2160×3840 True 4K</span>' +
            '<span>·</span>' +
            '<span class="tabular">' + esc(ep.fps) + ' fps (' + esc(ep.totalFrames) + ' frames)</span>' +
            '<span>·</span>' +
            audioStatus +
          '</div>' +
        '</section>' +

        '<section class="metrics-grid">' +
          '<div class="metric-cell"><div class="metric-label">Duration</div><div class="metric-val mono">' + Number(ep.duration).toFixed(1) + 's</div><div class="metric-sub tabular">' + esc(ep.beats.length) + ' production beats</div></div>' +
          '<div class="metric-cell"><div class="metric-label">Narration Script</div><div class="metric-val mono">' + esc(ep.wordCount) + ' words</div><div class="metric-sub">' + esc(ep.narrationLines.length) + ' timed lines</div></div>' +
          '<div class="metric-cell"><div class="metric-label">Scene Modules</div><div class="metric-val mono">' + esc(ep.srcFiles.length) + ' Python</div><div class="metric-sub">' + esc(ep.audioFiles.length) + ' audio scripts</div></div>' +
          '<div class="metric-cell"><div class="metric-label">Subtitles &amp; Modal</div><div class="metric-val">' + (ep.readiness.subtitles ? "ASS Ready" : "Pending") + '</div><div class="metric-sub">' + (ep.readiness.modalScript ? "Modal 10-GPU configured" : "No Modal script") + '</div></div>' +
        '</section>' +

        qaHtml +

        '<section>' +
          '<div class="section-head">' +
            '<h2>Production Breakdown</h2>' +
            '<div class="tabs" role="tablist">' +
              '<button class="tab-btn ' + (state.subTab === "beats" ? "active" : "") + '" type="button" data-tab="beats">6 Beats</button>' +
              '<button class="tab-btn ' + (state.subTab === "script" ? "active" : "") + '" type="button" data-tab="script">Narration (' + esc(ep.wordCount) + 'w)</button>' +
              '<button class="tab-btn ' + (state.subTab === "sfx" ? "active" : "") + '" type="button" data-tab="sfx">SFX Cues (' + esc(ep.sfxCues.length) + ')</button>' +
            '</div>' +
          '</div>' +
          tabTableHtml +
        '</section>' +

        '<section>' +
          '<div class="section-head"><h2>Render &amp; Audio Commands</h2></div>' +
          '<div class="cmd-grid">' +
            '<div class="cmd-box">' +
              '<div class="cmd-top"><span class="cmd-title">Trigger 10-GPU Modal 4K Cloud Render</span>' +
              '<button class="btn" type="button" id="copy-render">' + (state.copiedKey === "render" ? "Copied" : "Copy") + '</button></div>' +
              '<pre class="code mono">' + esc(ep.renderCommand) + '</pre>' +
            '</div>' +
            '<div class="cmd-box">' +
              '<div class="cmd-top"><span class="cmd-title">Rebuild Edge-TTS Voice, SFX, BGM &amp; Subtitles</span>' +
              '<button class="btn" type="button" id="copy-audio">' + (state.copiedKey === "audio" ? "Copied" : "Copy") + '</button></div>' +
              '<pre class="code mono">' + esc(ep.audioCommand) + '</pre>' +
            '</div>' +
          '</div>' +
        '</section>';

      main.querySelectorAll("button[data-tab]").forEach((b) => {
        b.addEventListener("click", () => {
          state.subTab = b.dataset.tab;
          render();
        });
      });
      document.getElementById("copy-render")?.addEventListener("click", () => copyCommand("render", ep.renderCommand));
      document.getElementById("copy-audio")?.addEventListener("click", () => copyCommand("audio", ep.audioCommand));
    }

    function renderQaPanel(qa) {
      if (qa.error) {
        return '<section class="panel qa-out"><div class="status-err">QA failed: ' + esc(qa.error) + '</div></section>';
      }
      const fOk = qa.framing?.ok;
      const lOk = qa.labels?.ok;
      const fSummary = (qa.framing?.stdout || qa.framing?.stderr || "").split(/\\r?\\n/).slice(-4).join("\\n");
      const lSummary = (qa.labels?.stdout || qa.labels?.stderr || "").split(/\\r?\\n/).slice(0, 10).join("\\n");
      return '<section class="panel qa-out">' +
        '<div class="meta-row">' +
          '<span class="' + (fOk ? "status-ok" : "status-err") + '">Framing QA: ' + (fOk ? "100% Passed" : "Failed") + '</span>' +
          (qa.labels ? '<span>·</span><span class="' + (lOk ? "status-ok" : "status-err") + '">3D Labels NDC/DOF QA: ' + (lOk ? "0 Errors" : "Failed") + '</span>' : '') +
        '</div>' +
        '<pre class="code mono">' + esc(fSummary) + (lSummary ? "\\n\\n" + esc(lSummary) : "") + '</pre>' +
      '</section>';
    }

    function renderRoadmapView() {
      const main = document.getElementById("main-content");
      const bc = document.getElementById("breadcrumb");
      document.getElementById("btn-run-qa").style.display = "none";
      bc.innerHTML = "Studio / <strong>30-Episode Viral Shorts Roadmap</strong>";

      const q = state.roadmapQuery.toLowerCase();
      const filtered = state.roadmap.filter((r) => {
        if (state.roadmapFilter !== "all" && r.state !== state.roadmapFilter) return false;
        if (q && !(r.title.toLowerCase().includes(q) || r.slug.toLowerCase().includes(q))) return false;
        return true;
      });

      const shipped = state.roadmap.filter((r) => r.state === "Shipped 4K").length;
      const active = state.roadmap.filter((r) => r.state === "In Progress").length;

      main.innerHTML =
        '<section class="hero-head">' +
          '<h1>Viral Shorts Production Roadmap</h1>' +
          '<div class="meta-row">' +
            '<span class="tabular">' + esc(shipped) + ' shipped in 4K</span>' +
            '<span>·</span>' +
            '<span class="tabular">' + esc(active) + ' in progress</span>' +
            '<span>·</span>' +
            '<span class="tabular">' + esc(state.roadmap.length - shipped - active) + ' queued</span>' +
          '</div>' +
        '</section>' +
        '<section>' +
          '<div class="section-head">' +
            '<div class="tabs">' +
              '<button class="tab-btn ' + (state.roadmapFilter === "all" ? "active" : "") + '" type="button" data-rf="all">All (' + state.roadmap.length + ')</button>' +
              '<button class="tab-btn ' + (state.roadmapFilter === "Shipped 4K" ? "active" : "") + '" type="button" data-rf="Shipped 4K">Shipped 4K (' + shipped + ')</button>' +
              '<button class="tab-btn ' + (state.roadmapFilter === "In Progress" ? "active" : "") + '" type="button" data-rf="In Progress">In Progress (' + active + ')</button>' +
              '<button class="tab-btn ' + (state.roadmapFilter === "Planned" ? "active" : "") + '" type="button" data-rf="Planned">Planned</button>' +
            '</div>' +
            '<input id="rm-search" class="search-input" type="search" placeholder="Filter topics or folder..." value="' + esc(state.roadmapQuery) + '" />' +
          '</div>' +
          '<div class="panel"><table><thead><tr>' +
            '<th class="num">#</th><th>Folder</th><th>Episode Title</th><th>Status</th>' +
          '</tr></thead><tbody>' +
          (filtered.length ? filtered.map((r) => {
            const cls = r.state === "Shipped 4K" ? "status-ok" : (r.state === "In Progress" ? "status-warn" : "");
            return '<tr>' +
              '<td class="num mono">' + esc(r.index) + '</td>' +
              '<td class="mono">' + esc(r.slug) + '</td>' +
              '<td>' + esc(r.title) + '</td>' +
              '<td class="' + cls + '">' + esc(r.statusText || r.state) + '</td>' +
            '</tr>';
          }).join("") : '<tr><td colspan="4">No matching topics found.</td></tr>') +
          '</tbody></table></div>' +
        '</section>';

      main.querySelectorAll("button[data-rf]").forEach((b) => {
        b.addEventListener("click", () => {
          state.roadmapFilter = b.dataset.rf;
          render();
        });
      });
      const inp = document.getElementById("rm-search");
      inp?.addEventListener("input", (e) => {
        state.roadmapQuery = e.target.value;
        render();
        const nextInp = document.getElementById("rm-search");
        if (nextInp) {
          nextInp.focus();
          nextInp.setSelectionRange(nextInp.value.length, nextInp.value.length);
        }
      });
    }

    function render() {
      renderSidebar();
      if (state.view === "roadmap") renderRoadmapView();
      else renderEpisodeView();
    }

    document.getElementById("nav-roadmap").addEventListener("click", () => {
      state.view = "roadmap";
      render();
    });
    document.getElementById("btn-run-qa").addEventListener("click", runQaForCurrent);

    loadStudio();
  </script>
</body>
</html>`;

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
    if (url.pathname === "/api/episodes") {
      return sendJson(res, 200, { episodes: listEpisodes() });
    }
    if (url.pathname === "/api/roadmap") {
      return sendJson(res, 200, { roadmap: parseRoadmap() });
    }
    if (url.pathname === "/api/qa") {
      const slug = (url.searchParams.get("episode") || "").replace(/[^a-z0-9-]/gi, "");
      const epDir = path.join(EPISODES_DIR, slug);
      if (!slug || !fs.existsSync(epDir)) {
        return sendJson(res, 404, { error: "Episode not found" });
      }
      const framingPath = path.join(epDir, "src", "verify_framing.py");
      const labelsPath = path.join(epDir, "src", "labels_3d.py");
      const framing = fs.existsSync(framingPath) ? await runPythonScript(framingPath) : null;
      const labels = fs.existsSync(labelsPath) ? await runPythonScript(labelsPath) : null;
      return sendJson(res, 200, { episode: slug, framing, labels });
    }
    if (url.pathname === "/" || url.pathname === "/index.html") {
      res.writeHead(200, {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "no-store",
      });
      res.end(INDEX_HTML);
      return;
    }
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Not found");
  } catch (err) {
    sendJson(res, 500, { error: String(err) });
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Science Shorts Studio dev server running at http://0.0.0.0:${PORT}`);
});
