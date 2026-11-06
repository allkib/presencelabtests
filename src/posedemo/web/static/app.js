"use strict";

const $ = (id) => document.getElementById(id);
const pretty = (s) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
const wsUrl = (path) => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;

const CAPTURE_MAX_SIDE = 640; // frames are downscaled before upload (model runs at 640)
const JPEG_QUALITY = 0.7;
const MIRROR = true;
const OBJECT_COLOR = "#f5f5f5";
const LABEL_FONT = "Inter, -apple-system, sans-serif";
const KP_NAMES = [
  "nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", "R shoulder", "L elbow",
  "R elbow", "L wrist", "R wrist", "L hip", "R hip", "L knee", "R knee", "L ankle", "R ankle",
];
const PALETTE = ["#a78bfa", "#7b8ce7", "#c084fc", "#5eead4", "#f0abfc", "#a5b4fc", "#fcd34d", "#f9a8d4"];

let config = { skeleton: [], kp_conf: 0.5, server_source: false };

// ---------- shared UI ----------

function colorFor(label) {
  if (label === "unknown") return "#737373";
  let h = 0;
  for (const c of label) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

function setStatus(text, kind = "") {
  $("status").textContent = text;
  $("status").className = `pill ${kind}`;
}

function setError(msg) {
  $("error").hidden = !msg;
  $("error").textContent = msg || "";
}

const fpsMeter = {
  times: [],
  tick() {
    const now = performance.now();
    this.times.push(now);
    while (this.times.length && now - this.times[0] > 1000) this.times.shift();
    $("fps").textContent = `${this.times.length} FPS`;
  },
  reset() {
    this.times = [];
    $("fps").textContent = "– FPS";
  },
};

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function debugHtml(p) {
  const rows = Object.entries(p.scores || {})
    .sort((a, b) => b[1] - a[1])
    .map(([name, score]) => {
      const best = name === p.pose ? " best" : "";
      return `<span class="${best}">${pretty(name)}</span>
        <div class="mini${best}"><span style="width:${Math.round(score * 100)}%"></span></div>
        <span class="${best}">${score.toFixed(2)}</span>`;
    })
    .join("");
  const seen = (p.keypoints || [])
    .map((k, i) => (k[2] >= config.kp_conf ? KP_NAMES[i] : null))
    .filter(Boolean);
  return `<div class="debug">${rows}
    <span class="kps">Visible joints (${seen.length}/17): ${seen.join(", ") || "none"}</span></div>`;
}

function renderPeople(people) {
  const debug = $("debug").checked;
  $("people").innerHTML = people.length
    ? people
        .map(
          (p) => `<li>
            <div class="person-head">
              <span class="pose-name" style="color:${colorFor(p.pose)}">${pretty(p.pose)}</span>
              <span>#${p.id ?? "?"}</span>
            </div>
            <div class="bar"><span style="width:${Math.round(p.score * 100)}%"></span></div>
            ${p.holding?.length ? `<div class="holding">Holding ${p.holding.map(escapeHtml).join(", ")}</div>` : ""}
            ${debug ? debugHtml(p) : ""}
          </li>`
        )
        .join("")
    : '<li class="empty">Nobody detected</li>';
  const active = new Set(people.map((p) => p.pose));
  document.querySelectorAll("#poses li").forEach((li) => {
    li.classList.toggle("active", active.has(li.dataset.pose));
  });
}

function renderObjects(objects = []) {
  $("objects").innerHTML = objects.length
    ? objects
        .map((o) => `<li><span class="label">${escapeHtml(o.label)}</span><span>${o.score.toFixed(2)}</span></li>`)
        .join("")
    : '<li class="empty">No objects detected</li>';
}

function renderVersion() {
  if (!config.version) return;
  const started = new Date(config.started_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  $("version").textContent = `v${config.version} · started ${started}`;
}

// ---------- browser camera mode ----------

const browser = {
  stream: null,
  ws: null,
  running: false,
  capture: document.createElement("canvas"),

  async start() {
    setError("");
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setError("Camera access needs HTTPS (or http://localhost). Open this page over HTTPS.");
      return;
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
    } catch (e) {
      const reasons = {
        NotAllowedError: "Camera permission was denied. Allow it in your browser's site settings and try again.",
        NotFoundError: "No camera was found on this device.",
        NotReadableError: "The camera is in use by another app.",
      };
      setError(reasons[e.name] || `Could not start camera: ${e.message}`);
      return;
    }
    const video = $("video");
    video.srcObject = this.stream;
    await video.play();
    $("browserStage").classList.toggle("mirror", MIRROR);
    $("startPanel").hidden = true;
    $("stopBtn").hidden = false;
    this.running = true;
    this.connect();
  },

  stop() {
    this.running = false;
    this.ws?.close();
    this.ws = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    $("video").srcObject = null;
    $("startPanel").hidden = false;
    $("stopBtn").hidden = true;
    clearOverlay();
    renderPeople([]);
    renderObjects([]);
    fpsMeter.reset();
    setStatus("camera off");
  },

  connect() {
    setStatus("connecting…");
    const ws = new WebSocket(wsUrl("/ws/infer"));
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => {
      setStatus("live", "ok");
      setError("");
      this.sendFrame();
    };
    ws.onmessage = (e) => this.onMessage(JSON.parse(e.data));
    ws.onclose = (e) => {
      if (!this.running || this.ws !== ws) return;
      setStatus("reconnecting…", "bad");
      if (e.code === 1013) setError(e.reason || "Server busy, retrying…");
      setTimeout(() => this.running && this.connect(), 1500);
    };
  },

  onMessage(msg) {
    if (msg.type === "result") {
      setStatus("live", "ok");
      setError("");
      drawOverlay(msg);
      renderPeople(msg.people);
      renderObjects(msg.objects);
      fpsMeter.tick();
      this.sendFrame(); // next frame only after a reply → no backlog
    } else if (msg.type === "status") {
      if (msg.status === "error") setError(`Model failed to load: ${msg.error}`);
      setStatus(msg.status === "loading" ? "loading model…" : msg.status, msg.status === "error" ? "bad" : "");
      setTimeout(() => this.sendFrame(), 500);
    } else if (msg.type === "error") {
      setError(msg.error);
      setTimeout(() => this.sendFrame(), 500);
    }
  },

  sendFrame() {
    const video = $("video");
    if (!this.running || this.ws?.readyState !== WebSocket.OPEN) return;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw) {
      setTimeout(() => this.sendFrame(), 100);
      return;
    }
    const scale = Math.min(1, CAPTURE_MAX_SIDE / Math.max(vw, vh));
    const cap = this.capture;
    cap.width = Math.round(vw * scale);
    cap.height = Math.round(vh * scale);
    cap.getContext("2d").drawImage(video, 0, 0, cap.width, cap.height);
    cap.toBlob(
      (blob) => {
        if (blob && this.ws?.readyState === WebSocket.OPEN) this.ws.send(blob);
      },
      "image/jpeg",
      JPEG_QUALITY
    );
  },
};

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
  ctx.fill();
}

function clearOverlay() {
  const c = $("overlay");
  c.getContext("2d").clearRect(0, 0, c.width, c.height);
}

function drawOverlay(msg) {
  const video = $("video");
  const canvas = $("overlay");
  const W = video.videoWidth;
  const H = video.videoHeight;
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const s = W / msg.frame_size[0];
  const X = (x) => (MIRROR ? W - x * s : x * s);
  const Y = (y) => y * s;
  const lw = Math.max(2, W / 320);
  const fontPx = Math.round(Math.max(14, W / 50));

  for (const o of msg.objects || []) {
    const [x1, y1, x2, y2] = o.bbox;
    const left = Math.min(X(x1), X(x2));
    const width = Math.abs(X(x2) - X(x1));
    ctx.strokeStyle = OBJECT_COLOR;
    ctx.lineWidth = lw;
    ctx.strokeRect(left, Y(y1), width, Y(y2) - Y(y1));
    const text = `${o.label} ${o.score.toFixed(2)}`;
    ctx.font = `500 ${fontPx}px ${LABEL_FONT}`;
    const tw = ctx.measureText(text).width;
    const top = Math.min(H, Y(y2) + fontPx + 10);
    ctx.fillStyle = OBJECT_COLOR;
    roundRect(ctx, left, top - fontPx - 10, tw + 12, fontPx + 10, 6);
    ctx.fillStyle = "#141414";
    ctx.fillText(text, left + 6, top - 8);
  }

  for (const p of msg.people) {
    const color = colorFor(p.pose);
    const kp = p.keypoints;
    const vis = (i) => kp[i][2] >= config.kp_conf;

    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.lineCap = "round";
    for (const [a, b] of config.skeleton) {
      if (!vis(a) || !vis(b)) continue;
      ctx.beginPath();
      ctx.moveTo(X(kp[a][0]), Y(kp[a][1]));
      ctx.lineTo(X(kp[b][0]), Y(kp[b][1]));
      ctx.stroke();
    }
    for (let i = 0; i < kp.length; i++) {
      if (!vis(i)) continue;
      ctx.beginPath();
      ctx.arc(X(kp[i][0]), Y(kp[i][1]), lw + 2, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.lineWidth = lw;
    }

    const [x1, y1, x2] = p.bbox;
    const left = Math.max(0, Math.min(X(x1), X(x2)));
    const top = Math.max(28, Y(y1));
    const holding = p.holding?.length ? ` + ${p.holding.join(", ")}` : "";
    const text = `#${p.id ?? "?"} ${pretty(p.pose)} ${p.score.toFixed(2)}${holding}`;
    ctx.font = `500 ${fontPx}px ${LABEL_FONT}`;
    const tw = ctx.measureText(text).width;
    const th = fontPx + 10;
    ctx.fillStyle = color;
    roundRect(ctx, left, top - th, tw + 12, th, 6);
    ctx.fillStyle = "#141414";
    ctx.fillText(text, left + 6, top - 8);
  }
}

// ---------- server camera mode ----------

const server = {
  ws: null,
  active: false,

  start() {
    this.active = true;
    $("stream").src = `/stream?t=${Date.now()}`;
    this.connect();
  },

  stop() {
    this.active = false;
    $("stream").removeAttribute("src");
    this.ws?.close();
    this.ws = null;
    fpsMeter.reset();
  },

  connect() {
    const ws = new WebSocket(wsUrl("/ws"));
    this.ws = ws;
    ws.onmessage = (e) => {
      const st = JSON.parse(e.data);
      const bad = st.status === "error";
      setStatus((st.status || "unknown").replace(/_/g, " "), bad ? "bad" : st.status === "running" ? "ok" : "");
      $("fps").textContent = `${st.fps ?? "–"} FPS`;
      setError(st.error || "");
      renderPeople(st.people || []);
      renderObjects(st.objects || []);
    };
    ws.onclose = () => {
      if (this.active && this.ws === ws) setTimeout(() => this.active && this.connect(), 1000);
    };
  },
};

$("stream").onerror = () => {
  if (server.active) setTimeout(() => server.active && ($("stream").src = `/stream?t=${Date.now()}`), 1000);
};

// ---------- mode switching ----------

function setMode(mode) {
  if (mode === "server" && !config.server_source) mode = "browser";
  localStorage.setItem("posedemo.mode", mode);
  document.querySelectorAll(".modes button").forEach((b) => {
    b.setAttribute("aria-selected", String(b.dataset.mode === mode));
  });
  setError("");
  renderPeople([]);
  renderObjects([]);
  if (mode === "browser") {
    server.stop();
    $("stream").hidden = true;
    $("browserStage").hidden = false;
    setStatus(browser.running ? "live" : "camera off", browser.running ? "ok" : "");
  } else {
    browser.stop();
    $("browserStage").hidden = true;
    $("stream").hidden = false;
    server.start();
  }
}

async function init() {
  const [cfg, poses] = await Promise.all([
    fetch("/api/config").then((r) => r.json()),
    fetch("/api/poses").then((r) => r.json()),
  ]);
  config = cfg;
  renderVersion();
  $("debug").checked = localStorage.getItem("posedemo.debug") === "1";
  $("debug").onchange = () => localStorage.setItem("posedemo.debug", $("debug").checked ? "1" : "0");
  $("poses").innerHTML = poses
    .map((p) => `<li data-pose="${p.name}"><b>${pretty(p.name)}</b>${p.description}</li>`)
    .join("");
  $("modeServer").hidden = !config.server_source;
  document.querySelectorAll(".modes button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  $("startBtn").onclick = () => browser.start();
  $("stopBtn").onclick = () => browser.stop();
  setMode(localStorage.getItem("posedemo.mode") || "browser");
}

init();
