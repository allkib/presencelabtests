const $ = (id) => document.getElementById(id);
const pretty = (s) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

async function loadPoses() {
  const poses = await (await fetch("/api/poses")).json();
  $("poses").innerHTML = poses
    .map((p) => `<li data-pose="${p.name}"><b>${pretty(p.name)}</b> — ${p.description}</li>`)
    .join("");
}

function render(state) {
  const status = $("status");
  status.textContent = (state.status || "unknown").replace(/_/g, " ");
  status.className = `pill ${state.status || ""}`;
  $("fps").textContent = `${state.fps ?? "–"} FPS`;

  const err = $("error");
  err.hidden = !state.error;
  err.textContent = state.error || "";

  const people = state.people || [];
  $("people").innerHTML = people.length
    ? people
        .map(
          (p) => `<li>
            <div class="person-head">
              <span class="pose-name">${pretty(p.pose)}</span>
              <span>#${p.id ?? "?"}</span>
            </div>
            <div class="bar"><span style="width:${Math.round(p.score * 100)}%"></span></div>
          </li>`
        )
        .join("")
    : '<li class="empty">Nobody detected</li>';

  const active = new Set(people.map((p) => p.pose));
  document.querySelectorAll("#poses li").forEach((li) => {
    li.classList.toggle("active", active.has(li.dataset.pose));
  });
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (e) => render(JSON.parse(e.data));
  ws.onclose = () => {
    $("status").textContent = "reconnecting…";
    $("status").className = "pill";
    setTimeout(connect, 1000);
  };
}

// Reload the MJPEG stream if it drops (e.g. server restart).
$("stream").onerror = () => setTimeout(() => ($("stream").src = `/stream?t=${Date.now()}`), 1000);

loadPoses();
connect();
