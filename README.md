# posedemo

Real-time webcam pose recognition with [Ultralytics YOLO pose](https://docs.ultralytics.com/tasks/pose/),
served to the browser with FastAPI. Draws a skeleton for each person, tracks people across
frames, and names the pose each person is doing: T-pose, hands up, left/right hand up,
arms crossed, squat, standing, or neutral. It also detects everyday **objects** (80 COCO classes:
cell phone, cup, bottle, book, laptop, …) and says which person is holding them.

There are two camera modes, which share one loaded model:

- **Browser camera** (default): the visitor's browser captures their webcam and uploads frames
  over a WebSocket. The server returns keypoints and pose names, and the browser draws them over the
  live video. This mode works when the app is hosted (Docker/Dokploy).
- **Server camera**: the server opens a camera attached to its own machine, or a video file or
  stream URL, and streams annotated frames as MJPEG. Useful locally, and for fixed cameras.

> YOLO pose only outputs 17 body keypoints per person. The pose *names* come from the rules in
> `src/posedemo/poses/rules.py`.

## Quick start

```bash
make setup          # creates .venv (Python 3.11) and installs the package + dev tools
make run            # → open http://127.0.0.1:8000
make dev            # same, but restarts automatically when you edit the code
```

The first run downloads `yolo11n-pose.pt` (~6 MB) into `models/`.

Click **Start camera** and allow access when the browser asks. Browsers only allow camera access
on `https://` pages or `http://localhost`.

> **Server camera mode on macOS:** the terminal app you launch from needs Camera permission
> (System Settings → Privacy & Security → Camera). Use `--browser-only` to skip the server camera.

### Options

```bash
make run ARGS="--model yolo11s-pose.pt"   # bigger / more accurate model
make run ARGS="--source 1"                # another camera
make run ARGS="--source clip.mp4"         # video file (loops), image, or RTSP/HTTP URL
make run ARGS="--device cpu --no-mirror --port 9000"
make run ARGS="--browser-only"            # no server-side camera
```

You can also set any option in `.env` (see `.env.example`, prefix `POSEDEMO_`).

## Poses and objects

- **Works at a desk.** Hand-raise poses only need your shoulders in view. A raised arm is
  recognised from the wrist, or from the elbow when the wrist is hidden (in front of your face or
  above the frame). An arm that's out of frame counts as "not raised".
- **Neutral** is a fallback for when you're in view but not doing a specific pose. **Unknown**
  means the model can't see enough of you, for example no shoulders.
- **Debug toggle** (People panel): shows every pose's score and which joints are visible, so you
  can see why a pose did or didn't match.
- **Objects** come from a second YOLO model (`yolo11n.pt`) and are drawn in amber. An object is
  "held" when one of a person's hands is inside or near its box. Use `--no-objects` (or
  `POSEDEMO_ENABLE_OBJECTS=false`) to turn this off, and `POSEDEMO_OBJECT_EVERY_N=2` to run it on
  every other frame to save CPU.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | Web UI |
| `WS /ws/infer` | Browser-camera mode: send JPEG frames (binary), receive `{people: [{pose, score, keypoints, bbox, id}]}` |
| `GET /api/config` | Available modes, skeleton edges |
| `GET /stream` | Server camera: MJPEG stream of annotated frames |
| `GET /snapshot.jpg` | Server camera: latest annotated frame |
| `GET /api/state` | Server camera: latest detections as JSON |
| `WS /ws` | Server camera: pushes the state JSON on every processed frame |
| `GET /api/poses` | Registered poses |
| `GET /health` | 200 when the model is loaded (and the server camera is healthy, if enabled), else 503 |

## Architecture

```
Server camera:  FrameSource ─▶ ┐                                                       ┌─▶ annotate ─▶ ResultBus ─▶ /stream, /ws
                               ├─▶ PoseDetector ─▶ IouTracker ─▶ PoseClassifier ─▶ LabelSmoother
Browser camera: /ws/infer  ──▶ ┘   (one shared model)  (per stream)                    └─▶ JSON reply ─▶ browser canvas
```

- **One model, many streams:** the YOLO model is loaded once and shared. Tracking and smoothing
  state is kept per stream (`vision/tracking.py`, `web/sessions.py`), so users never get each
  other's IDs or labels.
- **Server camera** runs in one inference thread (`pipeline/worker.py`) that publishes to a
  `ResultBus`. Any number of viewers read from the bus, so adding viewers doesn't add inference cost.
- **Browser camera** uses back-pressure: the browser sends its next frame only after the previous
  reply arrives, so a slow server lowers the frame rate instead of building up a queue.
  `POSEDEMO_MAX_BROWSER_SESSIONS` caps concurrent users.
- **Swappable parts:** `create_app(settings, source=..., detector=...)` accepts any `FrameSource` or
  `PoseDetector`, which is how the tests run without a camera or a model.
- **Pose rules** use joint angles and positions measured relative to torso length, so they work at
  any distance from the camera. Left/right are the person's own sides; the view is mirrored only
  for display.

### Adding a pose

Add a class to `src/posedemo/poses/rules.py` (or any imported module):

```python
@register
class Star(PoseRule):
    name = "star"
    description = "Arms up and out, legs apart"
    required = (KP.LEFT_WRIST, KP.RIGHT_WRIST, KP.LEFT_ANKLE, KP.RIGHT_ANKLE, ...)

    def score(self, p: Person) -> float:
        ...  # return 0..1
```

The new pose then appears automatically in the classifier, `/api/poses`, and the UI. Add a synthetic
skeleton case to `tests/test_rules.py` to test it.

## Branches and releases

| Branch | Purpose | Deploys? |
|---|---|---|
| `dev` (default) | Day-to-day work. Push here freely. | No |
| `main` | Production | Yes. Dokploy deploys every change. |

```bash
git switch dev            # work and push here; CI tests every push
make release              # opens a PR dev → main with the commit list
```

`main` is protected. It only accepts pull requests, and the `test` and `docker` CI checks must
pass before you can merge. Merging the release PR is what updates the live server.

## Deploying with Docker / Dokploy

The `Dockerfile` builds a CPU-only, browser-camera image. The model weights are included in the
image, the container runs as a non-root user, and `/health` drives the Docker health check.

```bash
docker compose up --build        # local test → http://localhost:8000
```

**Dokploy:**

1. Create an **Application** and point it at this GitHub repo with branch **`main`** (production).
2. Set **Build Type** to **Dockerfile** (path `Dockerfile`).
3. Under **Domains**, add your domain with **container port `8000`** and turn on **HTTPS**
   (Let's Encrypt). HTTPS is required, because browsers block camera access on plain-HTTP sites.
4. Deploy. WebSockets work through Dokploy's Traefik proxy without extra configuration.

Optional environment variables (Dokploy → Environment):

| Variable | Default | Notes |
|---|---|---|
| `POSEDEMO_MAX_BROWSER_SESSIONS` | `8` | Concurrent camera users; extra users get "server busy" |
| `POSEDEMO_IMGSZ` | `640` | `480` or `416` is faster on small CPUs |
| `POSEDEMO_CONF` | `0.5` | Person detection threshold |
| `POSEDEMO_ENABLE_OBJECTS` | `true` | Object detection. Roughly halves frames/s on CPU. |
| `POSEDEMO_OBJECT_EVERY_N` | `1` | `2`–`3` runs objects less often (faster on small servers) |

**Server requirements:** at least 2 GB RAM (PyTorch plus the model use about 1 GB at runtime) and
about 5 GB of free disk for Dokploy to build the image. The image is built for the server's own
CPU architecture, so x86 and ARM VPSes both work.

**After deploying**, check it end to end from any machine that has the dev install:

```bash
.venv/bin/python scripts/smoke_test.py https://your-domain.example
```

**CI:** every push to `main` runs `.github/workflows/ci.yml`. It runs lint and tests, then builds
this same image on GitHub, starts it, waits for the health check, and runs the smoke test inside
the container. A green check means the image Dokploy builds will start and serve poses.

Performance: one shared model on a CPU VPS handles roughly 5–15 frames/s in total, split across
active users. For a GPU host, build with
`--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124` and set `POSEDEMO_DEVICE=cuda`.

## Development

```bash
make test     # pytest (no camera or model needed)
make lint     # ruff check + format check
make format
```
