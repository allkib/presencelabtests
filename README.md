# posedemo

Real-time webcam pose recognition with [Ultralytics YOLO pose](https://docs.ultralytics.com/tasks/pose/),
served to the browser with FastAPI. Draws a skeleton for each person, tracks people across
frames, and names the pose each person is doing: T-pose, hands up, left/right hand up,
arms crossed, squat, or standing.

## Quick start

```bash
make setup          # creates .venv (Python 3.11) and installs the package + dev tools
make run            # → open http://127.0.0.1:8000
```

The first run downloads `yolo11n-pose.pt` (~6 MB) into `models/`.

> **macOS:** the terminal app you launch from needs Camera permission
> (System Settings → Privacy & Security → Camera). Without it, the page shows a clear error.

### Options

```bash
make run ARGS="--model yolo11s-pose.pt"   # bigger / more accurate model
make run ARGS="--source 1"                # another camera
make run ARGS="--source clip.mp4"         # video file (loops), image, or RTSP/HTTP URL
make run ARGS="--device cpu --no-mirror --port 9000"
```

You can also set any option in `.env` (see `.env.example`, prefix `POSEDEMO_`).

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | Web UI |
| `GET /stream` | MJPEG stream of annotated frames |
| `GET /snapshot.jpg` | Latest annotated frame |
| `GET /api/state` | Latest detections as JSON (`people[].pose`, `score`, `scores`, `bbox`, `id`) |
| `WS /ws` | Pushes the state JSON on every processed frame |
| `GET /api/poses` | Registered poses |
| `GET /health` | Pipeline status |

## Architecture

```
FrameSource ─▶ PoseDetector ─▶ PoseClassifier ─▶ LabelSmoother ─▶ annotate ─▶ ResultBus ─▶ FastAPI (/stream, /ws, /api)
 (sources/)     (vision/)        (poses/)          (poses/)      (pipeline/)   (pipeline/)    (web/)
```

- **One inference thread** (`pipeline/worker.py`) publishes to a `ResultBus`. Any number of browser
  clients read from the bus, so adding viewers doesn't add inference cost.
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

## Development

```bash
make test     # pytest (no camera or model needed)
make lint     # ruff check + format check
make format
```
