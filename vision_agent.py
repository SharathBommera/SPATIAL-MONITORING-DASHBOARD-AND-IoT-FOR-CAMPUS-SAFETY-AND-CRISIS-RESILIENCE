# =============================================================================
# FILE: vision_agent.py
# PURPOSE: CrowdPulse Perception Engine
#          Vision + Audio + IoT Fusion → Digital Twin State Updates
#
# UPGRADE vs. original:
#   - Replaced HSV blob fire detector with RT-DETR + AdaptiveFusionEngine
#     (imported from Multi Model/fusion_system/)
#   - Added zone-aware person detection (each person tagged to their zone)
#   - Added per-zone flow vectors
#   - Richer payload pushed to the Digital Twin Hub (crowdpulse_server.py)
#   - Reads latest IoT telemetry back from the hub to feed the fusion engine
# =============================================================================

import sys
import os
import faulthandler

# Enable faulthandler IMMEDIATELY — prints Python traceback on segfault/access-violation
faulthandler.enable()

# ── CRITICAL: Force single-threaded PyTorch BEFORE importing torch ───────────
# Without this, MKL/OpenMP spawns worker threads that use SIMD instructions
# (AVX-512 etc.) incompatible with this CPU → 0xc000001d ILLEGAL_INSTRUCTION
os.environ['OMP_NUM_THREADS']    = '1'
os.environ['MKL_NUM_THREADS']    = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# ── CRITICAL: Restrict CPU instruction set to AVX2 ─────────────────────────
# The i5-1335U (Raptor Lake) does NOT support AVX-512.  PyTorch's prebuilt
# MKL/ATen kernels may attempt AVX-512 instructions → 0xc000001d ILLEGAL
# INSTRUCTION → state corruption → access violation in _conv_forward.
# Forcing AVX2 prevents the crash entirely.
os.environ['ATEN_CPU_CAPABILITY']   = 'avx2'
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX2'
os.environ['MKL_THREADING_LAYER']    = 'SEQUENTIAL'

# ── Import path for the Multi Model fusion system ────────────────────────────
_FUSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'Multi Model', 'fusion_system')
sys.path.insert(0, _FUSION_DIR)

# ── Standard imports ─────────────────────────────────────────────────────────
from flask import Flask, Response
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
# ── CRITICAL: Disable Ultralytics telemetry ─────────────────────────────────
# Ultralytics spawns a hidden daemon thread (ultralytics.utils.events._post)
# that makes HTTPS requests via urllib.  On Python 3.13 Windows, this socket
# call triggers a fatal access-violation when running concurrently with
# PyTorch inference in the main thread.  Disabling sync stops the thread.
try:
    from ultralytics.utils import SETTINGS as _ul_settings
    _ul_settings['sync'] = False
except Exception:
    pass

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)          # pure inference — no autograd overhead
import time
import threading
from collections import deque
import queue
import http.client
import json as _json
from urllib.parse import urlparse
import subprocess
import tempfile

# ── Serial (ESP32 USB) ───────────────────────────────────────────────────────
SERIAL_AVAILABLE = False
_serial_port_name = None
try:
    import serial as _pyserial
    import serial.tools.list_ports as _serial_ports
    SERIAL_AVAILABLE = True
except ImportError:
    print("⚠️  pyserial not installed — run: pip install pyserial")

# ── Audio ─────────────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False
    print("⚠️  sounddevice not available — audio sensor disabled")

# ── Multi Model fusion imports ────────────────────────────────────────────────
try:
    # Temporarily change working dir so config.py relative paths resolve
    _orig_cwd = os.getcwd()
    os.chdir(_FUSION_DIR)
    from vision_wrapper import VisionWrapper
    from fusion_engine import AdaptiveFusionEngine, FusionInput, AlarmState
    os.chdir(_orig_cwd)
    FUSION_AVAILABLE = True
    print("✅ Multi Model fusion system imported")
except Exception as e:
    os.chdir(_orig_cwd) if '_orig_cwd' in dir() else None
    FUSION_AVAILABLE = False
    print(f"⚠️  Fusion system not available ({e}) — falling back to HSV detector")

# =============================================================================
# CONFIG
# =============================================================================
HUB_THING_ID  = os.getenv("CROUDPULSE_TWIN_ID", "org.campus:seminar_hall_01")
HUB_URL       = f"http://localhost:5000/api/2/things/{HUB_THING_ID}"
HUB_STATE_URL = "http://localhost:5000/api/twin/state"
CAMERA_ID     = 0
AI_INTERVAL   = 0.15
ESP32_IP      = os.getenv("CROUDPULSE_ESP32_IP", "10.243.83.181")
VISION_PORT   = 5010              # Port 5001 is often reserved by Windows Hyper-V/services
RTDETR_SUBPROCESS_MODE = (os.name == 'nt')
RTDETR_TIMEOUT_SECONDS = 20.0
RTDETR_MAX_FAILURES    = 5
CAMERA_STALE_SECONDS   = 2.5
CAMERA_REOPEN_DELAY    = 1.5
IOT_STALE_SECONDS      = 12.0
ESP32_STATUS_INTERVAL  = 10.0
FIRE_INFERENCE_INTERVAL = 0.5

RTDETR_MODEL_CANDIDATES = [
    os.getenv("CROUDPULSE_FIRE_MODEL_PATH"),
    os.getenv("CROUDPULSE_RTDETR_MODEL"),
    r"D:\Final Year Project\Multi Model\Output_Comparision\DF2_RT-DETR\runs_D_Fire_RT-DETR-L\train\weights\best.pt",
    os.path.abspath(os.path.join(_FUSION_DIR, '..', 'best.pt')),
]

# Zone definitions (normalised 0-1 coordinates, matching venue schema)
ZONES = {
    "entry_lobby":  {"x": 0.00, "y": 0.00, "w": 0.25, "h": 0.30},
    "stage_area":   {"x": 0.25, "y": 0.00, "w": 0.50, "h": 0.30},
    "main_seating": {"x": 0.00, "y": 0.30, "w": 0.70, "h": 0.70},
    "exit_corridor":{"x": 0.70, "y": 0.00, "w": 0.30, "h": 1.00},
}

# =============================================================================
# FLASK APP (video stream server)
# =============================================================================
app = Flask(__name__)
CORS(app)

# =============================================================================
# GLOBALS
# =============================================================================
global_frame      = np.zeros((480, 640, 3), dtype=np.uint8)
video_frame       = np.zeros((480, 640, 3), dtype=np.uint8)
global_detections = []
global_lock       = threading.Lock()
running           = True

# Audio
audio_volume      = 0.0
audio_stress_score = 0

# Latest IoT reading from ESP32 / hub. Start empty so we do not overwrite
# genuine hardware readings in the twin with placeholder defaults.
latest_iot = {
    "temperature": None,
    "humidity": None,
    "gas_level": None,
    "source": None,
    "last_update": 0.0,
}

hardware_state = {
    "esp32_online": False,
    "esp32_endpoint": None,
    "last_contact_ts": None,
    "data_source": "unavailable",
    "wifi_rssi": None,
    "sensor_ok": None,
    "alert_mode": "UNKNOWN",
    "light_active": None,
}

camera_state = {
    "online": False,
    "backend": None,
    "status": "Starting",
    "last_frame_ts": 0.0,
    "reconnects": 0,
}
_camera_backend_cursor = 0

# Isolated RT-DETR worker state. On Windows/CPU we keep RT-DETR out of the main
# process so a native illegal-instruction crash cannot kill the entire app.
_rtdetr_model_path = None
_rtdetr_calib_path = None
_rtdetr_worker = None
_rtdetr_worker_out = queue.Queue(maxsize=8)
_rtdetr_worker_failures = 0
_rtdetr_pending = None
_rtdetr_last_result = None
_rtdetr_launching = False
_esp32_base_urls = []
_active_esp32_base_url = None


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_base_url(raw):
    if not raw:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    parsed = urlparse(raw)
    if not parsed.hostname:
        return None
    base = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        base += f":{parsed.port}"
    return base


def _build_esp32_base_urls():
    return _dedupe_keep_order([
        _normalize_base_url(os.getenv("CROUDPULSE_ESP32_URL")),
        _normalize_base_url(os.getenv("CROUDPULSE_ESP32_HOST")),
        _normalize_base_url(ESP32_IP),
        _normalize_base_url("esp32-crowdpulse.local"),
        _normalize_base_url("esp32.local"),
    ])


def _select_rtdetr_model_path():
    for candidate in RTDETR_MODEL_CANDIDATES:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def _coerce_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if value is None:
        return None
    return bool(value)


def _make_status_frame(headline, detail=None, accent=(0, 165, 255)):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (18, 22, 30)
    cv2.putText(frame, "CrowdPulse Vision Agent", (28, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 240, 248), 2, cv2.LINE_AA)
    cv2.putText(frame, headline, (28, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, accent, 2, cv2.LINE_AA)
    if detail:
        cv2.putText(frame, detail, (28, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 208, 220), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (24, 245), (616, 260), (40, 48, 62), -1)
    cv2.rectangle(frame, (24, 245), (280, 260), accent, -1)
    return frame


def _render_annotated_frame(base_frame, detections):
    output = base_frame.copy()
    h, w = output.shape[:2]
    for z_id, b in ZONES.items():
        x1 = int(b["x"] * w)
        y1 = int(b["y"] * h)
        x2 = int((b["x"] + b["w"]) * w)
        y2 = int((b["y"] + b["h"]) * h)
        colors = {
            "entry_lobby":  (160, 32, 240),
            "stage_area":   (255, 200, 0),
            "main_seating": (0, 200, 100),
            "exit_corridor":(0, 100, 255)
        }
        c = colors.get(z_id, (200, 200, 200))
        cv2.rectangle(output, (x1, y1), (x2, y2), c, 1)
        cv2.putText(output, z_id.upper().replace("_", " "),
                    (x1 + 4, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

    for (x1, y1, x2, y2, label, color) in detections:
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        # Label background for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1 - 8, th + 4)
        cv2.rectangle(output, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 4), color, -1)
        cv2.putText(output, label, (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if camera_state.get("status"):
        cv2.putText(output, camera_state["status"], (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return output


def _set_global_frame(frame):
    global global_frame
    with global_lock:
        global_frame = frame.copy()


def _set_video_frame(frame):
    global video_frame
    with global_lock:
        video_frame = frame.copy()


def _set_camera_status(online, status, backend=None):
    camera_state["online"] = bool(online)
    camera_state["status"] = status
    if backend is not None:
        camera_state["backend"] = backend
    if online:
        camera_state["last_frame_ts"] = time.time()


def _camera_is_live():
    return (
        camera_state.get("online", False) and
        (time.time() - camera_state.get("last_frame_ts", 0.0)) < CAMERA_STALE_SECONDS
    )


def _update_latest_iot(data, source):
    if not isinstance(data, dict):
        return False
    temp = _coerce_float(data.get("temperature"))
    humidity = _coerce_float(data.get("humidity"))
    gas_level = _coerce_float(data.get("gas_level"))
    if temp is None and humidity is None and gas_level is None:
        return False
    if temp is not None:
        latest_iot["temperature"] = temp
    if humidity is not None:
        latest_iot["humidity"] = humidity
    if gas_level is not None:
        latest_iot["gas_level"] = gas_level
    latest_iot["source"] = source
    latest_iot["last_update"] = time.time()
    hardware_state["data_source"] = source
    if source.startswith("esp32"):
        hardware_state["esp32_online"] = True
        hardware_state["last_contact_ts"] = latest_iot["last_update"]
    return True


def _has_fresh_iot_data():
    return (
        latest_iot["temperature"] is not None and
        latest_iot["humidity"] is not None and
        latest_iot["gas_level"] is not None and
        (time.time() - latest_iot["last_update"]) < IOT_STALE_SECONDS
    )


def _get_iot_values():
    return (
        latest_iot["temperature"] if latest_iot["temperature"] is not None else 25.0,
        latest_iot["humidity"] if latest_iot["humidity"] is not None else 60.0,
        latest_iot["gas_level"] if latest_iot["gas_level"] is not None else 200.0,
    )


def _merge_hardware_status(data, base_url, source):
    if not isinstance(data, dict):
        return
    hardware_state["esp32_online"] = True
    hardware_state["esp32_endpoint"] = base_url
    hardware_state["last_contact_ts"] = time.time()
    hardware_state["data_source"] = source
    if "wifi_rssi" in data:
        hardware_state["wifi_rssi"] = _coerce_float(data.get("wifi_rssi"))
    if "sensor_ok" in data:
        hardware_state["sensor_ok"] = _coerce_bool(data.get("sensor_ok"))
    if "alert_mode" in data and data.get("alert_mode") is not None:
        hardware_state["alert_mode"] = str(data.get("alert_mode"))
    if "light_active" in data:
        hardware_state["light_active"] = _coerce_bool(data.get("light_active"))


def _mark_esp32_failure(base_url):
    hardware_state["esp32_online"] = False
    if base_url:
        hardware_state["esp32_endpoint"] = base_url

# =============================================================================
# INIT: Models
# =============================================================================
print("[INIT] Loading person detection model (YOLOv8n)...")
yolo_model = YOLO('yolov8n.pt')

# Vision fire detector (RT-DETR from Multi Model)
vision_wrapper   = None
fusion_engine    = None

if FUSION_AVAILABLE:
    try:
        model_path = _select_rtdetr_model_path()
        if not model_path:
            raise FileNotFoundError("No RT-DETR weights file found in configured search paths")
        calib_path = os.path.join(_FUSION_DIR, 'calibration_artifacts',
                                  'vision_platt_calibrator.pkl')
        print(f"[Vision] Using RT-DETR weights: {model_path}")
        _rtdetr_model_path = model_path
        _rtdetr_calib_path = calib_path
        if not RTDETR_SUBPROCESS_MODE:
            vision_wrapper = VisionWrapper(model_path=model_path,
                                           calibrator_path=calib_path)
            vision_wrapper.load_model()
            vision_wrapper.load_calibrator()
        fusion_engine = AdaptiveFusionEngine()
        if RTDETR_SUBPROCESS_MODE:
            print("✅ RT-DETR fire model configured for isolated worker mode")
        else:
            print("✅ RT-DETR fire model loaded")
    except Exception as e:
        print(f"⚠️  RT-DETR load failed ({e}) — using HSV fallback")
        vision_wrapper = None
        fusion_engine  = None

_esp32_base_urls = _build_esp32_base_urls()
_active_esp32_base_url = _esp32_base_urls[0] if _esp32_base_urls else None
if _esp32_base_urls:
    print(f"[INIT] ESP32 endpoints to try: {', '.join(_esp32_base_urls)}")
else:
    print("[INIT] No ESP32 endpoint configured; hardware polling disabled")

# =============================================================================
# MODEL WARM-UP — called in the MAIN thread before any workers start.
# ALL PyTorch inference runs in the main thread (ai_loop) to avoid native
# access-violation crashes in daemon threads on Python 3.13 Windows.
# =============================================================================
def _warmup_models():
    """Run a single inference on a blank frame — triggers JIT/MKL init."""
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    print("[WARMUP] YOLOv8n...", end=" ", flush=True)
    try:
        with torch.inference_mode():
            yolo_model(dummy, verbose=False)
        print("OK", flush=True)
    except Exception as e:
        print(f"WARN: {e}", flush=True)

    if vision_wrapper is not None:
        print("[WARMUP] RT-DETR...", end=" ", flush=True)
        try:
            with torch.inference_mode():
                vision_wrapper.predict(dummy)
            print("OK", flush=True)
        except Exception as e:
            print(f"WARN: {e}", flush=True)
    elif RTDETR_SUBPROCESS_MODE and _rtdetr_model_path and _rtdetr_calib_path:
        print("[WARMUP] RT-DETR... deferred (isolated worker)", flush=True)

cap = None   # Lazy init — opened inside camera_thread() to avoid COM conflicts


def _rtdetr_stdout_reader(proc):
    """Forward worker stdout lines into a queue for request/response handling."""
    try:
        while proc.stdout:
            line = proc.stdout.readline()
            if not line:
                break
            try:
                _rtdetr_worker_out.put(line.strip(), timeout=0.5)
            except queue.Full:
                pass
    except Exception:
        pass


def _rtdetr_stderr_reader(proc):
    """Surface worker stderr lines for diagnostics without blocking the worker."""
    try:
        while proc.stderr:
            line = proc.stderr.readline()
            if not line:
                break
            line = line.strip()
            if line:
                print(f"[RTDETR Worker] {line}", flush=True)
    except Exception:
        pass


def _drain_rtdetr_queue():
    """Clear any stale worker responses."""
    try:
        while True:
            _rtdetr_worker_out.get_nowait()
    except queue.Empty:
        return


def _delete_temp_file(path):
    if not path:
        return
    try:
        os.remove(path)
    except Exception:
        pass


def _shutdown_rtdetr_worker():
    """Terminate the isolated RT-DETR worker process if it is running."""
    global _rtdetr_worker, _rtdetr_pending
    proc = _rtdetr_worker
    _rtdetr_worker = None
    if _rtdetr_pending is not None:
        _delete_temp_file(_rtdetr_pending.get("temp_path"))
        _rtdetr_pending = None
    if proc is None:
        return
    # Send quit command through stdin
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.write("__quit__\n")
            proc.stdin.flush()
    except (OSError, BrokenPipeError, ValueError):
        pass
    # Close the stdin/stdout/stderr pipes to avoid OSError on interpreter exit
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if pipe and not pipe.closed:
                pipe.close()
        except (OSError, ValueError):
            pass
    # Terminate the process
    try:
        proc.terminate()
    except (OSError, PermissionError):
        pass
    try:
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except (OSError, PermissionError):
            pass


def _start_rtdetr_worker():
    """Start a dedicated RT-DETR worker subprocess and wait for readiness."""
    global _rtdetr_worker, _rtdetr_launching
    if _rtdetr_worker is not None and _rtdetr_worker.poll() is None:
        return True
    if not (_rtdetr_model_path and _rtdetr_calib_path):
        return False

    helper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rtdetr_fire_worker.py')
    if not os.path.exists(helper_path):
        return False

    _drain_rtdetr_queue()
    try:
        # CREATE_NEW_PROCESS_GROUP on Windows prevents Ctrl-C in the console
        # from propagating to the worker subprocess — the worker ignores SIGINT
        # and relies on "__quit__" on stdin for graceful shutdown.
        extra_flags = 0
        if os.name == 'nt':
            extra_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            [sys.executable, helper_path, _rtdetr_model_path, _rtdetr_calib_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            creationflags=extra_flags,
        )
    except Exception as e:
        print(f"⚠️  RT-DETR worker launch failed ({e})", flush=True)
        return False

    threading.Thread(
        target=_rtdetr_stdout_reader, args=(proc,), daemon=True, name="RTDETRWorkerStdout"
    ).start()
    threading.Thread(
        target=_rtdetr_stderr_reader, args=(proc,), daemon=True, name="RTDETRWorkerStderr"
    ).start()

    deadline = time.time() + max(RTDETR_TIMEOUT_SECONDS, 30.0)
    while time.time() < deadline:
        if proc.poll() is not None:
            _shutdown_rtdetr_worker()
            _rtdetr_launching = False
            return False
        try:
            msg = _rtdetr_worker_out.get(timeout=0.25)
        except queue.Empty:
            continue
        if msg == "READY":
            _rtdetr_worker = proc
            _rtdetr_launching = False
            return True
    _shutdown_rtdetr_worker()
    _rtdetr_launching = False
    return False


def _ensure_rtdetr_worker_async():
    global _rtdetr_launching
    if not RTDETR_SUBPROCESS_MODE:
        return
    if _rtdetr_worker is not None and _rtdetr_worker.poll() is None:
        return
    if _rtdetr_launching:
        return
    _rtdetr_launching = True

    def _launcher():
        try:
            _start_rtdetr_worker()
        finally:
            globals()["_rtdetr_launching"] = False

    threading.Thread(target=_launcher, daemon=True, name="RTDETRWorkerLauncher").start()


def _consume_rtdetr_queue():
    messages = []
    try:
        while True:
            messages.append(_rtdetr_worker_out.get_nowait())
    except queue.Empty:
        return messages


def _pump_rtdetr_worker():
    global _rtdetr_pending, _rtdetr_last_result, _rtdetr_worker_failures
    for msg in _consume_rtdetr_queue():
        if msg == "READY":
            continue
        if _rtdetr_pending is None:
            continue
        try:
            data = _json.loads(msg)
        except Exception:
            continue
        _delete_temp_file(_rtdetr_pending.get("temp_path"))
        _rtdetr_pending = None
        if data.get("ok"):
            _rtdetr_last_result = {
                "timestamp": time.time(),
                "data": data,
            }
            _rtdetr_worker_failures = 0
        else:
            _rtdetr_worker_failures += 1
            _rtdetr_last_result = None

    if _rtdetr_pending is not None:
        age = time.time() - _rtdetr_pending.get("started_at", 0.0)
        if age > RTDETR_TIMEOUT_SECONDS:
            print("⚠️  RT-DETR worker timed out; keeping last vision result and restarting worker", flush=True)
            _delete_temp_file(_rtdetr_pending.get("temp_path"))
            _rtdetr_pending = None
            _rtdetr_worker_failures += 1
            _shutdown_rtdetr_worker()


def _queue_rtdetr_inference(frame):
    global _rtdetr_pending, _rtdetr_worker_failures
    if not RTDETR_SUBPROCESS_MODE:
        return False
    if _rtdetr_pending is not None:
        return False
    if _rtdetr_worker_failures >= RTDETR_MAX_FAILURES:
        return False
    if _rtdetr_worker is None or _rtdetr_worker.poll() is not None:
        _ensure_rtdetr_worker_async()
        return False
    if _rtdetr_worker.stdin is None:
        return False

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
        if not cv2.imwrite(temp_path, frame):
            raise RuntimeError("failed to write temp frame")
        _rtdetr_worker.stdin.write(temp_path + "\n")
        _rtdetr_worker.stdin.flush()
        _rtdetr_pending = {
            "temp_path": temp_path,
            "started_at": time.time(),
        }
        return True
    except Exception as e:
        _delete_temp_file(temp_path)
        _rtdetr_worker_failures += 1
        print(f"⚠️  RT-DETR worker queue failed ({e})", flush=True)
        _shutdown_rtdetr_worker()
        return False


def _run_rtdetr_worker(frame):
    """Run RT-DETR inference in the isolated worker; returns a dict or None."""
    global _rtdetr_worker_failures
    if not RTDETR_SUBPROCESS_MODE:
        return None
    if _rtdetr_worker_failures >= RTDETR_MAX_FAILURES:
        return None
    if not _start_rtdetr_worker():
        _rtdetr_worker_failures += 1
        return None

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
        if not cv2.imwrite(temp_path, frame):
            raise RuntimeError("failed to write temp frame")

        _drain_rtdetr_queue()
        _rtdetr_worker.stdin.write(temp_path + "\n")
        _rtdetr_worker.stdin.flush()
        msg = _rtdetr_worker_out.get(timeout=RTDETR_TIMEOUT_SECONDS)
        data = _json.loads(msg)
        if data.get("ok"):
            _rtdetr_worker_failures = 0
            return data
        _rtdetr_worker_failures += 1
        return None
    except Exception as e:
        _rtdetr_worker_failures += 1
        print(f"⚠️  RT-DETR worker failed ({e}) — using HSV fallback", flush=True)
        _shutdown_rtdetr_worker()
        return None
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass

# =============================================================================
# HTTP I/O — uses ONLY Python stdlib http.client (NO urllib3 / NO requests).
#
# urllib3.create_connection causes access-violation crashes on Python 3.13
# Windows when running concurrently with PyTorch native code.  By using the
# stdlib http.client module we bypass urllib3 entirely.
#
# Architecture:
#   • Single worker thread processes a queue of HTTP requests sequentially
#   • Each request creates a fresh http.client.HTTPConnection (short-lived)
#   • No connection pooling, no urllib3, no requests library
# =============================================================================
_http_queue = queue.Queue(maxsize=64)

_esp32_reachable = True
_esp32_last_try  = 0


class _SimpleResp:
    """Minimal response object so callers can use .status_code and .json()."""
    __slots__ = ('status_code', '_body')
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
    def json(self):
        return _json.loads(self._body)


def _do_http(method, url, json_data=None, timeout=2.0):
    """Execute ONE HTTP request using stdlib http.client.  Returns _SimpleResp or None."""
    parsed = urlparse(url)
    host = parsed.hostname or 'localhost'
    port = parsed.port or 80
    path = parsed.path or '/'
    conn = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        headers = {}
        body = None
        if json_data is not None:
            body = _json.dumps(json_data).encode('utf-8')
            headers['Content-Type'] = 'application/json'
            headers['Content-Length'] = str(len(body))
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        resp_body = resp.read()
        return _SimpleResp(resp.status, resp_body)
    except Exception:
        return None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def _http_worker():
    """Runs in ONE dedicated thread.  Processes every HTTP request
    from the queue sequentially using stdlib http.client."""
    global _esp32_reachable, _esp32_last_try, _active_esp32_base_url
    while True:
        try:
            item = _http_queue.get()
            if item is None:
                break
            method, url, kwargs, evt, holder = item
            esp32_base = kwargs.get("esp32_base")
            try:
                json_data = kwargs.get('json')
                timeout   = kwargs.get('timeout', 2.0)
                r = _do_http(method, url, json_data=json_data, timeout=timeout)
                if holder is not None:
                    holder["response"] = r
                if esp32_base:
                    if r is not None and r.status_code == 200:
                        _esp32_reachable = True
                        _active_esp32_base_url = esp32_base
                    else:
                        _esp32_reachable = False
                        _esp32_last_try = time.time()
                        _mark_esp32_failure(esp32_base)
            except Exception:
                if esp32_base:
                    _esp32_reachable = False
                    _esp32_last_try = time.time()
                    _mark_esp32_failure(esp32_base)
                if holder is not None:
                    holder["response"] = None
            finally:
                if evt is not None:
                    evt.set()
                _http_queue.task_done()
        except Exception:
            pass

# Keep the legacy HTTP worker available for reference, but disable it.
# On this Windows/Python stack, main-thread outbound HTTP is more stable.
_http_worker_thread = None


def _sync_http(method, url, timeout=2.0, json_data=None, esp32_base=None, wait_timeout=None):
    global _esp32_reachable, _esp32_last_try, _active_esp32_base_url
    try:
        r = _do_http(method, url, json_data=json_data, timeout=timeout)
    except Exception:
        r = None
    if esp32_base:
        if r is not None and r.status_code == 200:
            _esp32_reachable = True
            _active_esp32_base_url = esp32_base
        else:
            _esp32_reachable = False
            _esp32_last_try = time.time()
            _mark_esp32_failure(esp32_base)
    return r


def _iter_esp32_base_urls():
    return _dedupe_keep_order([_active_esp32_base_url] + _esp32_base_urls)


def send_hardware_command(endpoint):
    """Send a GET command to the ESP32 IoT node.
    Alert commands (/alert/*) are safety-critical and bypass the cooldown."""
    if not _esp32_base_urls:
        return
    is_alert = "/alert/" in endpoint
    # Skip cooldown for alert commands — safety-critical
    if not is_alert and not _esp32_reachable and (time.time() - _esp32_last_try) < 10 and _active_esp32_base_url:
        return
    # For alerts, try ALL known base URLs until one responds
    targets = list(_iter_esp32_base_urls()) if is_alert else [_active_esp32_base_url or (_esp32_base_urls[0] if _esp32_base_urls else None)]
    for target_base in targets:
        if not target_base:
            continue
        r = _sync_http("GET", f"{target_base}{endpoint}", timeout=2.0 if is_alert else 1.0, esp32_base=target_base)
        if r is not None and r.status_code == 200:
            return   # success — stop trying
    # If we get here, no ESP32 responded


def poll_esp32_sensors():
    """Fetch live sensor data from ESP32 /data — synchronous.
    Skipped entirely if serial reader is already providing fresh data."""
    global _esp32_reachable, _esp32_last_try, _active_esp32_base_url
    # If serial is already feeding us fresh data, skip HTTP entirely
    if latest_iot.get("source") == "esp32_serial" and _has_fresh_iot_data():
        return None   # serial is working — no need for HTTP
    if not _esp32_base_urls:
        return None
    for base_url in _iter_esp32_base_urls():
        if not base_url:
            continue
        if not _esp32_reachable and (time.time() - _esp32_last_try) < 3 and base_url == _active_esp32_base_url:
            continue
        r = _sync_http("GET", f"{base_url}/data", timeout=1.5, esp32_base=base_url, wait_timeout=3.0)
        if r is not None and r.status_code == 200:
            try:
                data = r.json()
            except Exception:
                data = None
            if _update_latest_iot(data, "esp32_direct"):
                hardware_state["esp32_endpoint"] = base_url
                return data
    return None


def poll_esp32_status():
    """Fetch richer hardware state from ESP32 root endpoint when available."""
    if not _esp32_base_urls:
        return None
    for target_base in _iter_esp32_base_urls():
        if not target_base:
            continue
        r = _sync_http("GET", f"{target_base}/", timeout=1.5, esp32_base=target_base, wait_timeout=3.0)
        if r is not None and r.status_code == 200:
            try:
                data = r.json()
            except Exception:
                return None
            _merge_hardware_status(data, target_base, "esp32_direct")
            _update_latest_iot(data, "esp32_direct")
            return data
    return None


def poll_hub_iot():
    """Fetch IoT data from DT Hub — synchronous."""
    r = _sync_http("GET", HUB_STATE_URL, timeout=0.5, wait_timeout=2.0)
    if r is not None and r.status_code == 200:
        try:
            payload = r.json()
            env = payload.get("live_state", {}).get(
                "features", {}).get("environment", {}).get("properties", {})
            if env:
                _update_latest_iot(env, "hub_cache")
            hw = payload.get("live_state", {}).get(
                "features", {}).get("hardware_state", {}).get("properties", {})
            if hw:
                hardware_state.update(hw)
            return payload
        except Exception:
            return None
    return None


def push_to_hub(payload):
    """Push a PUT to the Digital Twin Hub from the main loop."""
    _sync_http("PUT", HUB_URL, timeout=0.5, json_data=payload)

# =============================================================================
# ESP32 SERIAL READER — reads USB serial data directly (no WiFi needed)
# Format from ESP32: "T:33.86,H:40.50,G:512,PRED:1,CONF:8"
# =============================================================================
def _find_esp32_serial_port():
    """Auto-detect ESP32 USB serial port (CP210x, CH340, FTDI)."""
    if not SERIAL_AVAILABLE:
        return None
    for port in _serial_ports.comports():
        desc = (port.description or "").lower()
        mfr  = (port.manufacturer or "").lower()
        vid  = port.vid or 0
        # Common ESP32 USB-serial chip vendor IDs and names
        if vid in (0x10C4, 0x1A86, 0x0403):       # SiLabs CP210x, WCH CH340, FTDI
            return port.device
        if any(k in desc for k in ("cp210", "ch340", "ftdi", "usb-serial", "silicon labs", "usb serial")):
            return port.device
        if any(k in mfr for k in ("silicon", "wch", "ftdi")):
            return port.device
    # Fallback: first COM port that isn't COM1
    ports = [p.device for p in _serial_ports.comports() if p.device.upper() != "COM1"]
    return ports[0] if ports else None


def _parse_esp32_serial_line(line):
    """Parse 'T:33.86,H:40.50,G:0,PRED:0,CONF:0' → dict."""
    data = {}
    try:
        for part in line.strip().split(","):
            if ":" not in part:
                continue
            key, val = part.split(":", 1)
            key = key.strip().upper()
            if key == "T":
                data["temperature"] = float(val)
            elif key == "H":
                data["humidity"] = float(val)
            elif key == "G":
                data["gas_level"] = float(val)
            elif key == "PRED":
                data["esp32_prediction"] = int(val)
            elif key == "CONF":
                data["esp32_confidence"] = int(val)
    except Exception:
        pass
    return data if "temperature" in data else None


def serial_reader_thread():
    """Continuously read ESP32 sensor data from USB serial port."""
    global running, _serial_port_name
    port_name = _find_esp32_serial_port()
    if not port_name:
        print("[Serial] ⚠️  No ESP32 serial port detected", flush=True)
        return

    _serial_port_name = port_name
    print(f"[Serial] Connecting to ESP32 on {port_name} @ 115200 baud...", flush=True)
    try:
        ser = _pyserial.Serial(port_name, 115200, timeout=2)
        print(f"[Serial] ✅ Connected to {port_name}", flush=True)
    except Exception as e:
        print(f"[Serial] ⚠️  Could not open {port_name}: {e}", flush=True)
        print(f"[Serial]    → Close Arduino Serial Monitor if it is open, then restart", flush=True)
        return

    consecutive_ok = 0
    while running:
        try:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            # Only parse data lines (start with "T:")
            if not line.startswith("T:"):
                continue
            data = _parse_esp32_serial_line(line)
            if data:
                _update_latest_iot(data, "esp32_serial")
                consecutive_ok += 1
                if consecutive_ok == 1:
                    print(f"[Serial] ✅ Receiving ESP32 data: T={data.get('temperature')}°C  "
                          f"H={data.get('humidity')}%  G={data.get('gas_level')}", flush=True)
        except Exception as e:
            if running:
                print(f"[Serial] ⚠️  Read error: {e}", flush=True)
            time.sleep(1)

    try:
        ser.close()
    except Exception:
        pass


# =============================================================================
# AUDIO SENSOR
# =============================================================================
def audio_callback(indata, frames, t, status):
    global audio_volume, audio_stress_score
    volume_norm = float(np.linalg.norm(indata) * 10)
    audio_volume = volume_norm
    if audio_volume > 15:
        audio_stress_score = 100
    elif audio_volume > 8:
        audio_stress_score = 50
    else:
        audio_stress_score = max(0, audio_stress_score - 5)

def start_audio_listener():
    if not AUDIO_AVAILABLE:
        return
    try:
        stream = sd.InputStream(callback=audio_callback)
        stream.start()
        print("✅ Audio sensor active")
    except Exception as e:
        print(f"⚠️  Audio sensor error: {e}")

# =============================================================================
# FIRE DETECTION — HSV fallback (used if RT-DETR not available)
# =============================================================================
def detect_fire_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Range 1: orange-red flames (H 0-35, high saturation + value)
    mask1 = cv2.inRange(hsv,
                        np.array([0, 100, 150], dtype="uint8"),
                        np.array([35, 255, 255], dtype="uint8"))
    # Range 2: bright yellow/white flame cores (low saturation, very high value)
    mask2 = cv2.inRange(hsv,
                        np.array([15, 50, 200], dtype="uint8"),
                        np.array([40, 180, 255], dtype="uint8"))
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best, max_area = None, 0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a > 300 and a > max_area:   # lowered from 1500 to catch small flames
            max_area = a
            best = cnt
    if best is not None:
        x, y, w, h = cv2.boundingRect(best)
        conf = min(0.8, 0.4 + (max_area / 5000.0))   # scale confidence with size
        return True, (x, y, w, h), conf
    return False, None, 0.0

# =============================================================================
# ZONE DETECTION — tag each person to a named zone
# =============================================================================
def get_zone(norm_x, norm_y):
    for zone_id, b in ZONES.items():
        if b["x"] <= norm_x <= b["x"] + b["w"] and b["y"] <= norm_y <= b["y"] + b["h"]:
            return zone_id
    return "main_seating"   # default fallback

# =============================================================================
# IOT FIRE PROBABILITY — simple threshold model on available sensors
# (used when full 8-feature RF model is not applicable)
# =============================================================================
def iot_fire_probability(temp, humidity, gas):
    """
    Map temperature + gas readings to a fire probability ∈ [0, 1].
    Conservative thresholds from the smoke_detection_iot dataset analysis.
    """
    score = 0.0
    if temp > 60:     score += 0.45
    elif temp > 45:   score += 0.25
    elif temp > 35:   score += 0.10

    # Gas sensor: MQ2 raw ADC value (0-4095)
    if gas > 2000:    score += 0.45
    elif gas > 1200:  score += 0.25
    elif gas > 600:   score += 0.10

    # High humidity can dampen fire signature
    if humidity > 80: score *= 0.7

    return min(1.0, score)

def iot_entropy(temp, humidity, gas):
    """Rough entropy: how uncertain we are about the IoT reading (0=certain, 1=max)."""
    p = iot_fire_probability(temp, humidity, gas)
    if p < 0.05 or p > 0.95: return 0.1   # certain
    if 0.40 < p < 0.60:      return 0.9   # very uncertain
    return abs(p - 0.5) * -2 + 1           # linear

# =============================================================================
# THREAD 1: Camera capture
# =============================================================================
def _open_camera_capture():
    candidates = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("MediaFoundation", cv2.CAP_MSMF),
        ("Auto", None),
    ]
    ordered = candidates[_camera_backend_cursor:] + candidates[:_camera_backend_cursor]
    for backend_name, backend in ordered:
        try:
            cam = cv2.VideoCapture(CAMERA_ID, backend) if backend is not None else cv2.VideoCapture(CAMERA_ID)
            if cam is not None and cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Let auto-exposure handle brightness — manual gain/brightness
                # cause green color cast on most webcams.
                cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)   # 3 = full auto on most backends
                cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # minimize capture latency
                return cam, backend_name
            if cam is not None:
                cam.release()
        except Exception:
            continue
    return None, None


def camera_thread():
    global global_frame, running, cap, _camera_backend_cursor
    status_frame = _make_status_frame("Opening camera...", "Trying available OpenCV backends")
    _set_global_frame(status_frame)
    _set_video_frame(status_frame)
    _set_camera_status(False, "Opening camera")
    consecutive_failures = 0
    dark_frames = 0

    while running:
        try:
            if cap is None or not cap.isOpened():
                cap, backend_name = _open_camera_capture()
                if cap is None or not cap.isOpened():
                    _set_camera_status(False, "Camera unavailable")
                    status_frame = _make_status_frame(
                        "Camera unavailable",
                        "The agent will keep retrying automatically",
                        accent=(0, 140, 255),
                    )
                    _set_global_frame(status_frame)
                    _set_video_frame(status_frame)
                    time.sleep(CAMERA_REOPEN_DELAY)
                    continue
                consecutive_failures = 0
                dark_frames = 0
                camera_state["reconnects"] += 1
                _set_camera_status(True, f"Camera live ({backend_name})", backend_name)
                print(f"✅ Camera opened successfully via {backend_name}")

            ret, raw_frame = cap.read()
            if ret and raw_frame is not None and getattr(raw_frame, "size", 0) > 0:
                raw_frame = cv2.resize(raw_frame, (640, 480))
                if len(raw_frame.shape) == 2:
                    raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
                mean_brightness = float(raw_frame.mean())

                # ── global_frame = RAW frame (untouched colors) ──────────
                # AI models (YOLO, RT-DETR, HSV fire) need true colors.
                # CLAHE/gamma distort hue → kills fire detection accuracy.
                _set_global_frame(raw_frame)

                # ── display_frame = CLAHE-enhanced (for human viewing only)
                display_frame = raw_frame
                if 2.0 <= mean_brightness < 100.0:
                    lab = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2LAB)
                    l, a, b_ch = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    if mean_brightness < 50.0:
                        gamma = 0.55 if mean_brightness < 25.0 else 0.75
                        lut = np.array([((i / 255.0) ** gamma) * 255
                                        for i in range(256)], dtype=np.uint8)
                        l = cv2.LUT(l, lut)
                    lab = cv2.merge([l, a, b_ch])
                    display_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                if mean_brightness < 2.0:
                    dark_frames += 1
                else:
                    dark_frames = 0
                # Render annotations on the DISPLAY frame (not the AI frame)
                _set_video_frame(_render_annotated_frame(display_frame, global_detections))
                _set_camera_status(True, f"Camera live ({camera_state.get('backend') or 'unknown'})")
                consecutive_failures = 0
                if dark_frames >= 20:
                    print("⚠️  Camera frames are nearly black; switching backend", flush=True)
                    _camera_backend_cursor = (_camera_backend_cursor + 1) % 3
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    dark_frames = 0
                    status_frame = _make_status_frame(
                        "Camera backend switch",
                        "Detected black frames; trying another backend",
                        accent=(0, 200, 255),
                    )
                    _set_video_frame(status_frame)
                    time.sleep(CAMERA_REOPEN_DELAY)
            else:
                consecutive_failures += 1
                if consecutive_failures >= 15:
                    _set_camera_status(False, "Reconnecting camera")
                    status_frame = _make_status_frame(
                        "Camera reconnecting...",
                        "Capture dropped frames; reopening stream",
                        accent=(0, 200, 255),
                    )
                    _set_global_frame(status_frame)
                    _set_video_frame(status_frame)
                    try:
                        if cap is not None:
                            cap.release()
                    except Exception:
                        pass
                    cap = None
                    time.sleep(CAMERA_REOPEN_DELAY)
                else:
                    time.sleep(0.08)
        except Exception as e:
            print(f"⚠️  Camera read error: {e}")
            _set_camera_status(False, "Camera error")
            status_frame = _make_status_frame(
                "Camera error",
                "Retrying capture automatically",
                accent=(0, 140, 255),
            )
            _set_global_frame(status_frame)
            _set_video_frame(status_frame)
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            cap = None
            time.sleep(CAMERA_REOPEN_DELAY)
        time.sleep(0.01)

# =============================================================================
# THREAD 2: AI Intelligence Engine
# =============================================================================
def ai_loop():
    """AI inference loop — MUST run in the MAIN thread on Windows.
    PyTorch native code (MKL/BLAS/Conv) crashes with access violations
    when called from daemon threads on Python 3.13.  Keeping all torch
    operations in the main OS thread avoids this entirely."""
    global global_detections, running, audio_stress_score, latest_iot

    last_sent        = 0
    last_podium_state = "OFF"
    last_fire_state   = False
    last_alert_mode   = "off"   # "off" | "warning" | "fire"
    last_hw_status_refresh = 0.0

    # Per-zone exit history for flow calculation
    zone_exit_history = deque(maxlen=5)
    fire_last_seen    = 0
    FIRE_COOLDOWN     = 3.0
    last_fire_eval_ts = 0.0
    last_fire_eval = {
        "fire_score": 0.0,
        "vision_det_count": 0,
        "vision_conf_var": 0.0,
        "fire_loc": None,
        "frame_has_fire": False,
        "detections": [],
    }

    frame_counter = 0
    _consecutive_errors = 0
    _MAX_ERRORS = 20  # after this many consecutive errors, pause for longer

    while running:
      try:
        # ── Grab frame ───────────────────────────────────────────────────────
        frame_snapshot = None
        with global_lock:
            if global_frame is not None:
                frame_snapshot = global_frame.copy()
        if frame_snapshot is None:
            time.sleep(0.1)
            continue

        height, width, _ = frame_snapshot.shape
        camera_live      = _camera_is_live()
        temp_detections  = []
        frame_counter   += 1

        # ── FIRE DETECTION ────────────────────────────────────────────────────
        fire_score       = 0.0
        vision_det_count = 0
        vision_conf_var  = 0.0
        fire_loc         = None
        frame_has_fire   = False

        if camera_live:
            _pump_rtdetr_worker()
            should_run_fire_model = (time.time() - last_fire_eval_ts) >= FIRE_INFERENCE_INTERVAL
            if should_run_fire_model:
                fresh_eval = {
                    "fire_score": 0.0,
                    "vision_det_count": 0,
                    "vision_conf_var": 0.0,
                    "fire_loc": None,
                    "frame_has_fire": False,
                    "detections": [],
                }
                if RTDETR_SUBPROCESS_MODE and _rtdetr_model_path and fusion_engine is not None:
                    _queue_rtdetr_inference(frame_snapshot)
                    worker_out = (_rtdetr_last_result or {}).get("data")
                    if worker_out is not None and (time.time() - (_rtdetr_last_result or {}).get("timestamp", 0.0)) < 5.0:
                        fresh_eval["fire_score"] = float(worker_out.get("calibrated_fire_score", 0.0))
                        fresh_eval["vision_det_count"] = int(worker_out.get("detection_count", 0))
                        fresh_eval["vision_conf_var"] = float(worker_out.get("confidence_variance", 0.0))
                        fresh_eval["frame_has_fire"] = bool(
                            worker_out.get("has_fire", False) or (fresh_eval["fire_score"] > 0.5)
                        )

                        for det in worker_out.get("detections", []):
                            x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
                            cx = (x1 + x2) / 2 / width if width else 0
                            cy = (y1 + y2) / 2 / height if height else 0
                            if det.get("class_name") == 'fire' and fresh_eval["fire_loc"] is None:
                                fresh_eval["fire_loc"] = {"x": cx, "y": cy}
                            color = (0, 0, 255) if det.get("class_name") == 'fire' else (0, 128, 255)
                            label = f"{str(det.get('class_name', 'obj')).upper()} {float(det.get('confidence', 0.0)):.2f}"
                            fresh_eval["detections"].append((x1, y1, x2, y2, label, color))
                elif vision_wrapper is not None:
                    try:
                        with torch.inference_mode():
                            v_out = vision_wrapper.predict(frame_snapshot)
                        fresh_eval["fire_score"] = v_out.calibrated_fire_score
                        fresh_eval["vision_det_count"] = v_out.detection_count
                        fresh_eval["vision_conf_var"] = v_out.confidence_variance
                        fresh_eval["frame_has_fire"] = v_out.has_fire or (fresh_eval["fire_score"] > 0.5)

                        for det in v_out.detections:
                            x1, y1, x2, y2 = map(int, det.bbox)
                            cx = (x1 + x2) / 2 / width
                            cy = (y1 + y2) / 2 / height
                            if det.class_name == 'fire' and fresh_eval["fire_loc"] is None:
                                fresh_eval["fire_loc"] = {"x": cx, "y": cy}
                            color = (0, 0, 255) if det.class_name == 'fire' else (0, 128, 255)
                            label = f"{det.class_name.upper()} {det.confidence:.2f}"
                            fresh_eval["detections"].append((x1, y1, x2, y2, label, color))
                    except Exception:
                        pass

                if fresh_eval["vision_det_count"] == 0 and not fresh_eval["frame_has_fire"]:
                    frame_has_fire, blob, hsv_score = detect_fire_hsv(frame_snapshot)
                    fresh_eval["frame_has_fire"] = frame_has_fire
                    fresh_eval["fire_score"] = max(fresh_eval["fire_score"], hsv_score)
                    if blob:
                        bx, by, bw, bh = blob
                        fresh_eval["fire_loc"] = {"x": (bx + bw/2)/width, "y": (by + bh/2)/height}
                        fresh_eval["detections"].append((bx, by, bx+bw, by+bh, "FIRE (HSV)", (0,0,255)))

                last_fire_eval = fresh_eval
                last_fire_eval_ts = time.time()

            fire_score = last_fire_eval["fire_score"]
            vision_det_count = last_fire_eval["vision_det_count"]
            vision_conf_var = last_fire_eval["vision_conf_var"]
            fire_loc = last_fire_eval["fire_loc"]
            frame_has_fire = last_fire_eval["frame_has_fire"]
            temp_detections.extend(last_fire_eval["detections"])

        # Temporal fire cooldown
        if frame_has_fire:
            fire_last_seen = time.time()
        stable_fire_flag = (time.time() - fire_last_seen) < FIRE_COOLDOWN

        # ── IoT DATA: poll ESP32 directly, fallback to hub ──────────────────
        if frame_counter % 10 == 0:   # refresh IoT every ~3 s
            esp_data = poll_esp32_sensors()
            if not esp_data:
                try:
                    poll_hub_iot()
                except Exception:
                    pass
        if (time.time() - last_hw_status_refresh) >= ESP32_STATUS_INTERVAL:
            try:
                poll_esp32_status()
            except Exception:
                pass
            last_hw_status_refresh = time.time()

        # ── SENSOR FUSION ─────────────────────────────────────────────────────
        iot_available = _has_fresh_iot_data()
        iot_temp, iot_humidity, iot_gas = _get_iot_values()
        p_iot = iot_fire_probability(iot_temp, iot_humidity, iot_gas) if iot_available else 0.0
        e_iot = iot_entropy(iot_temp, iot_humidity, iot_gas) if iot_available else 1.0
        vision_available = camera_live

        if fusion_engine is not None:
            fusion_input = FusionInput(
                iot_probability          = p_iot,
                iot_entropy              = e_iot,
                iot_vote_distribution    = [round(1 - p_iot, 2), round(p_iot, 2)],
                iot_available            = iot_available,
                vision_fire_score        = fire_score,
                vision_detection_count   = vision_det_count,
                vision_confidence_variance = vision_conf_var,
                vision_has_fire          = frame_has_fire,
                vision_has_smoke         = False,
                vision_available         = vision_available,
                timestamp                = time.time()
            )
            fusion_out     = fusion_engine.fuse(fusion_input)
            fused_fire     = fusion_out.alarm_triggered
            fusion_score   = round(fusion_out.fused_score * 100, 1)
            fusion_mode    = fusion_out.mode
            fusion_weights = {
                "w_iot":    round(fusion_out.w_iot, 3),
                "w_vision": round(fusion_out.w_vision, 3),
                "iot_rel":  round(fusion_out.iot_reliability, 3),
                "vis_rel":  round(fusion_out.vision_reliability, 3),
                "conflict": round(fusion_out.conflict_magnitude, 3),
            }
        else:
            # Simple threshold fusion
            fused_fire   = stable_fire_flag and ((iot_available and p_iot > 0.4) or (vision_available and fire_score > 0.5))
            fusion_score = round(max(fire_score * 100, p_iot * 100), 1)
            fusion_mode  = "threshold"
            fusion_weights = {"w_iot": 0.35, "w_vision": 0.65}

        # Safety-first override: if the fire model itself produced a fire box,
        # do not let calibration / IoT absence suppress the alert.
        if vision_available and frame_has_fire and vision_det_count > 0:
            fused_fire = True
            fusion_score = max(fusion_score, round(max(fire_score, 0.80) * 100, 1))
            if fusion_mode in ("vision_only", "dual", "threshold"):
                fusion_mode = f"{fusion_mode}_vision_confirmed"

        # ── PERSON DETECTION ──────────────────────────────────────────────────
        occupant_locations = []
        zone_occupancy     = {z: [] for z in ZONES}
        exit_count         = 0

        if camera_live:
            with torch.inference_mode():
                results = yolo_model(frame_snapshot, verbose=False, conf=0.15)
            for r in results:
                for box in r.boxes:
                    if yolo_model.names[int(box.cls[0])] == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        nx = cx / width
                        ny = cy / height
                        zone_id = get_zone(nx, ny)
                        loc = {"x": round(nx, 4), "y": round(ny, 4), "zone": zone_id}
                        occupant_locations.append(loc)
                        zone_occupancy[zone_id].append(loc)

                        if zone_id == "exit_corridor":
                            exit_count += 1
                            color = (0, 0, 255)
                            label = "EXIT SURGE"
                        elif zone_id == "stage_area":
                            color = (0, 255, 255)
                            label = "PODIUM"
                        else:
                            color = (0, 255, 0)
                            label = "OCCUPANT"
                        temp_detections.append((x1, y1, x2, y2, label, color))

        global_detections = temp_detections
        # video_frame is now updated by camera_thread at ~30fps using the
        # latest global_detections — no need to set it here.

        # ── CROWD STRESS INDEX (CSI) ──────────────────────────────────────────
        zone_exit_history.append(exit_count)
        avg_exit = sum(zone_exit_history) / len(zone_exit_history)
        flow_velocity = max(0.0, float(exit_count - avg_exit) * 2)

        stagnation_multiplier = 1.0
        if exit_count > 2:
            stagnation_multiplier = 1 + (exit_count / (flow_velocity + 1))

        visual_stress  = (exit_count * 10) * stagnation_multiplier

        # Fire component: active fire raises CSI proportionally to fusion score.
        # Even with zero crowd movement, a fire is inherently stressful.
        fire_stress = 0.0
        if stable_fire_flag:
            fire_stress = max(50.0, fusion_score * 0.9)   # 50-90 based on confidence

        final_csi      = max(visual_stress, audio_stress_score, fire_stress)
        final_csi      = min(100, int(final_csi))

        # CSI state machine
        is_podium_occupied = len(zone_occupancy.get("stage_area", [])) > 0
        if fused_fire or final_csi > 75:
            prediction  = "CRITICAL"
            podium_cmd  = "ON"
        elif final_csi > 40:
            prediction  = "WARNING"
            podium_cmd  = "ON"
        else:
            prediction  = "STABLE"
            podium_cmd  = "ON" if is_podium_occupied else "OFF"

        # ── HARDWARE CONTROL ──────────────────────────────────────────────────
        if podium_cmd != last_podium_state:
            send_hardware_command("/light/on" if podium_cmd == "ON" else "/light/off")
            last_podium_state = podium_cmd

        # Determine desired alert mode: fire > warning > off
        if fused_fire:
            desired_alert = "fire"
        elif prediction == "CRITICAL":
            desired_alert = "warning"
        else:
            desired_alert = "off"

        if desired_alert != last_alert_mode:
            send_hardware_command(f"/alert/{desired_alert}")
            last_alert_mode = desired_alert

        hardware_state["light_active"] = podium_cmd == "ON"
        hardware_state["alert_mode"] = desired_alert.upper()
        last_fire_state = fused_fire

        # ── FIRE LOCATION FALLBACK & ZONE TAGGING ────────────────────────────
        # If fire is detected but no visual bounding box (IoT-driven fire),
        # place the marker at the IoT sensor location (stage_area center).
        effective_fire_loc = fire_loc
        fire_zone = None
        if fused_fire:
            if effective_fire_loc is None:
                # IoT sensor is mounted at normalised (0.50, 0.18) in stage_area
                effective_fire_loc = {"x": 0.50, "y": 0.18}
            # Tag the zone the fire is in
            fire_zone = get_zone(effective_fire_loc["x"], effective_fire_loc["y"])

        # ── PUSH TO DIGITAL TWIN HUB ──────────────────────────────────────────
        if time.time() - last_sent > 0.5:
            try:
                features = {
                    "safety_state": {
                        "properties": {
                            "is_fire_detected": fused_fire,
                            "fire_location": effective_fire_loc,
                            "fire_zone": fire_zone,
                            "fusion_score_pct": fusion_score,
                            "fusion_mode": fusion_mode,
                            "fusion_weights": fusion_weights,
                            "iot_fire_prob": round(p_iot, 3),
                            "vision_fire_score": round(fire_score, 3)
                        }
                    },
                    "occupancy": {
                        "properties": {
                            "current": len(occupant_locations),
                            "locations": occupant_locations,
                            "zone_counts": {
                                z: len(locs)
                                for z, locs in zone_occupancy.items()
                            }
                        }
                    },
                    "automation": {
                        "properties": {
                            "podium_light": podium_cmd == "ON",
                            "nudge_active": prediction == "WARNING",
                            "alarm_active": prediction == "CRITICAL"
                        }
                    },
                    "crowd_analytics": {
                        "properties": {
                            "pressure_score": final_csi,
                            "flow_rate": round(max(flow_velocity, fire_stress / 30.0), 2),
                            "prediction": prediction,
                            "stagnation_multiplier": round(stagnation_multiplier, 2),
                            "audio_volume": round(audio_volume, 1),
                            "audio_stress": audio_stress_score
                        }
                    },
                    "hardware_state": {
                        "properties": {
                            "esp32_online": hardware_state.get("esp32_online", False),
                            "esp32_endpoint": hardware_state.get("esp32_endpoint"),
                            "last_contact_ts": hardware_state.get("last_contact_ts"),
                            "data_source": hardware_state.get("data_source"),
                            "wifi_rssi": hardware_state.get("wifi_rssi"),
                            "sensor_ok": hardware_state.get("sensor_ok"),
                            "alert_mode": hardware_state.get("alert_mode"),
                            "light_active": hardware_state.get("light_active")
                        }
                    },
                    "vision_state": {
                        "properties": {
                            "camera_online": camera_live,
                            "backend": camera_state.get("backend"),
                            "status": camera_state.get("status"),
                            "last_frame_ts": camera_state.get("last_frame_ts"),
                            "reconnects": camera_state.get("reconnects", 0)
                        }
                    }
                }
                # Always include environment — dashboard needs it even
                # when ESP32 is offline (shows defaults rather than --)
                features["environment"] = {
                    "properties": {
                        "temperature": iot_temp,
                        "humidity": iot_humidity,
                        "gas_level": iot_gas,
                        "iot_fresh": iot_available
                    }
                }
                payload = {"features": features}
                push_to_hub(payload)
                last_sent = time.time()
            except Exception:
                pass

        _consecutive_errors = 0  # reset on successful iteration
        time.sleep(AI_INTERVAL)

      except Exception as _loop_err:
        _consecutive_errors += 1
        import traceback as _tb
        print(f"[AI] ⚠️  Loop error #{_consecutive_errors}: {_loop_err}", flush=True)
        _tb.print_exc()
        if _consecutive_errors >= _MAX_ERRORS:
            print(f"[AI] Too many consecutive errors ({_MAX_ERRORS}), pausing 10s...", flush=True)
            time.sleep(10)
            _consecutive_errors = 0
        else:
            time.sleep(1)

    print("[AI] ai_loop exited (running=False)", flush=True)


# =============================================================================
# VIDEO STREAM (MJPEG)
# =============================================================================
def generate_video():
    while running:
        output = None
        with global_lock:
            output = video_frame.copy() if video_frame is not None else global_frame.copy()

        # Safety: skip encoding if frame is somehow None or empty
        if output is None or output.size == 0:
            time.sleep(0.1)
            continue

        flag, enc = cv2.imencode(".jpg", output,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), 55])
        if flag:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + bytearray(enc) + b'\r\n')
        time.sleep(0.033)   # ~30fps cap


@app.route("/video_feed")
def video_feed():
    return Response(generate_video(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# =============================================================================
# SAFE CAMERA TEST (runs in a subprocess — if it segfaults, main process lives)
# =============================================================================
def _test_camera_subprocess():
    """Returns True if camera 0 can be opened without crashing."""
    try:
        r = subprocess.run(
            [sys.executable, '-c',
             'import cv2,sys; c=cv2.VideoCapture(0); print(c.isOpened()); c.release()'],
            capture_output=True, text=True, timeout=15
        )
        return 'True' in r.stdout
    except Exception:
        return False


# =============================================================================
# ENTRY POINT
# =============================================================================
#
# ARCHITECTURE (Windows Python 3.13 safe):
#
#   MAIN THREAD  →  model warmup  →  ai_loop()  [ALL PyTorch here]
#   Daemon #1    →  Waitress / Flask HTTP server  (video feed)
#   Daemon #2    →  camera_thread()  (cv2.VideoCapture)
#
# WHY: PyTorch's native MKL/BLAS/Conv code triggers access-violation
# crashes when called from daemon threads on Python 3.13 Windows.
# The main OS thread has proper SEH handling and is always safe.
# =============================================================================

if __name__ == '__main__':
    import traceback
    import functools

    # Force-flush every print so output isn't lost on native crash
    _builtin_print = print
    print = functools.partial(_builtin_print, flush=True)

    # ── Catch unhandled thread exceptions (Python 3.8+) ─────────────────────
    def _thread_exception_handler(args):
        print(f"\n❌ UNHANDLED THREAD EXCEPTION in '{args.thread.name}':", flush=True)
        import traceback as _tb
        _tb.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
        print("   (Thread crashed but server is still running)\n", flush=True)

    threading.excepthook = _thread_exception_handler

    actual_port = VISION_PORT

    # ── 1. Warm up models IN THE MAIN THREAD before anything else ───────────
    # This triggers all lazy imports + JIT/MKL initialisation in the main
    # OS thread, which is the ONLY thread that will ever touch PyTorch.
    print("=" * 60)
    print("  CrowdPulse Vision Agent — starting up")
    print("=" * 60)
    _warmup_models()

    # ── 2. Test camera availability (subprocess — safe, no torch) ───────────
    print("[INIT] Testing camera availability (subprocess)...")
    CAMERA_SAFE = _test_camera_subprocess()
    if CAMERA_SAFE:
        print("[INIT] Camera test passed ✓")
    else:
        print("[INIT] Camera not available or unstable — running without live video")
        print("       (IoT fusion + AI analytics will still push data to hub)")

    start_audio_listener()

    # ── 2b. Start ESP32 serial reader (direct USB — no WiFi dependency) ────
    if SERIAL_AVAILABLE:
        serial_t = threading.Thread(target=serial_reader_thread, daemon=True, name="ESP32SerialReader")
        serial_t.start()
        time.sleep(0.5)   # give serial a moment to connect
    else:
        print("[Serial] Skipped (pyserial not installed). Using HTTP polling only.", flush=True)

    # ── 3. Start Waitress / Flask HTTP server in a DAEMON thread ────────────
    def _serve():
        try:
            from waitress import serve as waitress_serve
            print("   (Using Waitress production server)", flush=True)
            waitress_serve(app, host='0.0.0.0', port=actual_port, threads=4)
        except ImportError:
            print("   (Using Flask development server — install 'waitress' for stability)", flush=True)
            app.run(host="0.0.0.0", port=actual_port, debug=False,
                    threaded=True, use_reloader=False)
        except Exception as e:
            print(f"\n❌ Server error: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    server_t = threading.Thread(target=_serve, daemon=True, name="WaitressServer")
    server_t.start()
    time.sleep(1)   # let server bind before camera + AI start

    print(f"✅ HTTP server on http://0.0.0.0:{actual_port}")
    print(f"   Video feed: http://localhost:{actual_port}/video_feed")

    # ── 4. Start camera capture in a DAEMON thread ─────────────────────────
    if CAMERA_SAFE:
        def _safe_camera_thread():
            try:
                camera_thread()
            except Exception as e:
                print(f"❌ Camera thread crashed: {e}", flush=True)
                traceback.print_exc()

        cam_t = threading.Thread(target=_safe_camera_thread, daemon=True, name="CameraThread")
        cam_t.start()
        time.sleep(1)   # let camera open before AI reads frames

    # ── 5. Run AI inference loop IN THE MAIN THREAD ─────────────────────────
    # This is the key architectural decision: the main OS thread keeps
    # exclusive ownership of all PyTorch operations.  Camera and HTTP
    # run in daemon threads (no torch).  When Ctrl-C is pressed, the
    # main thread exits and all daemon threads are cleaned up.
    print("[INIT] Starting AI inference in main thread...")
    print("[INIT] All systems go ✓")
    print("-" * 60)

    try:
        ai_loop()      # blocks main thread — runs forever
    except KeyboardInterrupt:
        print("\n⚠️  Ctrl-C received — shutting down gracefully...")
        running = False
    except Exception as e:
        print(f"\n❌ AI loop crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        running = False

    # Shut down worker before Python tears down the interpreter
    try:
        _shutdown_rtdetr_worker()
    except Exception:
        pass

    # Release camera
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass

    print("\n⚠️  Vision Agent has stopped.")
    print("   Press Ctrl-C again or close this window.")