# SPATIAL MONITORING DASHBOARD AND IoT FOR CAMPUS SAFETY AND CRISIS RESILIENCE
The project is referred to as CROWDPULSE throughout the documentation.

## Overview

CrowdPulse is a real-time spatial monitoring and crisis resilience system built on a Digital Twin architecture. It fuses computer vision (YOLOv8 for person detection, RT-DETR for fire/smoke detection), IoT sensor data (ESP32 with AHT10 temperature/humidity and MQ2 gas sensors), and audio analysis into a unified awareness engine. The system maintains a semantic model of a physical venue (seminar hall), computes zone-level crowd analytics, performs adaptive confidence-based multi-modal fire detection, and presents all state through a React-based dashboard with a 2D canvas venue renderer, timeline replay, and what-if simulation capabilities.

The system pushes structured state updates to a Flask-based Digital Twin Hub at 0.5 Hz, enabling bidirectional synchronization between the physical space and its digital representation.

## Features

- **Zone-Aware Person Detection**: YOLOv8n detects occupants in camera frames and tags each person to a named venue zone (Entry Lobby, Stage/Podium, Main Seating, Exit Corridor) using normalised coordinate bounds.
- **Multi-Modal Fire Detection**: RT-DETR transformer-based detector with Platt-scaled confidence calibration, fused with IoT sensor probabilities via an Adaptive Fusion Engine. Falls back to HSV color-space blob detection when RT-DETR is unavailable.
- **Adaptive Confidence-Based Fusion**: Reliability-weighted fusion of IoT and vision modalities with entropy-based IoT reliability scoring, detection count/variance-based vision reliability scoring, temporal EMA smoothing, and a hysteresis state machine to prevent alarm flicker.
- **Crowd Stress Index (CSI)**: Composite metric (0-100) derived from exit corridor flow rates, stagnation multipliers, audio stress levels, and fire threat scores. Drives a three-state alarm machine: STABLE, WARNING, CRITICAL.
- **Semantic Venue Model**: The Digital Twin Hub maintains a full venue schema with named zones (capacities, bounds, colors), physical exits (positions, widths, statuses), sensor locations, and pre-computed evacuation routes with per-minute capacity ratings.
- **State History and Timeline Replay**: Circular buffer of the last 240 state snapshots (~2 minutes at 0.5 Hz). The dashboard provides a clickable sparkline chart to scrub through historical states.
- **What-If Simulation Engine**: POST endpoint that accepts a fire scenario (origin zone, current occupancy) and computes available evacuation routes, estimated clearance times, congested zones, and recommended interventions using the semantic venue model.
- **ESP32 IoT Node**: Arduino-based firmware providing on-device Random Forest fire pre-screening, WiFi HTTP server with sensor/control endpoints, USB serial output for direct Raspberry Pi connectivity, and bidirectional actuator control (buzzer alerts, LED/podium light).
- **Audio Stress Detection**: Real-time microphone monitoring via sounddevice. Volume-based stress scoring (thresholds at 8 and 15) feeds into the CSI calculation.
- **Browser Voice Nudge**: The dashboard uses the Web Speech Synthesis API to deliver audible evacuation instructions when the system transitions to WARNING or CRITICAL states.
- **Live Video Feed**: MJPEG video stream served by the Vision Agent on port 5010, with CLAHE-enhanced display frames (separate from raw AI inference frames) and annotated zone overlays, detection bounding boxes, and status text.
- **Event Logging**: Structured event log (last 100 events) capturing state transitions (CROWD_CRITICAL, FIRE_DETECTED, FIRE_CLEARED, SIMULATION_RUN) with timestamps and severity levels.
- **ESP32 Hardware Control**: Bidirectional commands to the ESP32 for buzzer alert modes (off, warning beep, continuous fire siren) and LED/light control, sent via HTTP GET endpoints.
- **Isolated RT-DETR Worker**: On Windows, RT-DETR inference runs in a separate subprocess to prevent native instruction crashes (AVX-512 incompatibility) from killing the main vision agent process.

## Tech Stack

### Backend

| Component | Technology |
|---|---|
| Digital Twin Hub | Python, Flask, Flask-CORS |
| Vision Agent | Python, OpenCV, PyTorch, Ultralytics (YOLOv8, RT-DETR) |
| Audio Processing | sounddevice, scipy |
| Fusion Engine | NumPy, scikit-learn (Random Forest, Platt Scaling) |
| Model Serialization | joblib |
| Data Analysis | pandas |
| HTTP Client | Python stdlib http.client (urllib3 intentionally avoided) |
| Production Server | Waitress (WSGI) |
| Serial Communication | pyserial |

### Frontend

| Component | Technology |
|---|---|
| Framework | React 19 |
| HTTP Client | Axios |
| Venue Renderer | HTML5 Canvas 2D (no external 3D libraries) |
| Build Tool | Create React App (react-scripts 5.0.1) |
| Voice Output | Web Speech Synthesis API |

### IoT / Embedded

| Component | Technology |
|---|---|
| Microcontroller | ESP32 |
| Firmware Language | C++ (Arduino framework) |
| Sensors | AHT10 (I2C, temperature/humidity), MQ2 (analog, gas) |
| On-Device ML | micromlgen-exported Random Forest (SentryModel.h) |
| Networking | WiFi (WebServer, HTTPClient, ESPmDNS) |
| Actuators | Buzzer (GPIO15), LED (GPIO2) |

### ML Models

| Model | Purpose | Framework |
|---|---|---|
| YOLOv8n | Person detection | Ultralytics |
| RT-DETR-L | Fire and smoke detection | Ultralytics (RTDETR) |
| Random Forest | IoT sensor fire classification (8 features) | scikit-learn |
| Platt Scaling (Logistic Regression) | Confidence calibration for IoT and Vision outputs | scikit-learn |

## Project Structure

```
CrowdPulse_Project/
├── crowdpulse_server.py         # Digital Twin Hub — Flask server (port 5000)
├── vision_agent.py              # Perception Engine — vision, audio, IoT fusion (port 5010)
├── rtdetr_fire_worker.py        # Isolated RT-DETR subprocess worker
├── server.py                    # Legacy Digital Twin Hub (simple merge, superseded)
├── setup.py                     # Twin schema initializer — seeds default state in hub
├── CrowdPulse_Start.bat         # Windows batch launcher for all components
├── requirements.txt             # Python dependencies
├── yolov8n.pt                   # YOLOv8n pre-trained weights (person detection)
├── fusion_log.csv               # Runtime fusion decision log
│
├── dt-dashboard/                # React frontend
│   ├── package.json             # Node.js dependencies and scripts
│   ├── public/
│   │   ├── index.html           # HTML entry point
│   │   └── Logo.png             # Application logo
│   └── src/
│       ├── App.js               # Main dashboard component (gauges, zones, timeline, sim)
│       ├── App.css              # Dashboard styles
│       ├── VenueTwin3D.jsx      # Canvas 2D venue renderer (zones, people, fire, routes)
│       ├── index.js             # React entry point
│       └── index.css            # Base styles
│
├── Multi Model/
│   ├── fusion_system/           # Adaptive fusion engine package
│   │   ├── __init__.py
│   │   ├── config.py            # All tunable parameters (thresholds, weights, paths)
│   │   ├── fusion_engine.py     # AdaptiveFusionEngine — weighted fusion + EMA + hysteresis
│   │   ├── vision_wrapper.py    # VisionWrapper — RT-DETR inference + Platt calibration
│   │   ├── enhanced_iot_model.py# EnhancedIoTModel — 8-feature Random Forest with CV
│   │   ├── calibration_pipeline.py # Platt scaling calibration for both modalities
│   │   ├── main_fire_detection.py  # Standalone multimodal fire detection system
│   │   ├── best.pt              # RT-DETR trained weights (fire/smoke detection)
│   │   ├── calibration_artifacts/  # Serialized calibrators and model files
│   │   │   ├── iot_rf_model.pkl
│   │   │   ├── iot_platt_calibrator.pkl
│   │   │   ├── vision_platt_calibrator.pkl
│   │   │   ├── iot_calibration_report.json
│   │   │   ├── vision_calibration_report.json
│   │   │   └── vision_calibration_data.json
│   │   ├── Multi_Hardware_v2/
│   │   │   └── Multi_Hardware_v2.ino   # ESP32 firmware v3 (Arduino)
│   │   └── requirements.txt
│   │
│   ├── Multi_Hardware/
│   │   ├── Multi_Hardware.ino   # ESP32 firmware (duplicate/alternate version)
│   │   └── SentryModel.h       # Exported RF model as C++ header for ESP32
│   │
│   ├── train_model.py           # Original model training script
│   ├── vision_verifier.py       # Original vision verifier (superseded by fusion_system)
│   ├── best.pt                  # RT-DETR weights (alternate location)
│   ├── rtdetr-l.pt              # RT-DETR-L base weights
│   ├── smoke_detection_iot.csv  # IoT training dataset (smoke detection)
│   ├── SentryModel.h            # Exported C++ RF model header
│   ├── D-Fire-2/                # D-Fire dataset directory
│   ├── Fire-Smoke-Indoor-5/     # Fire-Smoke-Indoor dataset directory
│   ├── D_Fire.py                # D-Fire dataset config (Roboflow)
│   ├── Indoor_Fire.py           # Indoor fire dataset config (Roboflow)
│   ├── Output_Comparision/      # Model comparison output artifacts
│   └── paper/                   # Research paper materials
│
└── Documentation/
    └── Major Project Review Template_final.pptx
```

## How It Works

### System Architecture

The system operates as four concurrent processes coordinated through the Digital Twin Hub:

1. **Digital Twin Hub** (`crowdpulse_server.py`, port 5000): Central Flask server that maintains the semantic venue model, receives PUT updates from the Vision Agent and ESP32, stores state in a circular history buffer, computes zone-level analytics, and serves the full twin state to the dashboard via REST API.

2. **Vision Agent** (`vision_agent.py`, port 5010): Perception engine running in the main thread for PyTorch safety. Captures webcam frames, runs YOLOv8n person detection and RT-DETR fire detection (in-process or via isolated subprocess on Windows), reads ESP32 sensor data (USB serial or HTTP polling), performs adaptive sensor fusion, computes the Crowd Stress Index, sends hardware commands to the ESP32, and pushes the merged state to the hub at 0.5 Hz.

3. **React Dashboard** (`dt-dashboard/`, port 3000): Polls the hub's `/api/twin/state` endpoint every 500ms and `/api/twin/history` every 5 seconds. Renders a canvas-based venue floor plan with animated occupant dots, fire effects, evacuation routes, zone overlays, sensor indicators, and HUD elements. Provides a simulation panel, event log, and timeline replay.

4. **ESP32 IoT Node**: Reads AHT10 (temperature/humidity) and MQ2 (gas) sensors at 1 Hz, runs on-device Random Forest inference, outputs structured serial data (`T:25.30,H:60.12,G:512,PRED:1,CONF:8`), serves a WiFi HTTP API, and pushes environment data to the hub every 2 seconds.

### Fire Detection Pipeline

```
Camera Frame ──→ RT-DETR (fire/smoke detection)
                   ├── Raw confidence scores
                   └── Platt scaling → Calibrated P(fire)_vision
                                              │
ESP32 Sensors ──→ IoT RF Model (8-feature Random Forest)
                   ├── predict_proba → P(fire)_iot
                   └── Tree vote entropy → IoT reliability
                                              │
                              ┌────────────────┘
                              ▼
                   Adaptive Fusion Engine
                   ├── Compute reliability metrics
                   ├── Derive adaptive weights (w_iot, w_vision)
                   ├── Weighted fusion: p = w_iot*p_iot + w_vision*p_vision
                   ├── Conflict detection and damping
                   ├── Temporal EMA smoothing (alpha=0.3)
                   └── Hysteresis state machine
                        ├── SAFE → ALARM: score ≥ 0.70
                        ├── ALARM → SAFE: score < 0.40
                        └── UNCERTAIN: conflict + mid-range score
```

When RT-DETR is unavailable, the system falls back to HSV color-space fire detection (orange-red and bright yellow masks with contour analysis).

### Crowd Stress Index (CSI)

The CSI is computed per-frame as the maximum of three components:
- **Visual stress**: Exit corridor person count multiplied by a stagnation multiplier (derived from flow velocity).
- **Audio stress**: Volume-based score from microphone input (thresholds at 8 and 15 normalized volume units).
- **Fire stress**: When fire is detected, CSI is set to at least 50, scaling up to 90 based on fusion confidence.

CSI drives the system state machine: STABLE (0-40), WARNING (41-75), CRITICAL (76-100 or fire detected).

## Installation

### Prerequisites

- Python 3.8 or later
- Node.js and npm
- A webcam (optional; the system operates without one)
- ESP32 with AHT10 and MQ2 sensors (optional; the system operates without IoT hardware)

### Backend Setup

```bash
cd CrowdPulse_Project
pip install -r requirements.txt
```

The `requirements.txt` specifies:
- `flask>=2.3.0`, `flask-cors>=4.0.0` (web server)
- `waitress>=2.1.0` (production WSGI server)
- `ultralytics>=8.0.0` (YOLOv8 + RT-DETR)
- `opencv-python>=4.8.0`, `torch>=2.0.0` (computer vision)
- `sounddevice>=0.4.6`, `scipy>=1.10.0` (audio)
- `numpy>=1.24.0`, `scikit-learn>=1.3.0`, `joblib>=1.3.0`, `pandas>=2.0.0` (fusion system)
- `requests>=2.31.0` (HTTP client, used by setup.py)

### Frontend Setup

```bash
cd dt-dashboard
npm install
```

### ESP32 Firmware (Optional)

Flash `Multi Model/fusion_system/Multi_Hardware_v2/Multi_Hardware_v2.ino` to an ESP32 using the Arduino IDE. Required libraries: `Adafruit_AHTX0`, `WiFi`, `WebServer`, `HTTPClient`, `ESPmDNS`. Update the WiFi credentials and Digital Twin Hub URL in the firmware before flashing.

### Calibration (Optional)

Run the calibration pipeline once before deploying the fusion system to generate Platt scaling calibrators:

```bash
cd "Multi Model/fusion_system"
python calibration_pipeline.py
```

This produces calibration artifacts in `calibration_artifacts/`.

## Usage

### Quick Start (Windows)

Run the batch launcher to start all components:

```bash
CrowdPulse_Start.bat
```

This sequentially launches:
1. Digital Twin Hub (port 5000)
2. Twin schema initializer
3. Vision Agent (port 5010)
4. React dashboard (port 3000)

### Manual Start

Start each component in a separate terminal:

```bash
# Terminal 1: Digital Twin Hub
python crowdpulse_server.py

# Terminal 2: Initialize twin schema
python setup.py

# Terminal 3: Vision Agent
python vision_agent.py

# Terminal 4: React Dashboard
cd dt-dashboard
npm start
```

Access the dashboard at `http://localhost:3000`.

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CROUDPULSE_TWIN_ID` | `org.campus:seminar_hall_01` | Digital Twin thing ID |
| `CROUDPULSE_ESP32_IP` | `10.243.83.181` | ESP32 IP address for HTTP polling |
| `CROUDPULSE_ESP32_URL` | (none) | Full ESP32 base URL override |
| `CROUDPULSE_ESP32_HOST` | (none) | ESP32 hostname override |
| `CROUDPULSE_FIRE_MODEL_PATH` | (none) | Path to RT-DETR weights file |
| `CROUDPULSE_RTDETR_MODEL` | (none) | Alternate RT-DETR model path |

### Fusion Engine Parameters (`Multi Model/fusion_system/config.py`)

| Parameter | Value | Description |
|---|---|---|
| `W_IOT_BASE` | 0.35 | Base weight for IoT modality |
| `W_VISION_BASE` | 0.65 | Base weight for Vision modality |
| `EMA_ALPHA` | 0.3 | Temporal smoothing factor |
| `ALARM_THRESHOLD_HIGH` | 0.70 | SAFE-to-ALARM transition threshold |
| `ALARM_THRESHOLD_LOW` | 0.40 | ALARM-to-SAFE transition threshold |
| `IOT_ONLY_THRESHOLD` | 0.80 | Fire threshold when only IoT is available |
| `VISION_ONLY_THRESHOLD` | 0.75 | Fire threshold when only vision is available |
| `CONFLICT_THRESHOLD` | 0.50 | Modality disagreement threshold |
| `CONFLICT_DAMPING` | 0.7 | Damping factor when conflict is detected |
| `RTDETR_CONF_THRESHOLD` | 0.25 | RT-DETR detection confidence threshold |
| `RTDETR_IMG_SIZE` | 416 | RT-DETR inference image size |

### Vision Agent Parameters (`vision_agent.py`)

| Parameter | Value | Description |
|---|---|---|
| `AI_INTERVAL` | 0.15s | AI inference loop interval |
| `FIRE_INFERENCE_INTERVAL` | 0.5s | Fire model inference interval |
| `CAMERA_STALE_SECONDS` | 2.5s | Camera frame staleness threshold |
| `IOT_STALE_SECONDS` | 12.0s | IoT data staleness threshold |
| `RTDETR_TIMEOUT_SECONDS` | 20.0s | RT-DETR worker response timeout |
| `RTDETR_MAX_FAILURES` | 5 | Max consecutive worker failures before disabling |
| `VISION_PORT` | 5010 | Vision Agent HTTP server port |

## API Documentation

### Digital Twin Hub (port 5000)

| Method | Route | Description |
|---|---|---|
| `PUT` | `/api/2/things/<thing_id>` | Update twin state (deep-merged). Called by vision_agent.py and ESP32. |
| `GET` | `/api/2/things/<thing_id>` | Get current state of a specific twin. |
| `GET` | `/api/2/things` | Get all registered twins. |
| `GET` | `/api/twin/state` | Full Digital Twin state with zone analytics, venue schema, and recent events. Primary endpoint for the dashboard. |
| `GET` | `/api/twin/history` | Time-series history (up to 240 snapshots). Supports `?limit=N` query parameter. |
| `GET` | `/api/twin/venue` | Static venue schema (zones, exits, sensors, evacuation routes). |
| `POST` | `/api/twin/simulate` | Run what-if fire scenario. Request body: `{"scenario": "fire_at_zone", "zone": "<zone_id>", "current_occupancy": <int>}`. Returns available routes, ETAs, congested zones, and interventions. |
| `GET` | `/api/twin/events` | Recent event log. Supports `?limit=N` (max 100). |

### Vision Agent (port 5010)

| Method | Route | Description |
|---|---|---|
| `GET` | `/video_feed` | MJPEG video stream with annotated detection overlays. |

### ESP32 IoT Node (port 80)

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Full JSON status (sensors, hardware state, uptime, WiFi RSSI). |
| `GET` | `/data` | Compact sensor JSON (temperature, humidity, gas_level). |
| `GET` | `/alert/fire` | Activate continuous fire alarm buzzer. |
| `GET` | `/alert/warning` | Activate intermittent warning beep. |
| `GET` | `/alert/off` | Silence buzzer. |
| `GET` | `/light/on` | Turn on indicator LED. |
| `GET` | `/light/off` | Turn off indicator LED. |

## Scripts

### Python

| Script | Purpose |
|---|---|
| `crowdpulse_server.py` | Start the Digital Twin Hub on port 5000. |
| `vision_agent.py` | Start the Vision/Fusion Perception Engine on port 5010. |
| `setup.py` | Initialize the twin schema with default feature properties. |
| `rtdetr_fire_worker.py` | RT-DETR isolated inference worker (launched automatically by vision_agent.py on Windows). |
| `Multi Model/fusion_system/calibration_pipeline.py` | Run Platt scaling calibration for both IoT and Vision models. |
| `Multi Model/fusion_system/main_fire_detection.py` | Standalone multimodal fire detection system (ESP32 serial + RT-DETR + fusion). |
| `Multi Model/fusion_system/enhanced_iot_model.py` | Train the 8-feature Random Forest IoT model with cross-validation. |

### npm (dt-dashboard)

| Script | Purpose |
|---|---|
| `npm start` | Start the React development server on port 3000. |
| `npm run build` | Build the production bundle. |
| `npm test` | Run tests. |

### Batch

| Script | Purpose |
|---|---|
| `CrowdPulse_Start.bat` | Launch all four system components in sequence (Windows). |

## Dependencies

### Python (key libraries)

| Library | Role |
|---|---|
| `flask` / `flask-cors` | REST API server for Digital Twin Hub and Vision Agent |
| `ultralytics` | YOLOv8n person detection and RT-DETR fire detection inference |
| `opencv-python` | Camera capture, image processing, CLAHE enhancement, MJPEG encoding |
| `torch` | PyTorch runtime for neural network inference |
| `sounddevice` | Real-time microphone audio capture |
| `numpy` | Numerical operations throughout the pipeline |
| `scikit-learn` | Random Forest classifier, Platt scaling calibration, cross-validation |
| `joblib` | Model serialization (calibrators, RF models) |
| `pandas` | IoT dataset loading and feature management |
| `waitress` | Production-grade WSGI server (replaces Flask dev server) |
| `pyserial` | USB serial communication with ESP32 |

### Frontend (key libraries)

| Library | Role |
|---|---|
| `react` / `react-dom` (v19) | UI framework |
| `axios` | HTTP client (available, though `fetch` is used directly in App.js) |

## Known Limitations

- **Thread Safety on Windows/Python 3.13**: PyTorch native code (MKL/BLAS) triggers access-violation crashes when called from daemon threads. The architecture forces all PyTorch inference into the main OS thread to avoid this. This is documented extensively in `vision_agent.py`.
- **AVX-512 Incompatibility**: The system explicitly restricts CPU instructions to AVX2 (`ATEN_CPU_CAPABILITY=avx2`, `MKL_ENABLE_INSTRUCTIONS=AVX2`) because the target hardware (i5-1335U) does not support AVX-512, which PyTorch's MKL kernels may attempt.
- **RT-DETR Subprocess Mode**: On Windows (`os.name == 'nt'`), RT-DETR runs in an isolated subprocess to contain potential native crashes. This adds latency due to file-based frame passing (temporary JPEG files).
- **urllib3 Avoided**: HTTP communication from the Vision Agent uses Python stdlib `http.client` instead of the `requests` library because `urllib3.create_connection` causes access-violation crashes on Python 3.13 Windows when running concurrently with PyTorch.
- **Single-threaded PyTorch**: Thread counts for OMP, MKL, OpenBLAS, and torch are all forced to 1 to avoid SIMD instruction conflicts.
- **Vision Calibration Data**: The vision Platt calibrator uses synthetic calibration data by default. The code notes that real RT-DETR confidence scores on labeled images should be collected for production use.
- **No Authentication**: All API endpoints (Hub, Vision Agent, ESP32) are unauthenticated and CORS-open (`*`).
- **Hardcoded Venue Schema**: The venue model (zone bounds, exits, routes) is defined in-code in `crowdpulse_server.py` and mirrored in `VenueTwin3D.jsx`. There is no dynamic venue configuration mechanism.
- **Camera Backend Cycling**: If the camera produces black frames for 20 consecutive reads, the Vision Agent cycles through OpenCV backends (DirectShow, MediaFoundation, Auto), which causes brief reconnection delays.

## License

Not explicitly defined in the codebase.
