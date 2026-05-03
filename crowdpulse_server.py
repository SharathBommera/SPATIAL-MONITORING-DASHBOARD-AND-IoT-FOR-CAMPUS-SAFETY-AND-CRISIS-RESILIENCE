# =============================================================================
# FILE: crowdpulse_server.py
# PURPOSE: CrowdPulse Digital Twin Hub — True Digital Twin Architecture
#
# What makes this a REAL Digital Twin (not just a dashboard):
#   1. SEMANTIC VENUE MODEL  — The physical space is modelled with named zones,
#      capacities, exits, sensors, and evacuation routes.
#   2. STATE HISTORY         — Circular buffer of the last 2 minutes of state
#      snapshots. The React UI can replay any moment in the past.
#   3. ZONE-LEVEL ANALYTICS  — Each zone owns its occupancy, pressure score,
#      and alarm state, not just the hall as a whole.
#   4. SIMULATION ENGINE     — POST /api/twin/simulate to run "what-if"
#      scenarios (fire at zone X → optimal evacuation path + ETA).
#   5. BIDIRECTIONAL SYNC    — Physical sensors write IN via PUT; digital
#      commands go OUT to actuators via the vision_agent.
# =============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import deque
import time
import copy
import logging
import math

# ── Silence noisy werkzeug logs ───────────────────────────────────────────────
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# =============================================================================
# SEMANTIC VENUE MODEL
# This is the "blueprint" of the physical space.  Every zone, exit, sensor
# and evacuation route is a first-class citizen in the digital twin.
# =============================================================================
VENUE_SCHEMA = {
    "id": "venue:seminar_hall_01",
    "name": "Engineering Seminar Hall A",
    "type": "SeminarHall",
    "institution": "CrowdPulse Campus",
    "total_capacity": 200,
    # World-unit dimensions fed to the Three.js renderer (metres)
    "dimensions": {"width": 10.0, "length": 8.0, "height": 3.5},
    # Zones — each zone maps to a coloured region on the 3D floor plan.
    # bounds: normalised 0-1 coords relative to the hall footprint.
    "zones": {
        "entry_lobby": {
            "name": "Entry Lobby",
            "capacity": 30,
            "bounds": {"x": 0.00, "y": 0.00, "w": 0.25, "h": 0.30},
            "color": "#9C27B0",
            "exits": ["exit_emergency"],
            "evacuation_priority": 2
        },
        "stage_area": {
            "name": "Stage / Podium",
            "capacity": 20,
            "bounds": {"x": 0.25, "y": 0.00, "w": 0.50, "h": 0.30},
            "color": "#2196F3",
            "exits": ["exit_main"],
            "evacuation_priority": 1
        },
        "main_seating": {
            "name": "Main Seating",
            "capacity": 120,
            "bounds": {"x": 0.00, "y": 0.30, "w": 0.70, "h": 0.70},
            "color": "#4CAF50",
            "exits": ["exit_main", "exit_emergency"],
            "evacuation_priority": 3
        },
        "exit_corridor": {
            "name": "Exit Corridor",
            "capacity": 30,
            "bounds": {"x": 0.70, "y": 0.00, "w": 0.30, "h": 1.00},
            "color": "#FF9800",
            "exits": ["exit_main"],
            "evacuation_priority": 4
        }
    },
    # Physical exits — rendered as gaps in walls in the 3D model.
    "exits": [
        {"id": "exit_main",      "name": "Main Exit",       "x": 1.00, "y": 0.50, "status": "OPEN", "width": 0.12},
        {"id": "exit_emergency", "name": "Emergency Exit",  "x": 0.00, "y": 0.50, "status": "OPEN", "width": 0.08},
        {"id": "exit_back",      "name": "Back Exit",       "x": 0.50, "y": 1.00, "status": "OPEN", "width": 0.10}
    ],
    # Physical sensor locations — rendered as device icons in the 3D model.
    "sensors": [
        {"id": "esp32_main",    "type": "IoT_Composite",   "name": "ESP32 Sensor Node",  "x": 0.50, "y": 0.15},
        {"id": "camera_front",  "type": "Vision_Camera",   "name": "AI Vision Camera",   "x": 0.50, "y": 0.02},
        {"id": "mic_main",      "type": "Microphone",      "name": "Panic Microphone",   "x": 0.50, "y": 0.50}
    ],
    # Pre-computed evacuation routes — animated in the 3D model on alarm.
    "evacuation_routes": [
        {
            "id": "route_primary",
            "name": "Primary — Main Exit",
            "path": [
                {"x": 0.35, "y": 0.50},
                {"x": 0.70, "y": 0.50},
                {"x": 1.00, "y": 0.50}
            ],
            "exit": "exit_main",
            "capacity_per_minute": 120,
            "priority": 1
        },
        {
            "id": "route_secondary",
            "name": "Emergency — Left Exit",
            "path": [
                {"x": 0.35, "y": 0.50},
                {"x": 0.10, "y": 0.50},
                {"x": 0.00, "y": 0.50}
            ],
            "exit": "exit_emergency",
            "capacity_per_minute": 80,
            "priority": 2
        },
        {
            "id": "route_tertiary",
            "name": "Overflow — Back Exit",
            "path": [
                {"x": 0.35, "y": 0.65},
                {"x": 0.50, "y": 0.80},
                {"x": 0.50, "y": 1.00}
            ],
            "exit": "exit_back",
            "capacity_per_minute": 60,
            "priority": 3
        }
    ]
}

# =============================================================================
# RUNTIME STATE
# =============================================================================
digital_twins = {}                              # live twin data (merged)
state_history = deque(maxlen=240)               # 2-min ring buffer at 0.5 Hz
event_log = deque(maxlen=100)                   # last 100 significant events

# =============================================================================
# HELPERS
# =============================================================================

def deep_merge(source, destination):
    """Recursively merge source dict into destination dict in-place."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.get(key)
            if not isinstance(node, dict):
                destination[key] = {}
                node = destination[key]
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination


def _compute_zone_analytics(twin_data):
    """
    Derive per-zone occupancy, density and alarm state from person locations.
    Returns a dict keyed by zone_id.
    """
    locations = []
    try:
        locations = twin_data["features"]["occupancy"]["properties"]["locations"]
    except (KeyError, TypeError):
        pass
    zone_counts = {}
    try:
        zone_counts = twin_data["features"]["occupancy"]["properties"]["zone_counts"]
    except (KeyError, TypeError):
        pass

    pressure = 0
    try:
        pressure = twin_data["features"]["crowd_analytics"]["properties"]["pressure_score"]
    except (KeyError, TypeError):
        pass

    zone_analytics = {}
    for zone_id, zone in VENUE_SCHEMA["zones"].items():
        b = zone["bounds"]
        if locations:
            count = 0
            for loc in locations:
                lx = loc.get("x", -1)
                ly = loc.get("y", -1)
                if b["x"] <= lx <= b["x"] + b["w"] and b["y"] <= ly <= b["y"] + b["h"]:
                    count += 1
        else:
            count = int(zone_counts.get(zone_id, 0) or 0)
        cap = zone["capacity"]
        density_pct = round((count / cap) * 100, 1) if cap > 0 else 0
        # Zone pressure: scale global pressure by local density
        zone_pressure = min(100, round(pressure * (density_pct / 100 + 0.1), 1))
        if density_pct > 80 or zone_pressure > 75:
            zone_state = "CRITICAL"
        elif density_pct > 50 or zone_pressure > 40:
            zone_state = "WARNING"
        else:
            zone_state = "STABLE"
        zone_analytics[zone_id] = {
            "occupancy": count,
            "capacity": cap,
            "density_pct": density_pct,
            "pressure": zone_pressure,
            "state": zone_state
        }
    return zone_analytics


def _log_event(event_type, message, severity="INFO"):
    """Append a structured event to the event log."""
    event_log.append({
        "timestamp": round(time.time(), 3),
        "type": event_type,
        "message": message,
        "severity": severity
    })


def _safe_feature_prop(twin_data, *path, default=None):
    """Safely read a deeply nested feature property."""
    cur = twin_data
    try:
        for key in path:
            cur = cur[key]
        return cur
    except (KeyError, TypeError):
        return default


def _route_path_zones(route):
    """Resolve which venue zones a route path passes through."""
    path_zones = set()
    for pt in route.get("path", []):
        for zone_id, zone in VENUE_SCHEMA["zones"].items():
            b = zone["bounds"]
            if b["x"] <= pt["x"] <= b["x"] + b["w"] and b["y"] <= pt["y"] <= b["y"] + b["h"]:
                path_zones.add(zone_id)
    return path_zones


def _route_exit_meta(exit_id):
    """Return exit metadata by id."""
    return next((ex for ex in VENUE_SCHEMA["exits"] if ex["id"] == exit_id), None)


def _add_to_history(twin_id, twin_data):
    """Snapshot current state into the ring buffer."""
    try:
        snapshot = {
            "timestamp": round(time.time(), 3),
            "twin_id": twin_id,
            "state": copy.deepcopy(twin_data),
            "zone_analytics": _compute_zone_analytics(twin_data)
        }
        state_history.append(snapshot)
    except Exception:
        pass  # never crash on history write


# =============================================================================
# API ROUTES — Backward-Compatible (server.py) + New Digital Twin Endpoints
# =============================================================================

@app.route('/api/2/things/<thing_id>', methods=['PUT'])
def update_thing(thing_id):
    """
    Standard Digital Twin update endpoint.
    Called by vision_agent.py every 0.5 s and by the ESP32 every 1 s.
    """
    new_data = request.get_json(silent=True) or {}
    previous = copy.deepcopy(digital_twins.get(thing_id, {}))

    if thing_id not in digital_twins:
        digital_twins[thing_id] = new_data
    else:
        deep_merge(new_data, digital_twins[thing_id])

    prev_pred = _safe_feature_prop(
        previous, "features", "crowd_analytics", "properties", "prediction", default=None
    )
    pred = _safe_feature_prop(
        digital_twins[thing_id], "features", "crowd_analytics", "properties", "prediction", default=None
    )
    if pred != prev_pred:
        if pred == "CRITICAL":
            _log_event("CROWD_CRITICAL", "Crowd Stress Index exceeded critical threshold", "CRITICAL")
        elif pred == "WARNING":
            _log_event("CROWD_WARNING", "Crowd stress rising and a nudge was initiated", "WARNING")
        elif pred == "STABLE" and prev_pred in {"WARNING", "CRITICAL"}:
            _log_event("CROWD_STABLE", "Crowd conditions returned to stable", "INFO")

    prev_fire = _safe_feature_prop(
        previous, "features", "safety_state", "properties", "is_fire_detected", default=False
    )
    fire = _safe_feature_prop(
        digital_twins[thing_id], "features", "safety_state", "properties", "is_fire_detected", default=False
    )
    if fire and not prev_fire:
        _log_event("FIRE_DETECTED", "Multi-modal fire detection triggered", "CRITICAL")
    elif prev_fire and not fire:
        _log_event("FIRE_CLEARED", "Fire condition cleared", "INFO")

    _add_to_history(thing_id, digital_twins[thing_id])

    return jsonify(digital_twins[thing_id]), 201


@app.route('/api/2/things/<thing_id>', methods=['GET'])
def get_thing(thing_id):
    """Get current state of a specific twin."""
    return jsonify(digital_twins.get(thing_id, {})), 200


@app.route('/api/2/things', methods=['GET'])
def get_all_things():
    """Get all twins (for multi-hall expansion)."""
    return jsonify(list(digital_twins.values())), 200


@app.route('/api/twin/state', methods=['GET'])
def get_twin_state():
    """
    Full Digital Twin state endpoint.
    Returns: live twin data + zone analytics + latest event log + system info.
    This is the primary endpoint consumed by the React 3D dashboard.
    """
    twin_id = "org.campus:seminar_hall_01"
    twin_data = digital_twins.get(twin_id, {})
    zone_analytics = _compute_zone_analytics(twin_data)

    return jsonify({
        "twin_id": twin_id,
        "timestamp": round(time.time(), 3),
        "live_state": twin_data,
        "zone_analytics": zone_analytics,
        "venue_schema": VENUE_SCHEMA,   # front-end caches this
        "recent_events": list(event_log)[-10:],
        "history_length": len(state_history)
    }), 200


@app.route('/api/twin/history', methods=['GET'])
def get_twin_history():
    """
    Time-series history endpoint — feeds the Timeline replay slider.
    Optional ?limit=N query param (default 120 = last 60 s at 0.5 Hz).
    """
    limit = min(int(request.args.get("limit", 120)), 240)
    snapshots = list(state_history)[-limit:]
    # Compact format: only the fields the timeline needs
    compact = []
    for s in snapshots:
        try:
            props = s["state"]["features"]["crowd_analytics"]["properties"]
            fire  = s["state"]["features"]["safety_state"]["properties"]["is_fire_detected"]
            compact.append({
                "t": s["timestamp"],
                "timestamp": s["timestamp"],
                "csi": props.get("pressure_score", 0),
                "flow": props.get("flow_rate", 0),
                "pred": props.get("prediction", "STABLE"),
                "fire": fire,
                "zone_analytics": s.get("zone_analytics", {}),
                "state": s.get("state", {})
            })
        except (KeyError, TypeError):
            compact.append({"t": s["timestamp"], "timestamp": s["timestamp"], "csi": 0, "flow": 0,
                            "pred": "STABLE", "fire": False, "zone_analytics": {},
                            "state": s.get("state", {})})
    return jsonify({
        "count": len(compact),
        "history": compact
    }), 200


@app.route('/api/twin/venue', methods=['GET'])
def get_venue_schema():
    """Static venue schema — React fetches once and caches for 3D rendering."""
    return jsonify(VENUE_SCHEMA), 200


@app.route('/api/twin/simulate', methods=['POST'])
def simulate_scenario():
    """
    Simulation Engine — run what-if scenarios against the digital twin.

    Request body:
        {
            "scenario": "fire_at_zone",
            "zone": "exit_corridor",
            "current_occupancy": 85
        }

    Returns:
        - Recommended evacuation routes (sorted by priority)
        - Estimated evacuation time (ETA) per route
        - Affected zones
        - Recommended interventions
    """
    body = request.get_json(silent=True) or {}
    scenario = body.get("scenario", "fire_at_zone")
    affected_zone = body.get("zone", "exit_corridor")
    total_people = body.get("current_occupancy", 50)

    # --- Simulate evacuation routing ---
    # Disable routes that pass through the fire zone
    available_routes = []
    for route in VENUE_SCHEMA["evacuation_routes"]:
        path_zones = _route_path_zones(route)
        if affected_zone not in path_zones:
            available_routes.append(route)

    # ETA calculation: people / (capacity_per_minute / 60)
    eta_results = []
    remaining = total_people
    for route in sorted(available_routes, key=lambda r: r["priority"]):
        cap = route["capacity_per_minute"]
        people_via_route = min(remaining, cap)
        eta_seconds = round((people_via_route / cap) * 60, 1) if cap > 0 else 999
        eta_results.append({
            "route_id": route["id"],
            "route_name": route["name"],
            "exit": route["exit"],
            "people": people_via_route,
            "eta_seconds": eta_seconds
        })
        remaining -= people_via_route
        if remaining <= 0:
            break

    # Compute worst-case ETA
    worst_eta = max((r["eta_seconds"] for r in eta_results), default=0)

    # Identify adjacent zones that may become congested
    congested_zones = []
    for z_id, z in VENUE_SCHEMA["zones"].items():
        if z_id == affected_zone:
            congested_zones.append(z_id)
            continue
        for exit_id in z.get("exits", []):
            exit_meta = _route_exit_meta(exit_id)
            if exit_meta is None:
                continue
            for route in VENUE_SCHEMA["evacuation_routes"]:
                if route["exit"] != exit_id:
                    continue
                if affected_zone in _route_path_zones(route):
                    congested_zones.append(z_id)
                    break
            if z_id in congested_zones:
                break

    interventions = [
        f"Activate {route['name']} - direct crowd to {route['exit']}"
        for route in available_routes
    ]
    if len(available_routes) < 2:
        interventions.append("CRITICAL: Limited evacuation routes. Activate all exits immediately.")
    interventions.append("Disable Smart Podium Eco-Mode. Trigger auditory alarms.")
    interventions.append(f"Estimated clearance time: {worst_eta} seconds for {total_people} occupants.")

    _log_event("SIMULATION_RUN", f"Simulated '{scenario}' at zone '{affected_zone}'", "INFO")

    return jsonify({
        "scenario": scenario,
        "affected_zone": affected_zone,
        "available_routes": len(available_routes),
        "total_routes": len(VENUE_SCHEMA["evacuation_routes"]),
        "eta_results": eta_results,
        "worst_case_eta_seconds": worst_eta,
        "congested_zones": congested_zones,
        "interventions": interventions,
        "simulation_timestamp": round(time.time(), 3)
    }), 200


@app.route('/api/twin/events', methods=['GET'])
def get_events():
    """Recent event log for the dashboard activity feed."""
    limit = min(int(request.args.get("limit", 20)), 100)
    return jsonify({
        "events": list(event_log)[-limit:]
    }), 200


# =============================================================================
# STARTUP
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  CROWDPULSE DIGITAL TWIN HUB")
    print("  ── True Digital Twin Architecture ──")
    print("  Port : 5000")
    print("  Endpoints:")
    print("    PUT/GET /api/2/things/<id>   (backward compat)")
    print("    GET     /api/twin/state      (full DT state)")
    print("    GET     /api/twin/history    (time-series replay)")
    print("    GET     /api/twin/venue      (venue schema)")
    print("    POST    /api/twin/simulate   (what-if simulation)")
    print("    GET     /api/twin/events     (event log)")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
