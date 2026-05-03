# FILE: setup.py
# PURPOSE: Setup Twin with PREDICTIVE ANALYTICS SCHEMA
import requests, json

HUB_URL = "http://localhost:5000/api/2/things"

seminar_hall = {
    "policyId": "org.campus:policy_01",
    "thingId": "org.campus:seminar_hall_01",
    "attributes": {
        "location": "Main Block, Floor 2",
        "type": "Seminar Hall"
    },
    "features": {
        "environment": {
            "properties": {
                "temperature": 0.0,
                "humidity": 0.0,
                "gas_ppm": 0,
                "status": "active"
            }
        },
        "occupancy": {
            "properties": {
                "current": 0,
                "limit": 100,
                "locations": []
            }
        },
        "safety_state": {
            "properties": {
                "is_fire_detected": False,
                "fire_location": None,
                "status": "safe"
            }
        },
        "automation": {
            "properties": {
                "podium_light": False
            }
        },
        # --- NEW: CROWD PREDICTION ENGINE ---
        "crowd_analytics": {
            "properties": {
                "pressure_score": 0,    # 0 to 100
                "flow_rate": 0.0,       # People entering exit zone per sec
                "prediction": "STABLE"  # STABLE, SURGE, CRITICAL
            }
        }
    }
}

print(f"Connecting to {HUB_URL}...")
try:
    requests.put(f"{HUB_URL}/{seminar_hall['thingId']}", json=seminar_hall)
    print("✅ SEMINAR HALL TWIN UPDATED WITH PREDICTIVE ENGINE!")
except Exception as e:
    print(f"❌ Error: {e}")