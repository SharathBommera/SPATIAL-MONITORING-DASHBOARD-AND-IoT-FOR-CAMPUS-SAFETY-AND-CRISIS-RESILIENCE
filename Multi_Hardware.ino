/*
================================================================================
ESP32 FIRMWARE v3 — CrowdPulse Digital Twin IoT Node
================================================================================
PURPOSE:
  Sensor acquisition (AHT10 + MQ2) + on-device ML pre-screening (SentryModel.h)
  + WiFi HTTP server for bidirectional Digital Twin communication.

ENDPOINTS SERVED:
  GET  /              → Full JSON status (sensors + hardware state + uptime)
  GET  /data          → Compact sensor JSON (temperature, humidity, gas_level)
  GET  /alert/fire    → Activate continuous fire alarm buzzer
  GET  /alert/warning → Activate intermittent CSI warning beep
  GET  /alert/off     → Silence buzzer
  GET  /light/on      → Turn on indicator LED (podium light proxy)
  GET  /light/off     → Turn off indicator LED

OUTBOUND:
  Every PUSH_INTERVAL_MS the node PUTs its sensor data to the Digital Twin Hub
  so the dashboard shows live IoT telemetry even without vision_agent running.

WIRE MAP:
  AHT10  → SDA=GPIO21, SCL=GPIO22 (I2C)
  MQ2    → Analog GPIO34 (ADC1_CH6)
  BUZZER → GPIO15 (active-HIGH buzzer or relay module)
  LED    → GPIO2  (ESP32 onboard LED, podium-light proxy)
  WAKE   → GPIO5  (optional, interrupt to Raspberry Pi)

SERIAL OUTPUT (backward compat with Pi):
  T:25.30,H:60.12,G:512,PRED:1,CONF:8
================================================================================
*/

#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <ESPmDNS.h>
#include <Adafruit_AHTX0.h>
#include "SentryModel.h"
#include <esp_task_wdt.h>

// =============================================================================
// *** USER CONFIG — UPDATE THESE BEFORE FLASHING ***
// =============================================================================
const char* WIFI_SSID      = "sharath";       // ← Your WiFi network name
const char* WIFI_PASSWORD   = "sharath66";    // ← Your WiFi password

// Digital Twin Hub address (laptop running crowdpulse_server.py)
// After WiFi connects, update vision_agent.py ESP32_IP to the IP printed on Serial
const char* DT_HUB_URL = "http://192.168.0.4:5000/api/2/things/org.campus:seminar_hall_01";
// ↑ Change 192.168.0.4 to your laptop's local IP (run `ipconfig` on Windows)

// =============================================================================
// PIN DEFINITIONS
// =============================================================================
#define MQ2_PIN        34
#define WAKE_UP_PIN    5
#define LED_PIN        2
#define BUZZER_PIN     15

// =============================================================================
// TIMING CONFIGURATION
// =============================================================================
#define SERIAL_BAUD        115200
#define SAMPLE_RATE_MS     1000     // Sensor read interval (1 Hz)
#define MQ2_WARMUP_MS      20000    // MQ2 preheat time
#define PUSH_INTERVAL_MS   2000     // Push to DT Hub every 2s
#define WAKE_HOLD_MS       1000
#define WAKE_COOLDOWN_MS   5000
#define WDT_TIMEOUT_S      30
#define ADC_SAMPLES        5
#define WIFI_RETRY_MS      10000    // WiFi reconnect interval
#define WARNING_BEEP_MS    300      // Buzzer on-time during warning beep
#define WARNING_PAUSE_MS   700      // Buzzer off-time during warning beep

// =============================================================================
// ALERT MODES
// =============================================================================
enum AlertMode {
  ALERT_OFF     = 0,
  ALERT_WARNING = 1,   // Intermittent beep (CSI > 75)
  ALERT_FIRE    = 2    // Continuous siren (fire detected)
};

// =============================================================================
// GLOBAL STATE
// =============================================================================
Adafruit_AHTX0 aht;
Eloquent::ML::Port::RandomForest classifier;
WebServer server(80);

// Sensor readings (updated every SAMPLE_RATE_MS)
float g_temperature  = 0.0;
float g_humidity     = 0.0;
float g_gas_raw      = 0.0;
int   g_fire_pred    = 0;
int   g_fire_votes   = 0;

// Hardware control state
AlertMode g_alert_mode   = ALERT_OFF;
bool      g_light_active = false;
bool      g_sensor_ok    = true;

// Timing state
unsigned long g_last_push     = 0;
unsigned long g_last_wake     = 0;
unsigned long g_last_beep     = 0;
bool          g_beep_state    = false;   // Current buzzer toggle for warning mode
unsigned long g_last_wifi_try = 0;

// Humidity rate tracking
float         g_prev_humidity    = 50.0;
float         g_humidity_rate    = 0.0;
unsigned long g_last_humidity_ts = 0;

// =============================================================================
// WiFi CONNECT
// =============================================================================
void connectWiFi() {
  Serial.print("[WiFi] Connecting to ");
  Serial.print(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(" OK");
    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.println("[WiFi] *** Update ESP32_IP in vision_agent.py to this IP ***");

    if (MDNS.begin("esp32-crowdpulse")) {
      Serial.println("[mDNS] Reachable at esp32-crowdpulse.local");
      MDNS.addService("http", "tcp", 80);
    }
  } else {
    Serial.println(" FAILED");
    Serial.println("[WiFi] Continuing in serial-only mode. Will retry periodically.");
  }

  digitalWrite(LED_PIN, LOW);
}

// =============================================================================
// CORS HELPER
// =============================================================================
void sendJSON(int code, const String& json) {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  server.send(code, "application/json", json);
}

// =============================================================================
// HTTP HANDLERS
// =============================================================================
void handleRoot() {
  String json = "{";
  json += "\"device\":\"CrowdPulse ESP32 IoT Node v3\",";
  json += "\"temperature\":" + String(g_temperature, 2) + ",";
  json += "\"humidity\":" + String(g_humidity, 2) + ",";
  json += "\"gas_level\":" + String((int)g_gas_raw) + ",";
  json += "\"fire_prediction\":" + String(g_fire_pred) + ",";
  json += "\"fire_confidence\":" + String(g_fire_votes) + ",";
  json += "\"alert_mode\":\"" + String(g_alert_mode == ALERT_FIRE ? "FIRE" : g_alert_mode == ALERT_WARNING ? "WARNING" : "OFF") + "\",";
  json += "\"light_active\":" + String(g_light_active ? "true" : "false") + ",";
  json += "\"humidity_rate\":" + String(g_humidity_rate, 3) + ",";
  json += "\"sensor_ok\":" + String(g_sensor_ok ? "true" : "false") + ",";
  json += "\"wifi_rssi\":" + String(WiFi.RSSI()) + ",";
  json += "\"uptime_ms\":" + String(millis());
  json += "}";
  sendJSON(200, json);
}

void handleData() {
  // Compact sensor JSON — polled by vision_agent.py every few seconds
  String json = "{";
  json += "\"temperature\":" + String(g_temperature, 2) + ",";
  json += "\"humidity\":" + String(g_humidity, 2) + ",";
  json += "\"gas_level\":" + String((int)g_gas_raw);
  json += "}";
  sendJSON(200, json);
}

void handleAlertFire() {
  g_alert_mode = ALERT_FIRE;
  digitalWrite(BUZZER_PIN, HIGH);      // Continuous ON for fire
  sendJSON(200, "{\"alert\":\"FIRE\",\"buzzer\":\"ON_CONTINUOUS\"}");
  Serial.println("[CMD] ALERT MODE: FIRE (continuous buzzer)");
}

void handleAlertWarning() {
  g_alert_mode = ALERT_WARNING;
  g_last_beep  = millis();
  g_beep_state = true;
  digitalWrite(BUZZER_PIN, HIGH);      // Start first beep
  sendJSON(200, "{\"alert\":\"WARNING\",\"buzzer\":\"INTERMITTENT\"}");
  Serial.println("[CMD] ALERT MODE: WARNING (intermittent beep)");
}

void handleAlertOff() {
  g_alert_mode = ALERT_OFF;
  g_beep_state = false;
  digitalWrite(BUZZER_PIN, LOW);
  sendJSON(200, "{\"alert\":\"OFF\",\"buzzer\":\"OFF\"}");
  Serial.println("[CMD] ALERT MODE: OFF");
}

// Legacy endpoints — map to new modes for backward compat
void handleAlertOnLegacy() {
  handleAlertFire();   // /alert/on → same as /alert/fire
}

void handleLightOn() {
  g_light_active = true;
  digitalWrite(LED_PIN, HIGH);
  sendJSON(200, "{\"light\":\"ON\"}");
  Serial.println("[CMD] LED ON");
}

void handleLightOff() {
  g_light_active = false;
  digitalWrite(LED_PIN, LOW);
  sendJSON(200, "{\"light\":\"OFF\"}");
  Serial.println("[CMD] LED OFF");
}

void handleNotFound() {
  sendJSON(404, "{\"error\":\"endpoint not found\"}");
}

void handleOptions() {
  // CORS preflight
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, PUT, POST, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
  server.send(204);
}

// =============================================================================
// SENSOR HELPERS
// =============================================================================
float readMQ2Averaged() {
  long sum = 0;
  for (int i = 0; i < ADC_SAMPLES; i++) {
    sum += analogRead(MQ2_PIN);
    delayMicroseconds(200);
  }
  return (float)(sum / ADC_SAMPLES);
}

int getFireVoteCount(float* features) {
  int prediction = classifier.predict(features);
  return prediction == 1 ? 10 : 0;
}

// =============================================================================
// PUSH DATA TO DIGITAL TWIN HUB
// =============================================================================
void pushToHub() {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(DT_HUB_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(2000);

  // Build the environment feature payload matching the DT schema
  String payload = "{\"features\":{\"environment\":{\"properties\":{";
  payload += "\"temperature\":" + String(g_temperature, 2) + ",";
  payload += "\"humidity\":" + String(g_humidity, 2) + ",";
  payload += "\"gas_level\":" + String((int)g_gas_raw);
  payload += "}}}}";

  int code = http.PUT(payload);
  if (code > 0) {
    // Success — hub received our data
  } else {
    // Hub not reachable — that's fine, vision_agent polls us directly too
  }
  http.end();
}

// =============================================================================
// BUZZER STATE MACHINE (non-blocking)
// =============================================================================
void updateBuzzer() {
  unsigned long now = millis();

  switch (g_alert_mode) {
    case ALERT_FIRE:
      // Continuous buzzer — already HIGH from handleAlertFire()
      // Just ensure it stays on
      digitalWrite(BUZZER_PIN, HIGH);
      break;

    case ALERT_WARNING:
      // Intermittent beep pattern: ON for WARNING_BEEP_MS, OFF for WARNING_PAUSE_MS
      if (g_beep_state) {
        if (now - g_last_beep >= WARNING_BEEP_MS) {
          digitalWrite(BUZZER_PIN, LOW);
          g_beep_state = false;
          g_last_beep = now;
        }
      } else {
        if (now - g_last_beep >= WARNING_PAUSE_MS) {
          digitalWrite(BUZZER_PIN, HIGH);
          g_beep_state = true;
          g_last_beep = now;
        }
      }
      break;

    case ALERT_OFF:
    default:
      digitalWrite(BUZZER_PIN, LOW);
      break;
  }
}

// =============================================================================
// SETUP
// =============================================================================
void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(100);

  Serial.println();
  Serial.println("================================================");
  Serial.println("  CrowdPulse ESP32 IoT Node v3");
  Serial.println("  Digital Twin & IoT for Campus Safety");
  Serial.println("================================================");

  // Configure pins
  pinMode(MQ2_PIN, INPUT);
  pinMode(WAKE_UP_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(WAKE_UP_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  // ESP32 ADC: 12-bit resolution, full 0-3.3V range
  analogSetAttenuation(ADC_11db);
  analogSetWidth(12);

  // Initialize AHT10
  Wire.begin(21, 22);
  if (!aht.begin()) {
    Serial.println("[SENSOR] AHT10 NOT FOUND — running MQ2 only");
    g_sensor_ok = false;
  } else {
    Serial.println("[SENSOR] AHT10 OK");
  }

  // MQ2 preheat (LED on during warmup as indicator)
  Serial.println("[SENSOR] MQ2 warming up (20s)...");
  digitalWrite(LED_PIN, HIGH);
  delay(MQ2_WARMUP_MS);
  digitalWrite(LED_PIN, LOW);
  Serial.println("[SENSOR] MQ2 OK");

  // Quick buzzer test — one short beep to confirm wiring
  Serial.println("[BUZZER] Testing...");
  digitalWrite(BUZZER_PIN, HIGH);
  delay(150);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println("[BUZZER] OK");

  // Connect WiFi
  connectWiFi();

  // Register HTTP endpoints
  server.on("/",              HTTP_GET,     handleRoot);
  server.on("/data",          HTTP_GET,     handleData);
  server.on("/alert/fire",    HTTP_GET,     handleAlertFire);
  server.on("/alert/warning", HTTP_GET,     handleAlertWarning);
  server.on("/alert/off",     HTTP_GET,     handleAlertOff);
  server.on("/alert/on",      HTTP_GET,     handleAlertOnLegacy);   // backward compat
  server.on("/light/on",      HTTP_GET,     handleLightOn);
  server.on("/light/off",     HTTP_GET,     handleLightOff);
  // CORS preflight for all paths
  server.on("/",              HTTP_OPTIONS, handleOptions);
  server.on("/data",          HTTP_OPTIONS, handleOptions);
  server.on("/alert/fire",    HTTP_OPTIONS, handleOptions);
  server.on("/alert/warning", HTTP_OPTIONS, handleOptions);
  server.on("/alert/off",     HTTP_OPTIONS, handleOptions);
  server.on("/alert/on",      HTTP_OPTIONS, handleOptions);
  server.on("/light/on",      HTTP_OPTIONS, handleOptions);
  server.on("/light/off",     HTTP_OPTIONS, handleOptions);
  server.onNotFound(handleNotFound);
  server.begin();
  Serial.println("[HTTP] Web server started on port 80");

  // Hardware watchdog
  esp_task_wdt_config_t twdt_config = {
    .timeout_ms   = WDT_TIMEOUT_S * 1000,
    .idle_core_mask = (1 << 0),
    .trigger_panic = true
  };
  esp_task_wdt_init(&twdt_config);
  esp_task_wdt_add(NULL);

  Serial.println("[READY] CrowdPulse ESP32 IoT Node operational");
  Serial.println("================================================");
  g_last_humidity_ts = millis();
}

// =============================================================================
// MAIN LOOP
// =============================================================================
void loop() {
  esp_task_wdt_reset();
  unsigned long now = millis();

  // ── 1. Handle incoming HTTP requests (non-blocking) ────────────────────
  server.handleClient();

  // ── 2. Update buzzer state machine ─────────────────────────────────────
  updateBuzzer();

  // ── 3. Read sensors ────────────────────────────────────────────────────
  if (g_sensor_ok) {
    sensors_event_t humidity_event, temp_event;
    aht.getEvent(&humidity_event, &temp_event);
    g_temperature = temp_event.temperature;
    g_humidity    = humidity_event.relative_humidity;
  }
  g_gas_raw = readMQ2Averaged();

  // ── 4. Humidity rate of change (fire signature: fast humidity drop) ────
  if (now - g_last_humidity_ts >= 10000) {
    float dt = (now - g_last_humidity_ts) / 1000.0;
    g_humidity_rate = (g_prev_humidity - g_humidity) / dt;
    g_prev_humidity = g_humidity;
    g_last_humidity_ts = now;
  }

  // ── 5. On-device ML inference ──────────────────────────────────────────
  float features[] = {g_temperature, g_humidity, g_gas_raw};
  g_fire_pred  = classifier.predict(features);
  g_fire_votes = getFireVoteCount(features);

  // ── 6. Serial output (backward compat with Raspberry Pi) ──────────────
  Serial.print("T:");
  Serial.print(g_temperature, 2);
  Serial.print(",H:");
  Serial.print(g_humidity, 2);
  Serial.print(",G:");
  Serial.print(g_gas_raw, 0);
  Serial.print(",PRED:");
  Serial.print(g_fire_pred);
  Serial.print(",CONF:");
  Serial.println(g_fire_votes);

  // ── 7. Local fire detection → auto-activate buzzer ─────────────────────
  if (g_fire_pred == 1 && g_alert_mode == ALERT_OFF) {
    // On-device model detected fire — activate alarm locally
    // (Digital Twin can override via /alert/off if it's a false positive)
    g_alert_mode = ALERT_FIRE;
    digitalWrite(BUZZER_PIN, HIGH);
    Serial.println("[LOCAL] On-device fire detection → buzzer ON");
  }

  // ── 8. GPIO wake signal to Raspberry Pi (optional) ─────────────────────
  if (g_fire_pred == 1 && now - g_last_wake > WAKE_COOLDOWN_MS) {
    digitalWrite(WAKE_UP_PIN, HIGH);
    delay(WAKE_HOLD_MS);
    digitalWrite(WAKE_UP_PIN, LOW);
    g_last_wake = now;
  }

  // ── 9. Push sensor data to Digital Twin Hub ────────────────────────────
  if (now - g_last_push >= PUSH_INTERVAL_MS) {
    pushToHub();
    g_last_push = now;
  }

  // ── 10. WiFi reconnect if dropped ──────────────────────────────────────
  if (WiFi.status() != WL_CONNECTED && now - g_last_wifi_try > WIFI_RETRY_MS) {
    Serial.println("[WiFi] Reconnecting...");
    WiFi.reconnect();
    g_last_wifi_try = now;
  }

  // ── 11. Wait for next sample cycle ─────────────────────────────────────
  delay(SAMPLE_RATE_MS);
}
