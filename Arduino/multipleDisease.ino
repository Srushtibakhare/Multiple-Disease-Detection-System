#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <math.h>

const char* ssid = "Hotspot";
const char* password = "sam123456";

const char* reportUrl = "http://localhost:5000/reportNewIPAddress";
const char* secretToken = "myValidity";

#define I2C_SDA 21
#define I2C_SCL 22
#define ONE_WIRE_BUS 4
#define ECG_PIN 34

WebServer server(80);

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensor(&oneWire);

MAX30105 particleSensor;
long lastIR = 0;
long lastRed = 0;
bool fingerDetected = false;

const float ADC_REF_VOLT = 3.3f;
const int ADC_RESOLUTION = 4095;

float lastTempC = NAN;
unsigned long lastTempUpdate = 0;
const unsigned long TEMP_UPDATE_INTERVAL = 1500;

int lastEcgRaw = 0;
float lastEcgVoltage = 0.0f;


unsigned long lastBeatMillis = 0;
float bpmBuffer[8];
int bpmIndex = 0;
bool bpmFilled = false;
float lastBPM = 0.0f;

const uint32_t IR_MIN_FINGER = 50000UL;


// SpO2 estimation buffer
static const int SPO2_WINDOW = 100;
long redBuffer[SPO2_WINDOW];
long irBuffer[SPO2_WINDOW];
int spo2Index = 0;
int spo2Count = 0;
float lastSpO2 = NAN;
unsigned long lastSpo2Calc = 0;
const unsigned long SPO2_CALC_INTERVAL = 1000;

float getSmoothedBPM(float newBPM) {
  bpmBuffer[bpmIndex++] = newBPM;
  if (bpmIndex >= 8) {
    bpmIndex = 0;
    bpmFilled = true;
  }

  int count = bpmFilled ? 8 : bpmIndex;
  if (count <= 0) return 0.0f;

  float sum = 0.0f;
  for (int i = 0; i < count; i++) sum += bpmBuffer[i];
  return sum / count;
}

void updateTemperatureIfNeeded() {
  if (millis() - lastTempUpdate < TEMP_UPDATE_INTERVAL) return;
  lastTempUpdate = millis();

  tempSensor.requestTemperatures();
  float t = tempSensor.getTempCByIndex(0);
  if (t != DEVICE_DISCONNECTED_C) {
    lastTempC = t;
  }
}

float computeSpO2FromWindow() {
  if (spo2Count < SPO2_WINDOW) return NAN;

  double redMean = 0.0;
  double irMean = 0.0;

  for (int i = 0; i < SPO2_WINDOW; i++) {
    redMean += redBuffer[i];
    irMean += irBuffer[i];
  }
  redMean /= SPO2_WINDOW;
  irMean /= SPO2_WINDOW;

  if (redMean <= 0.0 || irMean <= 0.0) return NAN;

  double redAc = 0.0;
  double irAc = 0.0;

  for (int i = 0; i < SPO2_WINDOW; i++) {
    double rd = redBuffer[i] - redMean;
    double iv = irBuffer[i] - irMean;
    redAc += rd * rd;
    irAc += iv * iv;
  }

  redAc = sqrt(redAc / SPO2_WINDOW);
  irAc = sqrt(irAc / SPO2_WINDOW);

  if (redAc <= 0.0 || irAc <= 0.0) return NAN;

  double ratio = (redAc / redMean) / (irAc / irMean);

  // Estimate only, not medical-grade
  double spo2 = 110.0 - 25.0 * ratio;

  if (spo2 > 100.0) spo2 = 100.0;
  if (spo2 < 0.0) spo2 = 0.0;

  return (float)spo2;
}

void updateMax30105() {
  particleSensor.check();

  while (particleSensor.available()) {
    long ir = particleSensor.getIR();
    long red = particleSensor.getRed();
    lastIR = ir;
    lastRed = red;
    fingerDetected = (ir >= (long)IR_MIN_FINGER);

    if (fingerDetected) {
      if (checkForBeat(ir)) {
        unsigned long now = millis();
        if (lastBeatMillis > 0) {
          unsigned long delta = now - lastBeatMillis;
          if (delta > 200 && delta < 2000) {
            float bpm = 60000.0f / (float)delta;
            lastBPM = getSmoothedBPM(bpm);
          }
        }
        lastBeatMillis = now;
      }

      redBuffer[spo2Index] = red;
      irBuffer[spo2Index] = ir;
      spo2Index = (spo2Index + 1) % SPO2_WINDOW;
      if (spo2Count < SPO2_WINDOW) spo2Count++;
    } else {
      lastBPM = 0.0f;
      spo2Count = 0;
      spo2Index = 0;
    }

    particleSensor.nextSample();
  }

  if (millis() - lastSpo2Calc >= SPO2_CALC_INTERVAL) {
    lastSpo2Calc = millis();
    if (fingerDetected && spo2Count >= SPO2_WINDOW) {
      lastSpO2 = computeSpO2FromWindow();
    } else {
      lastSpO2 = NAN;
    }
  }
}

void reportIPToServer() {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(reportUrl);
  http.addHeader("Content-Type", "application/x-www-form-urlencoded");

  String body = "token=" + String(secretToken) +
                "&ip=" + WiFi.localIP().toString();

  int httpCode = http.POST(body);
  String payload = http.getString();

  Serial.print("IP report HTTP code: ");
  Serial.println(httpCode);
  Serial.print("Server response: ");
  Serial.println(payload);

  http.end();
}

void handleTemp() {
  String json = "{";
  json += "\"temp_c\":";
  if (isnan(lastTempC)) json += "null";
  else json += String(lastTempC, 2);
  json += "}";

  server.send(200, "application/json", json);
}

void handleEcg() {
  String json = "{";
  json += "\"ecg_raw\":" + String(lastEcgRaw) + ",";
  json += "\"ecg_voltage\":" + String(lastEcgVoltage, 3);
  json += "}";

  server.send(200, "application/json", json);
}

void handleOximeter() {
  String json = "{";
  json += "\"finger_detected\":" + String(fingerDetected ? "true" : "false") + ",";
  json += "\"heart_rate_bpm\":";
  if (fingerDetected && lastBPM > 0.0f) json += String((int)(lastBPM + 0.5f));
  else json += "null";
  json += ",";
  json += "\"spo2_percent\":";
  if (fingerDetected && !isnan(lastSpO2)) json += String(lastSpO2, 1);
  else json += "null";
  json += ",";
  json += "\"red_raw\":" + String(lastRed) + ",";
  json += "\"ir_raw\":" + String(lastIR);
  json += "}";

  server.send(200, "application/json", json);
}

void handleRoot() {
  server.send(404, "text/plain", "Use /temp, /ecg, /oximeter");
}
void updateEcgCache() {
  lastEcgRaw = analogRead(ECG_PIN);
  lastEcgVoltage = (lastEcgRaw * 3.3) / 4095.0;
} 

void setup() {
  Serial.begin(115200);
  delay(100);

  pinMode(ECG_PIN, INPUT);
  Wire.begin(I2C_SDA, I2C_SCL);

  tempSensor.begin();

  Serial.println("Before MAX init");

if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD, 0x57)) {
  Serial.println("MAX30105 not found at 0x57");
  // while (1) delay(1000);   // optional: freeze if not found
} else {
  Serial.println("MAX30105 detected!");
}

Serial.println("After MAX init");

  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.setPulseAmplitudeGreen(0);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");

int attempts = 0;

while (WiFi.status() != WL_CONNECTED && attempts < 10) {
  delay(500);
  Serial.print(".");
  Serial.print(" Status: ");
  Serial.println(WiFi.status());
  attempts++;
}

if (WiFi.status() == WL_CONNECTED) {
  Serial.println("\nWiFi connected");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
} else {
  Serial.println("\nWiFi FAILED (continuing without WiFi)");
}

  Serial.println();
  Serial.println("WiFi connected");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  reportIPToServer();

  server.on("/", handleRoot);
  server.on("/temp", HTTP_GET, handleTemp);
  server.on("/ecg", HTTP_GET, handleEcg);
  server.on("/oximeter", HTTP_GET, handleOximeter);
  server.begin();

  tempSensor.requestTemperatures();
  float t = tempSensor.getTempCByIndex(0);
  if (t != DEVICE_DISCONNECTED_C) lastTempC = t;
}

void loop() {

  static bool reportedOnce = false;
  static unsigned long lastAttempt = 0;

  if (WiFi.status() == WL_CONNECTED) {

    if (!reportedOnce) {
      Serial.println("WiFi Connected Successfully");
      Serial.print("IP: ");
      Serial.println(WiFi.localIP());

      reportIPToServer();
      reportedOnce = true;
    }

  } else {

    if (millis() - lastAttempt > 5000) {
      Serial.println("Reconnecting WiFi...");
      WiFi.disconnect();
      WiFi.begin(ssid, password);
      lastAttempt = millis();
      reportedOnce = false;
    }

    Serial.print("WiFi Status: ");
    Serial.println(WiFi.status());
  }

  server.handleClient();

  updateTemperatureIfNeeded();
  updateEcgCache();
  updateMax30105();

  delay(2);
}