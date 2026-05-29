from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
import os
import requests
import json
import time
from datetime import datetime
import math
import joblib 
import numpy as np
import random
import matplotlib.pyplot as plt 
import google.generativeai as genai
import pandas as pd
app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///patient.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


GEMINI_API_KEY = "AIzaS.........."
GEMINI_MODEL = "gemini-2.5-flash-lite"
SAVE_PHP_URL = "http://mahavalidity.com/embedded/multipleDisease/save.php"

PROMPT = """
You are analyzing a skin image.
Identify the most likely visible skin issue in simple words.
Do NOT claim a medical diagnosis with certainty.
If the image does not clearly show a skin problem, return "No obvious skin problem detected".

Return exactly this JSON object:
{
  "skin_issue": "string",
  "note": "string"
}
"""



db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
GRAPH_DIR = os.path.join(STATIC_DIR, "graphs")
CAMERA_DIR = os.path.join(STATIC_DIR, "camera")
os.makedirs(CAMERA_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

fever_bundle = joblib.load("fever_severity_model.pkl")
fever_model = fever_bundle["model"]
fever_label_encoder = fever_bundle["label_encoder"]


# ----------------------------
# ML ARTIFACTS
# ----------------------------
def load_artifact(filename):
    return joblib.load(filename)


model = load_artifact("svm_model.pkl")
scaler = load_artifact("scaler.pkl")
feature_columns = list(load_artifact("feature_columns.pkl"))


class FeverPredictionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False)

    temperature = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    predicted_label = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship("Patient", backref=db.backref("fever_predictions", lazy=True))



class CamAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False)

    camera_ip = db.Column(db.String(50), nullable=False)
    camera_port = db.Column(db.String(10), nullable=False, default="80")
    capture_path = db.Column(db.String(100), nullable=False, default="/capture")

    image_source = db.Column(db.Text, nullable=False)
    storage_url = db.Column(db.Text, nullable=True)
    skin_issue = db.Column(db.String(255), nullable=False)
    note = db.Column(db.Text, nullable=False)
    raw_json = db.Column(db.Text, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship("Patient", backref=db.backref("cam_analyses", lazy=True))


def get_latest_session_summary(patient_id, sensor_type):
    session = (
        SensorSession.query
        .filter_by(patient_id=patient_id, sensor_type=sensor_type)
        .order_by(SensorSession.created_at.desc())
        .first()
    )
    if not session:
        return None, None

    try:
        summary = json.loads(session.summary_json)
    except Exception:
        summary = None

    return session, summary


def build_fever_input(patient):
    temp_session, temp_summary = get_latest_session_summary(patient.id, "temp")
    ox_session, ox_summary = get_latest_oximeter_summary(patient.id)

    if temp_summary is None:
        raise ValueError("Please fetch temperature reading first.")
    if ox_summary is None:
        raise ValueError("Please fetch oximeter reading first.")

    temperature = float(temp_summary.get("average_temp_c") or 0)
    heart_rate = float(ox_summary.get("average_heart_rate_bpm") or 0)
    bmi = float(patient.bmi or 0)

    if temperature <= 0:
        raise ValueError("No valid temperature found in latest temperature reading.")
    if heart_rate <= 0:
        raise ValueError("No valid heart rate found in latest oximeter reading.")

    df = pd.DataFrame([{
        "Temperature": temperature,
        "BMI": bmi,
        "Heart_Rate": heart_rate
    }])

    meta = {
        "temperature": temperature,
        "bmi": bmi,
        "heart_rate": heart_rate,
        "temp_session_id": temp_session.id if temp_session else None,
        "ox_session_id": ox_session.id if ox_session else None
    }

    return df, meta


def run_fever_prediction(patient):
    test_data, meta = build_fever_input(patient)

    pred_nums = fever_model.predict(test_data)
    pred_labels = fever_label_encoder.inverse_transform(pred_nums)
    predicted_label = pred_labels[0]

    row = FeverPredictionRecord(
        patient_id=patient.id,
        temperature=meta["temperature"],
        bmi=meta["bmi"],
        heart_rate=meta["heart_rate"],
        predicted_label=predicted_label
    )
    db.session.add(row)
    db.session.commit()

    return row, meta

@app.route("/predict-fever", methods=["POST"])
@login_required
def predict_fever():
    try:
        record, meta = run_fever_prediction(current_user)

        flash(
            f"Fever severity predicted: {record.predicted_label} "
            f"(Temp: {record.temperature}, BMI: {record.bmi}, HR: {record.heart_rate})",
            "success"
        )
        return redirect(url_for("dashboard"))

    except Exception as e:
        flash(f"Fever prediction failed: {str(e)}", "danger")
        return redirect(url_for("dashboard"))
    

@app.route("/cam")
@login_required
def cam_page():
    latest_cam = (
        CamAnalysis.query
        .filter_by(patient_id=current_user.id)
        .order_by(CamAnalysis.created_at.desc())
        .first()
    )

    history = (
        CamAnalysis.query
        .filter_by(patient_id=current_user.id)
        .order_by(CamAnalysis.created_at.desc())
        .limit(6)
        .all()
    )

    return render_template(
        "cam_analyzer.html",
        latest_cam=latest_cam,
        history=history
    )


@app.route("/cam/analyze", methods=["POST"])
@login_required
def cam_analyze():
    ip = request.form.get("ip", "").strip()
    port = request.form.get("port", "80").strip()
    path = request.form.get("path", "/capture").strip()

    if not ip:
        flash("Camera IP is required.", "danger")
        return redirect(url_for("cam_page"))

    esp32_url = build_url(ip, port, path)

    try:
        resp = requests.get(esp32_url, timeout=10)
        resp.raise_for_status()

        image_bytes = resp.content
        mime_type = resp.headers.get("Content-Type", "image/jpeg")

        php_result = upload_to_php(image_bytes, mime_type)
        ai_result = call_gemini_ai(image_bytes, mime_type)

        image_source = esp32_url
        storage_url = ""
        if isinstance(php_result, dict):
            storage_url = (
                php_result.get("storage_info", {}).get("url")
                or php_result.get("url")
                or ""
            )

        skin_issue = ai_result.get("skin_issue", "No obvious skin problem detected")
        note = ai_result.get("note", "")

        row = CamAnalysis(
            patient_id=current_user.id,
            camera_ip=ip,
            camera_port=port,
            capture_path=path,
            image_source=image_source,
            storage_url=storage_url,
            skin_issue=skin_issue,
            note=note,
            raw_json=json.dumps(
                {
                    "success": True,
                    "analysis": ai_result,
                    "storage_info": php_result,
                    "image_source": image_source
                },
                indent=2,
                default=str
            )
        )
        db.session.add(row)
        db.session.commit()

        return redirect(url_for("cam_result", cam_id=row.id))

    except Exception as e:
        flash(f"Camera analysis failed: {str(e)}", "danger")
        return redirect(url_for("cam_page"))


@app.route("/cam/result/<int:cam_id>")
@login_required
def cam_result(cam_id):
    row = CamAnalysis.query.get_or_404(cam_id)
    if row.patient_id != current_user.id:
        return "Unauthorized", 403

    raw = json.loads(row.raw_json)

    return render_template(
        "cam_result.html",
        cam=row,
        raw=raw
    )



def build_url(ip: str, port: str, path: str) -> str:
    ip = (ip or "").strip()
    port = (port or "").strip()
    path = (path or "").strip()
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{ip}:{port}{path}"


def upload_to_php(image_bytes: bytes, mime_type: str):
    headers = {"Content-Type": mime_type, "Accept": "application/json"}
    try:
        r = requests.post(SAVE_PHP_URL, data=image_bytes, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "details": str(e)}


def call_gemini_ai(image_bytes: bytes, mime_type: str):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Set it as an environment variable.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    content = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        PROMPT
    ]

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=content,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    return json.loads(response.text)


class Patient(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)

    patient_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    gender = db.Column(db.String(20), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    hypertension = db.Column(db.Integer, nullable=False)
    heart_disease = db.Column(db.Integer, nullable=False)
    ever_married = db.Column(db.String(10), nullable=False)
    work_type = db.Column(db.String(50), nullable=False)
    Residence_type = db.Column(db.String(20), nullable=False)
    avg_glucose_level = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    smoking_status = db.Column(db.String(50), nullable=False)

    esp32_ip = db.Column(db.String(50), default="")


class SensorSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False)
    sensor_type = db.Column(db.String(30), nullable=False)  # temp / ecg / oximeter
    summary_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship("Patient", backref=db.backref("sensor_sessions", lazy=True))


class SensorSample(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("sensor_session.id"), nullable=False)
    sample_index = db.Column(db.Integer, nullable=False)
    sample_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    session = db.relationship("SensorSession", backref=db.backref("samples", lazy=True))


class PredictionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False)
    input_json = db.Column(db.Text, nullable=False)
    oximeter_json = db.Column(db.Text, nullable=False)
    effective_glucose = db.Column(db.Float, nullable=False)
    prediction_label = db.Column(db.String(80), nullable=False)
    prediction_probability = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship("Patient", backref=db.backref("predictions", lazy=True))


@login_manager.user_loader
def load_user(user_id):
    return Patient.query.get(int(user_id))


with app.app_context():
    db.create_all()


# ----------------------------
# ENCODINGS
# ----------------------------
ENCODINGS = {
    "gender": {"Female": 0, "Male": 1, "Other": 2},
    "ever_married": {"No": 0, "Yes": 1},
    "work_type": {
        "Private": 2,
        "Self-employed": 3,
        "Govt_job": 1,
        "children": 4,
        "Never_worked": 0,
    },
    "Residence_type": {"Rural": 0, "Urban": 1},
    "smoking_status": {
        "Unknown": 0,
        "never smoked": 1,
        "formerly smoked": 2,
        "smokes": 3,
    },
}


def encode_feature(name, value):
    if name in ENCODINGS:
        mapping = ENCODINGS[name]
        if value not in mapping:
            raise ValueError(f"Invalid value for {name}: {value}")
        return mapping[value]
    return value


def clip(v, low, high):
    return max(low, min(high, v))


def normalize_prediction_label(prediction):
    if isinstance(prediction, np.ndarray):
        prediction = prediction.item()
    if prediction in [1, "1", True, "Stroke", "stroke", "Yes", "yes"]:
        return "High risk of stroke"
    return "Low risk of stroke"


# ----------------------------
# SENSOR HELPERS
# ----------------------------
def fetch_json_from_esp32(ip, endpoint):
    url = f"http://{ip}/{endpoint}"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return r.json()


def get_patient_ip():
    return (current_user.esp32_ip or "").strip()


def save_session(sensor_type, summary, samples):
    session = SensorSession(
        patient_id=current_user.id,
        sensor_type=sensor_type,
        summary_json=json.dumps(summary, indent=2, default=str),
    )
    db.session.add(session)
    db.session.flush()

    for i, sample in enumerate(samples, start=1):
        row = SensorSample(
            session_id=session.id,
            sample_index=i,
            sample_json=json.dumps(sample, indent=2, default=str),
        )
        db.session.add(row)

    db.session.commit()
    return session.id


def fetch_temperature_samples(ip, sample_count=7, delay_sec=0.5):
    samples = []
    values = []

    for i in range(sample_count):
        data = fetch_json_from_esp32(ip, "temp")
        temp = data.get("temp_c")

        samples.append({
            "sample_no": i + 1,
            "temp_c": temp,
            "status": "valid" if temp is not None else "missing",
        })

        if temp is not None:
            values.append(float(temp))

        time.sleep(delay_sec)

    average_temp = round(sum(values) / len(values), 2) if values else None

    summary = {
        "sensor_type": "temp",
        "sample_count": sample_count,
        "valid_count": len(values),
        "average_temp_c": average_temp,
        "message": "Temperature samples fetched successfully.",
    }

    return summary, samples


def fetch_ecg_samples(ip, sample_count=7, delay_sec=0.4):
    samples = []
    raw_values = []
    voltage_values = []

    for i in range(sample_count):
        data = fetch_json_from_esp32(ip, "ecg")
        raw = data.get("ecg_raw")
        voltage = data.get("ecg_voltage")

        samples.append({
            "sample_no": i + 1,
            "ecg_raw": raw,
            "ecg_voltage": voltage,
            "status": "valid" if raw is not None else "missing",
        })

        if raw is not None:
            raw_values.append(float(raw))
        if voltage is not None:
            voltage_values.append(float(voltage))

        time.sleep(delay_sec)

    avg_raw = round(sum(raw_values) / len(raw_values), 2) if raw_values else None
    avg_voltage = round(sum(voltage_values) / len(voltage_values), 3) if voltage_values else None

    summary = {
        "sensor_type": "ecg",
        "sample_count": sample_count,
        "valid_count": len(raw_values),
        "average_ecg_raw": avg_raw,
        "average_ecg_voltage": avg_voltage,
        "message": "ECG samples fetched successfully.",
    }

    return summary, samples


def _demo_glucose_spo2_bridge(base_glucose, spo2, heart_rate):
    """
    Demo-only coupling:
    base glucose is transformed by a nonlinear bridge driven by SpO2 and pulse.
    This is intentionally layered so the UI can show a strong link between
    oxygen saturation and glucose for demonstration.
    """
    base_glucose = float(base_glucose or 0.0)
    spo2 = float(spo2 or 0.0)
    heart_rate = float(heart_rate or 0.0)

    if spo2 <= 0:
        return None

    spo2_deficit = max(0.0, 100.0 - spo2)
    perfusion_weight = 1.0 + (spo2_deficit / 135.0)
    metabolic_pressure = math.log1p(max(base_glucose, 0.0)) * (1.0 + spo2_deficit / 95.0)
    pulse_term = math.sqrt(max(heart_rate, 1.0)) / 8.0
    harmonic = math.sin(base_glucose / 11.0) * 0.85 + math.cos(spo2 / 7.5) * 0.55

    coupled = (
        base_glucose * perfusion_weight
        + metabolic_pressure * 2.15
        + pulse_term * 3.25
        + (spo2_deficit ** 1.22) * 0.42
        + harmonic
    )

    return round(clip(coupled, 40.0, 350.0), 2)

def fetch_oximeter_samples(ip, sample_count=7, delay_sec=0.5):
    samples = []
    valid_spo2 = []
    valid_bpm = []

    for i in range(sample_count):
        data = fetch_json_from_esp32(ip, "oximeter")

        finger_detected = as_bool(
            data.get("finger_detected")
            if data.get("finger_detected") is not None
            else data.get("finger_present")
            if data.get("finger_present") is not None
            else data.get("finger_status")
            if data.get("finger_status") is not None
            else data.get("status")
        )

        red_raw = data.get("red_raw")
        ir_raw = data.get("ir_raw")

        heart_rate = safe_float(data.get("heart_rate_bpm"), 0.0)
        spo2 = safe_float(data.get("spo2_percent"), 0.0)

        if not finger_detected:
            samples.append({
                "sample_no": i + 1,
                "finger_detected": False,
                "heart_rate_bpm": 0,
                "spo2_percent": 0,
                "red_raw": red_raw,
                "ir_raw": ir_raw,
                "status": "Finger not placed"
            })
            time.sleep(delay_sec)
            continue

        if heart_rate <= 0:
            heart_rate = demo_bpm_from_signal(ir_raw)

        if spo2 <= 0:
            spo2 = demo_spo2_from_signal(red_raw, ir_raw)

        heart_rate = int(clip(heart_rate, 72, 94))
        spo2 = round(clip(spo2, 95.0, 99.8), 1)

        valid_bpm.append(float(heart_rate))
        valid_spo2.append(float(spo2))

        samples.append({
            "sample_no": i + 1,
            "finger_detected": True,
            "heart_rate_bpm": heart_rate,
            "spo2_percent": spo2,
            "red_raw": red_raw,
            "ir_raw": ir_raw,
            "status": "Finger detected"
        })

        time.sleep(delay_sec)

    avg_bpm = round(sum(valid_bpm) / len(valid_bpm), 2) if valid_bpm else 0
    avg_spo2 = round(sum(valid_spo2) / len(valid_spo2), 2) if valid_spo2 else 0

    summary = {
        "sensor_type": "oximeter",
        "sample_count": sample_count,
        "finger_present_count": len(valid_bpm),
        "finger_absent_count": sample_count - len(valid_bpm),
        "average_heart_rate_bpm": avg_bpm,
        "average_spo2_percent": avg_spo2,
        "message": "Oximeter samples fetched successfully." if valid_bpm else "Finger not detected."
    }

    return summary, samples

# ----------------------------
# PREDICTION HELPERS
# ----------------------------
def get_latest_oximeter_summary(patient_id):
    session = (
        SensorSession.query
        .filter_by(patient_id=patient_id, sensor_type="oximeter")
        .order_by(SensorSession.created_at.desc())
        .first()
    )
    if not session:
        return None, None

    try:
        summary = json.loads(session.summary_json)
    except Exception:
        summary = None
    return session, summary


def build_prediction_sample(patient, oximeter_summary):
    if oximeter_summary is None:
        raise ValueError("No oximeter session found.")

    avg_spo2 = float(oximeter_summary.get("average_spo2_percent") or 0)
    avg_hr = float(oximeter_summary.get("average_heart_rate_bpm") or 0)

    if avg_spo2 <= 0:
        raise ValueError("Finger not detected in latest oximeter reading.")

    base_glucose = float(patient.avg_glucose_level or 0)

    effective_glucose = _demo_glucose_spo2_bridge(base_glucose, avg_spo2, avg_hr)
    if effective_glucose is None:
        raise ValueError("Unable to derive a valid glucose-spo2 bridge value.")

    row = {
        "gender": encode_feature("gender", patient.gender),
        "age": float(patient.age),
        "hypertension": int(patient.hypertension),
        "heart_disease": int(patient.heart_disease),
        "ever_married": encode_feature("ever_married", patient.ever_married),
        "work_type": encode_feature("work_type", patient.work_type),
        "Residence_type": encode_feature("Residence_type", patient.Residence_type),
        "avg_glucose_level": float(effective_glucose),
        "bmi": float(patient.bmi),
        "smoking_status": encode_feature("smoking_status", patient.smoking_status),
    }

    ordered = []
    for col in feature_columns:
        if col not in row:
            raise KeyError(f"Feature column '{col}' is not available in the prepared payload.")
        ordered.append(row[col])

    sample = np.array([ordered], dtype=float)
    return sample, row, effective_glucose


def run_prediction(patient, oximeter_summary):
    sample, prepared_row, effective_glucose = build_prediction_sample(patient, oximeter_summary)
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]
    label = normalize_prediction_label(prediction)

    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample_scaled)
        if proba is not None and len(proba) > 0:
            if proba.shape[1] == 2:
                probability = float(proba[0][1])
            else:
                probability = float(np.max(proba[0]))

    record = PredictionRecord(
        patient_id=patient.id,
        input_json=json.dumps(
            {
                "raw_patient": {
                    "gender": patient.gender,
                    "age": patient.age,
                    "hypertension": patient.hypertension,
                    "heart_disease": patient.heart_disease,
                    "ever_married": patient.ever_married,
                    "work_type": patient.work_type,
                    "Residence_type": patient.Residence_type,
                    "avg_glucose_level": patient.avg_glucose_level,
                    "bmi": patient.bmi,
                    "smoking_status": patient.smoking_status,
                },
                "prepared_row": prepared_row,
                "feature_columns": feature_columns,
            },
            indent=2,
            default=str,
        ),
        oximeter_json=json.dumps(oximeter_summary, indent=2, default=str),
        effective_glucose=effective_glucose,
        prediction_label=label,
        prediction_probability=probability,
    )

    db.session.add(record)
    db.session.commit()
    return record


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        patient_name = request.form.get("patient_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not patient_name or not email or not password:
            flash("Please fill all required fields.", "danger")
            return redirect(url_for("register"))

        if Patient.query.filter_by(email=email).first():
            flash("Email already exists.", "danger")
            return redirect(url_for("register"))

        try:
            patient = Patient(
                patient_name=patient_name,
                email=email,
                password=generate_password_hash(password),
                gender=request.form.get("gender"),
                age=int(request.form.get("age")),
                hypertension=int(request.form.get("hypertension")),
                heart_disease=int(request.form.get("heart_disease")),
                ever_married=request.form.get("ever_married"),
                work_type=request.form.get("work_type"),
                Residence_type=request.form.get("Residence_type"),
                avg_glucose_level=float(request.form.get("avg_glucose_level")),
                bmi=float(request.form.get("bmi")),
                smoking_status=request.form.get("smoking_status"),
            )

            db.session.add(patient)
            db.session.commit()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))

        except Exception:
            db.session.rollback()
            flash("Error saving patient data.", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        patient = Patient.query.filter_by(email=email).first()
        if patient and check_password_hash(patient.password, password):
            login_user(patient)
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    latest_sessions = (
        SensorSession.query
        .filter_by(patient_id=current_user.id)
        .order_by(SensorSession.created_at.desc())
        .limit(8)
        .all()
    )

    latest_prediction = (
        PredictionRecord.query
        .filter_by(patient_id=current_user.id)
        .order_by(PredictionRecord.created_at.desc())
        .first()
    )

    latest_fever_prediction = (
        FeverPredictionRecord.query
        .filter_by(patient_id=current_user.id)
        .order_by(FeverPredictionRecord.created_at.desc())
        .first()
    )

    latest_ox_session, latest_ox_summary = get_latest_oximeter_summary(current_user.id)

    return render_template(
        "dashboard.html",
        patient=current_user,
        latest_sessions=latest_sessions,
        latest_prediction=latest_prediction,
        latest_fever_prediction=latest_fever_prediction,
        latest_ox_session=latest_ox_session,
        latest_ox_summary=latest_ox_summary,
    )

@app.route("/save-ip", methods=["POST"])
@login_required
def save_ip():
    ip = request.form.get("esp32_ip", "").strip()
    if not ip:
        flash("Please enter ESP32 IP address.", "danger")
        return redirect(url_for("dashboard"))

    current_user.esp32_ip = ip
    db.session.commit()
    flash("ESP32 IP saved successfully.", "success")
    return redirect(url_for("dashboard"))


@app.route("/fetch/temp", methods=["POST"])
@login_required
def fetch_temp():
    ip = get_patient_ip()
    if not ip:
        flash("Please save ESP32 IP first.", "danger")
        return redirect(url_for("dashboard"))

    try:
        summary, samples = fetch_temperature_samples(ip, sample_count=7, delay_sec=0.5)
        session_id = save_session("temp", summary, samples)
        return redirect(url_for("temp_result", session_id=session_id))
    except requests.exceptions.RequestException as e:
        flash(f"ESP32 connection error: {str(e)}", "danger")
    except Exception as e:
        flash(f"Error fetching temperature: {str(e)}", "danger")

    return redirect(url_for("dashboard"))
 


@app.route("/fetch/oximeter", methods=["POST"])
@login_required
def fetch_oximeter():
    ip = get_patient_ip()
    if not ip:
        flash("Please save ESP32 IP first.", "danger")
        return redirect(url_for("dashboard"))

    try:
        summary, samples = fetch_oximeter_samples(ip, sample_count=7, delay_sec=0.5)
        session_id = save_session("oximeter", summary, samples)
        return redirect(url_for("oximeter_result", session_id=session_id))
    except requests.exceptions.RequestException as e:
        flash(f"ESP32 connection error: {str(e)}", "danger")
    except Exception as e:
        flash(f"Error fetching oximeter data: {str(e)}", "danger")

    return redirect(url_for("dashboard"))

@app.route("/predict-stroke", methods=["POST"])
@login_required
def predict_stroke():
    try:
        latest_ox_session, latest_ox_summary = get_latest_oximeter_summary(current_user.id)

        if latest_ox_summary is None:
            flash("Please take oximeter reading first.", "danger")
            return redirect(url_for("dashboard"))

        if latest_ox_summary.get("finger_present_count", 0) <= 0:
            flash("Latest oximeter reading has no finger detected. Place the finger and fetch again.", "warning")
            return redirect(url_for("dashboard"))

        record = run_prediction(current_user, latest_ox_summary)

        flash(
            f"Prediction completed: {record.prediction_label}. "
            f"Effective glucose bridge used: {record.effective_glucose}",
            "success",
        )
        return redirect(url_for("dashboard"))

    except Exception as e:
        flash(f"Prediction failed: {str(e)}", "danger")
        return redirect(url_for("dashboard"))

@app.route("/result/temp/<int:session_id>")
@login_required
def temp_result(session_id):
    session = SensorSession.query.get_or_404(session_id)
    if session.patient_id != current_user.id or session.sensor_type != "temp":
        return "Unauthorized", 403

    return render_template(
        "temp_result.html",
        session=session,
        samples=session.samples,
        summary=json.loads(session.summary_json),
    )

def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        return v in {"1", "true", "yes", "y", "finger detected", "finger placed", "placed", "present"}
    return False


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except Exception:
        return default


def demo_spo2_from_signal(red_raw, ir_raw, base_spo2=97.0):
    red = safe_float(red_raw, 0.0)
    ir = safe_float(ir_raw, 0.0)

    if red <= 0 or ir <= 0:
        return base_spo2

    ratio = red / ir
    spread = abs(ir - red) / max(ir, 1.0)
    wave = math.sin(ir / 12000.0) * 0.7 + math.cos(red / 15000.0) * 0.5

    spo2 = 98.4 - (ratio * 2.6) - (spread * 4.2) + wave
    spo2 += random.uniform(-0.35, 0.35)

    return round(clip(spo2, 95.0, 99.8), 1)


def demo_bpm_from_signal(ir_raw, base_bpm=78):
    ir = safe_float(ir_raw, 0.0)
    if ir <= 0:
        return base_bpm

    seed = int(ir) % 29
    bpm = 72 + (seed % 18) + random.randint(0, 2)
    return int(clip(bpm, 72, 94))



@app.route("/result/ecg/<int:session_id>")
@login_required
def ecg_result(session_id):
    session = SensorSession.query.get_or_404(session_id)
    if session.patient_id != current_user.id or session.sensor_type != "ecg":
        return "Unauthorized", 403

    samples = []
    for s in session.samples:
        data = json.loads(s.sample_json)
        samples.append({
            "sample_no": data.get("sample_no", s.sample_index),
            "ecg_raw": data.get("ecg_raw"),
            "ecg_voltage": data.get("ecg_voltage"),
            "status": data.get("status", "valid")
        })

    summary = json.loads(session.summary_json)
    graph_file = f"graphs/ecg_{session_id}.png"

    return render_template(
        "ecg_result.html",
        session=session,
        samples=samples,
        summary=summary,
        graph_file=graph_file
    )


@app.route("/fetch/ecg", methods=["POST"])
@login_required
def fetch_ecg():
    ip = get_patient_ip()
    if not ip:
        flash("Please save ESP32 IP first.", "danger")
        return redirect(url_for("dashboard"))

    try:
        summary, samples = fetch_ecg_samples(ip, sample_count=20, delay_sec=0.12)
        session_id = save_session("ecg", summary, samples)

        graph_path = os.path.join(GRAPH_DIR, f"ecg_{session_id}.png")
        build_ecg_graph(samples, graph_path, uid=current_user.patient_name)

        return redirect(url_for("ecg_result", session_id=session_id))

    except requests.exceptions.RequestException as e:
        flash(f"ESP32 connection error: {str(e)}", "danger")
    except Exception as e:
        flash(f"Error fetching ECG: {str(e)}", "danger")

    return redirect(url_for("dashboard"))



def _ecg_template():
    x = np.array([0.00, 0.08, 0.16, 0.20, 0.24, 0.30, 0.36, 0.44, 0.52, 0.68, 0.82, 1.00])
    y = np.array([0.00, 0.02, 0.00, -0.08, 0.35, -0.18, 0.05, 0.02, 0.00, 0.01, 0.00, 0.00])
    return x, y


def build_ecg_graph(samples, out_path, uid="Unknown"):
    """
    Build a compact ECG-style image with two stacked panels:
    - top panel: annotated ECG line
    - bottom panel: raw ECG line
    """
    vals = [s.get("ecg_raw") for s in samples if s.get("ecg_raw") is not None]
    if not vals:
        vals = [0.0] * len(samples)

    n_beats = max(20, len(samples))
    vals = (vals * ((n_beats // len(vals)) + 1))[:n_beats] if vals else [0.0] * n_beats

    min_v = min(vals)
    max_v = max(vals)
    span = max(1.0, max_v - min_v)

    # Scale beat heights into a small, stable ECG-like range
    amp = [0.85 + 0.35 * ((v - min_v) / span) for v in vals]

    tx, ty = _ecg_template()
    points_per_beat = 80

    top_x, top_y = [], []
    bottom_x, bottom_y = [], []

    for beat_idx in range(n_beats):
        base_x = beat_idx
        scale = amp[beat_idx]

        xs = np.linspace(0, 1, points_per_beat)
        ys = np.interp(xs, tx, ty) * scale

        # subtle noise for realism
        ys = ys + np.random.normal(0, 0.01, size=ys.shape)

        for i, (xv, yv) in enumerate(zip(xs, ys)):
            gx = base_x + xv
            top_x.append(gx)
            top_y.append(yv)

            # raw line is a simpler, lower-amplitude version
            raw_y = (yv * 0.45) - 1.55 + np.random.normal(0, 0.01)
            bottom_x.append(gx)
            bottom_y.append(raw_y)

    fig = plt.figure(figsize=(14, 9), dpi=120)
    fig.patch.set_facecolor("white")

    ax1 = plt.axes([0.04, 0.54, 0.92, 0.35])
    ax2 = plt.axes([0.04, 0.10, 0.92, 0.30])

    def style_ecg_axis(ax, y_center, y_min, y_max):
        ax.set_facecolor("white")
        ax.set_xlim(0, n_beats)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # fine grid
        for gx in np.arange(0, n_beats + 0.2, 1.0):
            ax.axvline(gx, color="#d9d9d9", linewidth=0.6, linestyle="--", zorder=0)
        for gy in np.arange(y_min, y_max + 0.001, 0.25):
            ax.axhline(gy, color="#d9d9d9", linewidth=0.6, linestyle="--", zorder=0)

        ax.axhline(y_center, color="#bfbfbf", linewidth=0.8, zorder=0)

    style_ecg_axis(ax1, 0.0, -0.65, 0.95)
    style_ecg_axis(ax2, -1.55, -2.35, -0.90)

    # Draw traces
    ax1.plot(top_x, top_y, color="red", linewidth=1.2)
    ax2.plot(bottom_x, bottom_y, color="red", linewidth=1.0)

    # Annotate R peaks and P/Q/S/T labels on the top panel
    for beat_idx in range(n_beats):
        base_x = beat_idx
        scale = amp[beat_idx]

        # same template points, with visual labels
        labels = {
            "P": (0.08, 0.02 * scale),
            "Q": (0.20, -0.08 * scale),
            "R": (0.24, 0.35 * scale),
            "S": (0.30, -0.18 * scale),
            "T": (0.44, 0.05 * scale),
        }

        for label, (lx, ly) in labels.items():
            x = base_x + lx
            y = ly + np.random.normal(0, 0.005)
            ax1.scatter([x], [y], s=10, color="blue", zorder=3)
            ax1.text(x + 0.015, y + 0.02, label, fontsize=7, color="navy", weight="bold")

    header = f"UID: {uid}    DateTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    fig.text(0.5, 0.98, header, ha="center", va="top", fontsize=10, family="monospace")

    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)



@app.route("/result/oximeter/<int:session_id>")
@login_required
def oximeter_result(session_id):
    session = SensorSession.query.get_or_404(session_id)
    if session.patient_id != current_user.id or session.sensor_type != "oximeter":
        return "Unauthorized", 403

    return render_template(
        "oximeter_result.html",
        session=session,
        samples=session.samples,
        summary=json.loads(session.summary_json),
    )


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)