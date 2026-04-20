import os
import sys

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from werkzeug.utils import secure_filename

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.detector import AnomalyDetector
from ml.preprocessor import ForensicPreprocessor
from ml.risk_scorer import RiskScorer

app = Flask(__name__)
CORS(app)
TEMP_DIR = os.path.join(ROOT_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

preprocessor = ForensicPreprocessor()
detector = AnomalyDetector(contamination=0.1)
scorer = RiskScorer()

LABEL_COLUMN_CANDIDATES = ["Label", "label", "Class", "class", "Target", "target"]


def _save_uploaded_file():
    if "file" not in request.files:
        return None, (jsonify({"error": "No file uploaded"}), 400)

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return None, (jsonify({"error": "Uploaded filename is empty"}), 400)

    # Sanitize incoming filename before writing to disk.
    safe_name = secure_filename(uploaded_file.filename)
    filepath = os.path.join(TEMP_DIR, safe_name)
    uploaded_file.save(filepath)
    return filepath, None


def _extract_binary_ground_truth(df):
    label_col = next((c for c in LABEL_COLUMN_CANDIDATES if c in df.columns), None)
    if not label_col:
        return None, None

    labels = df[label_col].astype(str).str.strip().str.lower()
    # CICIDS2017 uses "BENIGN" vs attack family labels.
    y_true = np.where(labels.isin(["benign", "normal", "0"]), 0, 1)
    return y_true, label_col


def _evaluate_predictions(y_true, predictions, decision_scores):
    # IsolationForest returns -1 for anomaly and 1 for normal.
    y_pred = np.where(predictions == -1, 1, 0)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    if len(np.unique(y_true)) > 1:
        # IsolationForest decision_function: higher = more normal.
        # Negate so higher means more likely attack/anomaly.
        auc_score = roc_auc_score(y_true, -decision_scores)
        metrics["roc_auc"] = round(float(auc_score), 4)
    else:
        metrics["roc_auc"] = None

    return metrics

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "cyber-triage-backend"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint."""
    filepath, error_response = _save_uploaded_file()
    if error_response:
        return error_response

    try:
        # run_pipeline returns (scaled, raw, clean)
        df_scaled, df_raw, df_clean = preprocessor.run_pipeline(filepath)

        detector.train(df_scaled)
        predictions, scores = detector.predict(df_scaled)
        results_df = scorer.score_dataframe(df_clean, scores)

        top_results = results_df.head(100).to_dict(orient="records")
        summary = {
            "total_records": int(len(results_df)),
            "critical": int((results_df["priority"] == "CRITICAL").sum()),
            "high": int((results_df["priority"] == "HIGH").sum()),
            "medium": int((results_df["priority"] == "MEDIUM").sum()),
            "low": int((results_df["priority"] == "LOW").sum()),
        }

        response_payload = {"summary": summary, "artifacts": top_results}
        # Add metrics when a supported label column is available.
        y_true, label_column = _extract_binary_ground_truth(df_clean)
        if y_true is not None:
            response_payload["evaluation"] = {
                "label_column": label_column,
                "metrics": _evaluate_predictions(y_true, predictions, scores),
            }

        return jsonify(response_payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/report", methods=["POST"])
def generate_report():
    """Generate PDF report."""
    return jsonify({"message": "Report generation will be added in a later phase."}), 501


@app.route("/api/evaluate", methods=["POST"])
def evaluate_model():
    """
    Evaluate anomaly model using uploaded labeled dataset.
    Returns classic classification metrics for project reporting.
    """
    filepath, error_response = _save_uploaded_file()
    if error_response:
        return error_response

    try:
        df_scaled, _, df_clean = preprocessor.run_pipeline(filepath)
        y_true, label_column = _extract_binary_ground_truth(df_clean)
        if y_true is None:
            return (
                jsonify(
                    {
                        "error": "No label column found. Expected one of: "
                        + ", ".join(LABEL_COLUMN_CANDIDATES)
                    }
                ),
                400,
            )

        detector.train(df_scaled)
        predictions, scores = detector.predict(df_scaled)
        metrics = _evaluate_predictions(y_true, predictions, scores)

        return jsonify(
            {
                "status": "ok",
                "label_column": label_column,
                "records_evaluated": int(len(df_clean)),
                "metrics": metrics,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
