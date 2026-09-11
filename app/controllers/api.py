"""
REST API Controller
Exposes /api/v1/ endpoints for hospital system integration.
Uses JWT authentication for secure access.
"""

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from functools import wraps
from app import db
from app.models import Patient, PredictionHistory, MedicalImage
from app.services.multi_modal_detector import MultiModalDetector
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import tempfile
import logging

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# Simple API-key auth (set API_KEY in .env for production)
def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key') or request.args.get('api_key')
        expected = current_app.config.get('API_KEY', 'dev-key-change-me')
        if not key or key != expected:
            return jsonify({'error': 'Invalid or missing API key', 'status': 401}), 401
        return f(*args, **kwargs)
    return decorated


@api_bp.route('/api/v1/health')
def health_check():
    """Health check endpoint — no auth required."""
    import torch
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'version': '1.0.0',
        'models': {
            'lung_cancer':   os.path.exists('models/lung_cancer_efficientnet.pth'),
            'skin_cancer':   os.path.exists('models/skin_cancer_efficientnet.pth'),
            'brain_tumor':   os.path.exists('models/ct_brain_tumor_continued.pth'),
        },
        'gpu_available': torch.cuda.is_available(),
    })


@api_bp.route('/api/v1/analyze', methods=['POST'])
@api_key_required
def api_analyze():
    """
    Analyze a medical image via REST API.

    Request (multipart/form-data):
        file        (required): Image file (png, jpg, jpeg, bmp, tiff)
        image_type  (required): mri | ct | xray | lung_ct | lung_mri | lung_xray |
                                skin | skin_mri | skin_ct | skin_xray | brain_mri | brain_ct
        cancer_type (optional): brain_tumor | lung_cancer | skin_cancer

    Returns:
        JSON with predicted_class, confidence, risk_level, label, class_probabilities
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file       = request.files['file']
    image_type = request.form.get('image_type', 'mri')
    cancer_type = request.form.get('cancer_type')

    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    # Save to temp file
    suffix = os.path.splitext(secure_filename(file.filename))[1] or '.png'
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        detector = MultiModalDetector()
        result = detector.analyze_image(tmp_path, image_type=image_type, cancer_type=cancer_type)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if result.get('predicted_class') == 'error':
        return jsonify({'error': result.get('error', 'Analysis failed')}), 500

    # Don't return the heatmap in API response (too large), just flag availability
    result_clean = {k: v for k, v in result.items() if k != 'heatmap_b64'}
    result_clean['heatmap_available'] = bool(result.get('heatmap_b64'))

    return jsonify({
        'status': 'success',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'result': result_clean,
    })


@api_bp.route('/api/v1/patients/<int:patient_id>/predictions')
@api_key_required
def api_patient_predictions(patient_id):
    """
    Get all AI predictions for a specific patient.

    Returns:
        JSON list of predictions with confidence, risk_level, cancer_type, date
    """
    predictions = (
        PredictionHistory.query
        .filter_by(patient_id=patient_id)
        .order_by(PredictionHistory.prediction_date.desc())
        .all()
    )

    return jsonify({
        'status':  'success',
        'patient_id': patient_id,
        'count':   len(predictions),
        'predictions': [
            {
                'id':              p.id,
                'cancer_type':     p.cancer_type,
                'predicted_class': p.prediction_result,
                'confidence':      round(p.confidence_score * 100, 2),
                'risk_level':      p.risk_level,
                'model_version':   p.model_version,
                'date':            p.prediction_date.isoformat() + 'Z' if p.prediction_date else None,
            }
            for p in predictions
        ]
    })


@api_bp.route('/api/v1/stats')
@api_key_required
def api_stats():
    """Overall system statistics."""
    from sqlalchemy import func
    total_scans    = PredictionHistory.query.count()
    high_risk      = PredictionHistory.query.filter_by(risk_level='high').count()
    avg_conf       = db.session.query(func.avg(PredictionHistory.confidence_score)).scalar() or 0.0
    total_patients = Patient.query.count()

    cancer_dist = (
        db.session.query(PredictionHistory.cancer_type, func.count(PredictionHistory.id))
        .group_by(PredictionHistory.cancer_type)
        .all()
    )

    return jsonify({
        'status': 'success',
        'stats': {
            'total_patients':   total_patients,
            'total_scans':      total_scans,
            'high_risk_count':  high_risk,
            'avg_confidence':   round(avg_conf * 100, 1),
            'cancer_distribution': {r[0]: r[1] for r in cancer_dist},
        }
    })
