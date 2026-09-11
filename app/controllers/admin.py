"""
Admin Analytics Dashboard Controller
Provides system-wide statistics and charts for admin users.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import Patient, Doctor, PredictionHistory, MedicalImage, Appointment
from datetime import datetime, timedelta
from sqlalchemy import func

admin_bp = Blueprint('admin', __name__)


def require_admin():
    if not current_user.is_authenticated or not current_user.is_admin():
        flash('Admin access required.', 'error')
        return redirect(url_for('main.index'))
    return None


@admin_bp.route('/admin/dashboard')
@login_required
def admin_dashboard():
    """
    Main admin analytics dashboard with KPI cards and charts.
    """
    guard = require_admin()
    if guard:
        return guard

    # ── KPI Cards ──────────────────────────────────────────────────────────
    total_patients    = Patient.query.count()
    total_doctors     = Doctor.query.count()
    total_scans       = PredictionHistory.query.count()
    total_appointments = Appointment.query.count() if hasattr(Appointment, '__tablename__') else 0

    high_risk_count = PredictionHistory.query.filter(
        PredictionHistory.risk_level == 'high'
    ).count()

    avg_conf = db.session.query(func.avg(PredictionHistory.confidence_score)).scalar() or 0.0

    # Scans in the last 7 days
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_scans_count = PredictionHistory.query.filter(
        PredictionHistory.prediction_date >= week_ago
    ).count()

    # ── Recent predictions ─────────────────────────────────────────────────
    recent_predictions = (
        PredictionHistory.query
        .order_by(PredictionHistory.prediction_date.desc())
        .limit(15)
        .all()
    )

    # ── Cancer type distribution ───────────────────────────────────────────
    cancer_dist = (
        db.session.query(PredictionHistory.cancer_type, func.count(PredictionHistory.id))
        .group_by(PredictionHistory.cancer_type)
        .all()
    )
    cancer_labels = [r[0].replace('_', ' ').title() for r in cancer_dist]
    cancer_counts = [r[1] for r in cancer_dist]

    # ── Risk level distribution ────────────────────────────────────────────
    risk_dist = (
        db.session.query(PredictionHistory.risk_level, func.count(PredictionHistory.id))
        .group_by(PredictionHistory.risk_level)
        .all()
    )
    risk_labels = [(r[0] or 'unknown').capitalize() for r in risk_dist]
    risk_counts = [r[1] for r in risk_dist]

    # ── Scans per day (last 14 days) ───────────────────────────────────────
    days_data = []
    for i in range(13, -1, -1):
        day = datetime.utcnow().date() - timedelta(days=i)
        count = PredictionHistory.query.filter(
            func.date(PredictionHistory.prediction_date) == day
        ).count()
        days_data.append({'date': day.strftime('%b %d'), 'count': count})

    daily_labels = [d['date'] for d in days_data]
    daily_counts = [d['count'] for d in days_data]

    return render_template(
        'dashboard/admin/dashboard.html',
        user=current_user,
        total_patients=total_patients,
        total_doctors=total_doctors,
        total_scans=total_scans,
        total_appointments=total_appointments,
        high_risk_count=high_risk_count,
        avg_confidence=round(avg_conf * 100, 1),
        recent_scans_count=recent_scans_count,
        recent_predictions=recent_predictions,
        cancer_labels=cancer_labels,
        cancer_counts=cancer_counts,
        risk_labels=risk_labels,
        risk_counts=risk_counts,
        daily_labels=daily_labels,
        daily_counts=daily_counts,
    )


@admin_bp.route('/admin/api/stats')
@login_required
def admin_stats_api():
    """JSON API for real-time stats refresh."""
    guard = require_admin()
    if guard:
        return jsonify({'error': 'Access denied'}), 403

    return jsonify({
        'total_patients':  Patient.query.count(),
        'total_scans':     PredictionHistory.query.count(),
        'high_risk':       PredictionHistory.query.filter_by(risk_level='high').count(),
        'avg_confidence':  round((db.session.query(func.avg(PredictionHistory.confidence_score)).scalar() or 0) * 100, 1),
    })
