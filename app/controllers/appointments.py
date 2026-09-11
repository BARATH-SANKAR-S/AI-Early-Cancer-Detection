"""
Appointment Booking Controller
Handles booking, confirmation, and cancellation of appointments.
"""

from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import Patient, Doctor, Appointment, PredictionHistory
from datetime import datetime

appointments_bp = Blueprint('appointments', __name__)


@appointments_bp.route('/appointments')
@login_required
def list_appointments():
    """Patient's appointment list."""
    if not current_user.is_patient():
        flash('Access denied.', 'error')
        return redirect(url_for('main.index'))

    appts = (
        Appointment.query
        .filter_by(patient_id=current_user.id)
        .order_by(Appointment.scheduled_at.desc())
        .all()
    )
    doctors = Doctor.query.order_by(Doctor.full_name).all()
    predictions = PredictionHistory.query.filter_by(patient_id=current_user.id).order_by(
        PredictionHistory.prediction_date.desc()
    ).all()

    return render_template(
        'dashboard/patient/appointments.html',
        user=current_user,
        appointments=appts,
        doctors=doctors,
        predictions=predictions,
    )


@appointments_bp.route('/appointments/book', methods=['POST'])
@login_required
def book_appointment():
    """Book a new appointment."""
    if not current_user.is_patient():
        flash('Access denied.', 'error')
        return redirect(url_for('main.index'))

    doctor_id     = request.form.get('doctor_id', type=int)
    scheduled_str = request.form.get('scheduled_at')
    reason        = request.form.get('reason', '').strip()
    prediction_id = request.form.get('prediction_id', type=int) or None

    if not doctor_id or not scheduled_str:
        flash('Please select a doctor and appointment time.', 'error')
        return redirect(url_for('appointments.list_appointments'))

    try:
        scheduled_at = datetime.strptime(scheduled_str, '%Y-%m-%dT%H:%M')
    except ValueError:
        flash('Invalid date/time format.', 'error')
        return redirect(url_for('appointments.list_appointments'))

    if scheduled_at < datetime.utcnow():
        flash('Appointment time must be in the future.', 'error')
        return redirect(url_for('appointments.list_appointments'))

    appt = Appointment(
        patient_id=current_user.id,
        doctor_id=doctor_id,
        prediction_id=prediction_id,
        scheduled_at=scheduled_at,
        reason=reason,
        status='pending',
    )
    db.session.add(appt)
    db.session.commit()

    flash('Appointment request submitted successfully! Doctor will confirm shortly.', 'success')
    return redirect(url_for('appointments.list_appointments'))


@appointments_bp.route('/appointments/<int:appt_id>/cancel', methods=['POST'])
@login_required
def cancel_appointment(appt_id):
    """Cancel an appointment (patient side)."""
    appt = Appointment.query.get_or_404(appt_id)
    if appt.patient_id != current_user.id:
        flash('Access denied.', 'error')
        return redirect(url_for('appointments.list_appointments'))

    appt.status = 'cancelled'
    db.session.commit()
    flash('Appointment cancelled.', 'info')
    return redirect(url_for('appointments.list_appointments'))


# ── Doctor-side appointment management ─────────────────────────────────────

@appointments_bp.route('/doctor/appointments')
@login_required
def doctor_appointments():
    """Doctor's appointment schedule."""
    if not current_user.is_doctor():
        flash('Access denied.', 'error')
        return redirect(url_for('main.index'))

    pending   = Appointment.query.filter_by(doctor_id=current_user.id, status='pending').order_by(Appointment.scheduled_at).all()
    confirmed = Appointment.query.filter_by(doctor_id=current_user.id, status='confirmed').order_by(Appointment.scheduled_at).all()
    past      = Appointment.query.filter(
        Appointment.doctor_id == current_user.id,
        Appointment.status.in_(['completed', 'cancelled'])
    ).order_by(Appointment.scheduled_at.desc()).limit(20).all()

    return render_template(
        'dashboard/doctor/appointments.html',
        user=current_user,
        pending=pending,
        confirmed=confirmed,
        past=past,
    )


@appointments_bp.route('/doctor/appointments/<int:appt_id>/confirm', methods=['POST'])
@login_required
def confirm_appointment(appt_id):
    """Doctor confirms an appointment."""
    if not current_user.is_doctor():
        flash('Access denied.', 'error')
        return redirect(url_for('main.index'))

    appt = Appointment.query.get_or_404(appt_id)
    if appt.doctor_id != current_user.id:
        flash('Access denied.', 'error')
        return redirect(url_for('appointments.doctor_appointments'))

    appt.status = 'confirmed'
    appt.notes  = request.form.get('notes', '')
    db.session.commit()
    flash('Appointment confirmed!', 'success')
    return redirect(url_for('appointments.doctor_appointments'))


@appointments_bp.route('/doctor/appointments/<int:appt_id>/complete', methods=['POST'])
@login_required
def complete_appointment(appt_id):
    """Mark appointment as completed."""
    if not current_user.is_doctor():
        flash('Access denied.', 'error')
        return redirect(url_for('main.index'))

    appt = Appointment.query.get_or_404(appt_id)
    if appt.doctor_id != current_user.id:
        flash('Access denied.', 'error')
        return redirect(url_for('appointments.doctor_appointments'))

    appt.status = 'completed'
    db.session.commit()
    flash('Appointment marked as completed.', 'success')
    return redirect(url_for('appointments.doctor_appointments'))
