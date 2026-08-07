from flask import Blueprint, render_template, redirect, url_for, request, flash, current_app
from flask_login import login_required, current_user
from app import db
from app.models import Patient, Doctor, ClinicalData, MedicalImage, PredictionHistory, Report
import os
from werkzeug.utils import secure_filename
from datetime import datetime

dashboard_bp = Blueprint('dashboard', __name__)

# Allowed file extensions for medical images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== PATIENT DASHBOARD ROUTES ====================

@dashboard_bp.route('/patient/dashboard')
@login_required
def patient_dashboard():
    """
    Patient dashboard welcome page.
    Shows overview of patient's account and quick actions.
    """
    if not current_user.is_patient():
        flash('Access denied. Patient dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    # Get patient's prediction count
    prediction_count = PredictionHistory.query.filter_by(patient_id=current_user.id).count()
    recent_predictions = PredictionHistory.query.filter_by(patient_id=current_user.id).order_by(PredictionHistory.prediction_date.desc()).limit(5).all()
    
    return render_template('dashboard/patient/dashboard.html', 
                          user=current_user,
                          prediction_count=prediction_count,
                          recent_predictions=recent_predictions)

@dashboard_bp.route('/patient/clinical-data', methods=['GET', 'POST'])
@login_required
def clinical_data():
    """
    Clinical data form for patients.
    Collects symptoms and medical history for prediction.
    """
    if not current_user.is_patient():
        flash('Access denied. Patient dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        # Collect clinical data from form
        symptoms = request.form.get('symptoms')
        family_history = request.form.get('family_history')
        lifestyle = request.form.getlist('lifestyle')  # Get multiple checkboxes
        additional_notes = request.form.get('additional_notes')
        
        # Create clinical data record
        clinical_data = ClinicalData(
            patient_id=current_user.id,
            symptoms=symptoms,
            family_history=family_history,
            lifestyle_factors=lifestyle,
            additional_notes=additional_notes
        )
        
        db.session.add(clinical_data)
        db.session.commit()
        
        flash('Clinical data submitted successfully. Prediction will be processed.', 'success')
        return redirect(url_for('dashboard.patient_history'))
    
    return render_template('dashboard/patient/clinical_data.html', user=current_user)

@dashboard_bp.route('/patient/upload-image', methods=['GET', 'POST'])
@login_required
def upload_image():
    """
    Medical image upload for patients.
    Allows uploading medical images for analysis.
    """
    if not current_user.is_patient():
        flash('Access denied. Patient dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{current_user.id}_{timestamp}_{filename}"
            
            # Create upload directory if it doesn't exist
            upload_dir = os.path.join(current_app.root_path, 'static', 'uploads', 'medical_images')
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            # Determine image type based on file selection or form
            image_type = request.form.get('image_type', 'xray')
            
            # Create medical image record
            medical_image = MedicalImage(
                patient_id=current_user.id,
                image_type=image_type,
                file_path=f'/static/uploads/medical_images/{filename}',
                file_name=filename,
                file_size=os.path.getsize(file_path)
            )
            
            db.session.add(medical_image)
            db.session.commit()
            
            flash('Image uploaded successfully. Analysis will be processed.', 'success')
            return redirect(url_for('dashboard.patient_history'))
        else:
            flash('Invalid file type. Allowed types: png, jpg, jpeg, gif, bmp, tiff', 'error')
    
    return render_template('dashboard/patient/upload_image.html', user=current_user)

@dashboard_bp.route('/patient/history')
@login_required
def patient_history():
    """
    Patient prediction history view.
    Shows all predictions made by the patient.
    """
    if not current_user.is_patient():
        flash('Access denied. Patient dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    predictions = PredictionHistory.query.filter_by(patient_id=current_user.id).order_by(PredictionHistory.prediction_date.desc()).all()
    
    return render_template('dashboard/patient/history.html', 
                          user=current_user,
                          predictions=predictions)

# ==================== DOCTOR DASHBOARD ROUTES ====================

@dashboard_bp.route('/doctor/dashboard')
@login_required
def doctor_dashboard():
    """
    Doctor dashboard welcome page.
    Shows overview of all patients and statistics.
    """
    if not current_user.is_doctor():
        flash('Access denied. Doctor dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    # Get statistics
    total_patients = Patient.query.count()
    total_predictions = PredictionHistory.query.count()
    recent_predictions = PredictionHistory.query.order_by(PredictionHistory.prediction_date.desc()).limit(10).all()
    
    return render_template('dashboard/doctor/dashboard.html',
                          user=current_user,
                          total_patients=total_patients,
                          total_predictions=total_predictions,
                          recent_predictions=recent_predictions)

@dashboard_bp.route('/doctor/patients')
@login_required
def doctor_patients():
    """
    View all patients for doctors.
    Lists all registered patients.
    """
    if not current_user.is_doctor():
        flash('Access denied. Doctor dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    patients = Patient.query.order_by(Patient.created_at.desc()).all()
    
    return render_template('dashboard/doctor/patients.html',
                          user=current_user,
                          patients=patients)

@dashboard_bp.route('/doctor/search-patient', methods=['GET', 'POST'])
@login_required
def search_patient():
    """
    Search patient functionality for doctors.
    Allows searching patients by name, email, or ID.
    """
    if not current_user.is_doctor():
        flash('Access denied. Doctor dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    patients = []
    search_query = request.args.get('q', '')
    
    if search_query:
        # Search by username, email, or full name
        patients = Patient.query.filter(
            (Patient.username.ilike(f'%{search_query}%')) |
            (Patient.email.ilike(f'%{search_query}%')) |
            (Patient.full_name.ilike(f'%{search_query}%'))
        ).all()
    
    return render_template('dashboard/doctor/search_patient.html',
                          user=current_user,
                          patients=patients,
                          search_query=search_query)

@dashboard_bp.route('/doctor/patient/<int:patient_id>')
@login_required
def doctor_patient_detail(patient_id):
    """
    View detailed patient information for doctors.
    Shows patient profile and prediction history.
    """
    if not current_user.is_doctor():
        flash('Access denied. Doctor dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    patient = Patient.query.get_or_404(patient_id)
    
    predictions = PredictionHistory.query.filter_by(patient_id=patient_id).order_by(PredictionHistory.prediction_date.desc()).all()
    reports = Report.query.filter_by(patient_id=patient_id).order_by(Report.report_date.desc()).all()
    
    return render_template('dashboard/doctor/patient_detail.html',
                          user=current_user,
                          patient=patient,
                          predictions=predictions,
                          medical_records=reports)

@dashboard_bp.route('/doctor/predictions')
@login_required
def doctor_predictions():
    """
    View all predictions for doctors.
    Shows prediction history across all patients.
    """
    if not current_user.is_doctor():
        flash('Access denied. Doctor dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    predictions = PredictionHistory.query.order_by(PredictionHistory.prediction_date.desc()).all()
    
    return render_template('dashboard/doctor/predictions.html',
                          user=current_user,
                          predictions=predictions)

@dashboard_bp.route('/doctor/reports')
@login_required
def doctor_reports():
    """
    View reports for doctors.
    Shows medical records and reports.
    """
    if not current_user.is_doctor():
        flash('Access denied. Doctor dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    reports = Report.query.order_by(Report.report_date.desc()).all()
    
    return render_template('dashboard/doctor/reports.html',
                          user=current_user,
                          medical_records=reports)
