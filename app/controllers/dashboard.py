from flask import Blueprint, render_template, redirect, url_for, request, flash, current_app, send_file, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import Patient, Doctor, ClinicalData, MedicalImage, PredictionHistory, Report, Appointment, DoctorNote
from app.services.vit_inference import predict_image
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from pathlib import Path

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
            image_type = request.form.get('image_type', 'mri')
            
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
            
            flash('Image uploaded successfully. Click "Analyze" to process with AI.', 'success')
            return redirect(url_for('dashboard.patient_history'))
        else:
            flash('Invalid file type. Allowed types: png, jpg, jpeg, gif, bmp, tiff', 'error')
    
    return render_template('dashboard/patient/upload_image.html', user=current_user)


@dashboard_bp.route('/patient/analyze-image/<int:image_id>', methods=['POST'])
@login_required
def analyze_image(image_id):
    """
    Analyze uploaded medical image using AI models.
    Runs real AI prediction and saves results to database.
    """
    import sys
    import traceback
    
    print(f"[ANALYZE] Starting analysis for image_id: {image_id}", file=sys.stderr)
    
    if not current_user.is_patient():
        flash('Access denied. Patient dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    # Get the medical image
    medical_image = MedicalImage.query.get_or_404(image_id)
    print(f"[ANALYZE] Medical image found: {medical_image.file_name}", file=sys.stderr)
    
    # Verify ownership
    if medical_image.patient_id != current_user.id:
        flash('Access denied. You can only analyze your own images.', 'error')
        return redirect(url_for('dashboard.patient_history'))
    
    # Check if already analyzed (we allow re-analysis)
    existing_prediction = PredictionHistory.query.filter_by(medical_image_id=medical_image.id).first()
    
    try:
        # Get full file path - handle both absolute and relative paths
        if medical_image.file_path.startswith('/'):
            # Remove leading slash and join with root path
            file_path = os.path.join(current_app.root_path, medical_image.file_path.lstrip('/'))
        else:
            # Use as-is if it's already relative
            file_path = os.path.join(current_app.root_path, medical_image.file_path)
        
        print(f"[ANALYZE] File path: {file_path}", file=sys.stderr)
        
        # Verify file exists
        if not os.path.exists(file_path):
            flash(f'Image file not found at: {file_path}', 'error')
            return redirect(url_for('dashboard.patient_history'))
        
        # Determine cancer type based on image type
        image_type = medical_image.image_type if medical_image.image_type else 'mri'
        print(f"[ANALYZE] Image type: {image_type}", file=sys.stderr)

        CANCER_TYPE_MAP = {
            'mri':         'brain_tumor',
            'ct':          'brain_tumor',
            'brain_mri':   'brain_tumor',
            'brain_ct':    'brain_tumor',
            'xray':        'lung_cancer',
            'lung':        'lung_cancer',
            'lung_ct':     'lung_cancer',
            'lung_mri':    'lung_cancer',
            'lung_xray':   'lung_cancer',
            'skin':        'skin_cancer',
            'skin_mri':    'skin_cancer',
            'skin_ct':     'skin_cancer',
            'skin_xray':   'skin_cancer',
            'dermoscopy':  'skin_cancer',
        }
        cancer_type = CANCER_TYPE_MAP.get(image_type.lower(), 'lung_cancer' if 'lung' in image_type.lower() else ('skin_cancer' if 'skin' in image_type.lower() else 'brain_tumor'))
        
        print(f"[ANALYZE] Cancer type: {cancer_type}", file=sys.stderr)
        
        # Use multi-modal detector for automatic image type detection
        print(f"[ANALYZE] Loading detector...", file=sys.stderr)
        from app.services.multi_modal_detector import MultiModalDetector
        detector = MultiModalDetector()
        
        print(f"[ANALYZE] Starting analysis...", file=sys.stderr)
        try:
            prediction_results = detector.analyze_image(file_path, image_type, cancer_type)
            print(f"[ANALYZE] Analysis complete: {prediction_results}", file=sys.stderr)
        except Exception as e:
            print(f"[ANALYZE] Error in detector.analyze_image: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            flash(f'Analysis error: {str(e)}', 'error')
            return redirect(url_for('dashboard.patient_history'))
        
        # Resolve risk level (never allow None)
        risk_level = prediction_results.get('risk_level')
        if not risk_level or risk_level == '':
            from app.services.brain_inference import get_brain_risk_level
            risk_level = get_brain_risk_level(
                prediction_results.get('predicted_class'),
                prediction_results.get('confidence', 0.0)
            )

        # Create or update prediction history record
        additional_data = {
            'class_probabilities': prediction_results.get('class_probabilities', {}),
            'image_type': image_type,
            'label': prediction_results.get('label', ''),
            'risk_level': risk_level,
            'is_malignant': prediction_results.get('is_malignant', False),
            'heatmap_b64': prediction_results.get('heatmap_b64', None),
        }

        if existing_prediction:
            prediction = existing_prediction
            prediction.cancer_type = cancer_type
            prediction.prediction_result = prediction_results['predicted_class']
            prediction.confidence_score = prediction_results['confidence']
            prediction.risk_level = risk_level
            prediction.model_version = prediction_results.get('model_name', 'Unknown')
            prediction.additional_data = additional_data
            prediction.prediction_date = datetime.utcnow()
        else:
            prediction = PredictionHistory(
                patient_id=current_user.id,
                cancer_type=cancer_type,
                prediction_result=prediction_results['predicted_class'],
                confidence_score=prediction_results['confidence'],
                risk_level=risk_level,
                medical_image_id=medical_image.id,
                model_version=prediction_results.get('model_name', 'Unknown'),
                additional_data=additional_data,
            )
            db.session.add(prediction)
        
        db.session.commit()
        
        # Redirect to dedicated results page instead of history list
        return redirect(url_for('dashboard.analysis_result', image_id=image_id))
        
    except Exception as e:
        print(f"[ANALYZE] General error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        flash(f'Error during analysis: {str(e)}', 'error')
        return redirect(url_for('dashboard.patient_history'))

# ─── PDF Report Download ────────────────────────────────────────────────────
@dashboard_bp.route('/download-report/<int:prediction_id>')
@dashboard_bp.route('/patient/download-report/<int:prediction_id>')
@login_required
def download_report(prediction_id):
    """
    Generate and serve a professional PDF medical report for a given prediction.
    Accessible by the patient who owns it, or by doctors and administrators.
    """
    prediction = PredictionHistory.query.get_or_404(prediction_id)

    if current_user.is_patient():
        if prediction.patient_id != current_user.id:
            flash('Access denied. You can only download your own reports.', 'error')
            return redirect(url_for('dashboard.patient_history'))
    elif not current_user.is_doctor() and not current_user.is_admin():
        flash('Access denied.', 'error')
        return redirect(url_for('main.index'))

    try:
        from app.services.report_generator import generate_pdf_report
        target_patient = prediction.patient or current_user
        pdf_buf = generate_pdf_report(prediction, target_patient)
        filename = f'cancer_report_{prediction_id}_{datetime.utcnow().strftime("%Y%m%d")}.pdf'
        return send_file(
            pdf_buf,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'error')
        if current_user.is_patient():
            return redirect(url_for('dashboard.patient_history'))
        return redirect(request.referrer or url_for('dashboard.doctor_dashboard'))


# ─── Analysis Result Page ───────────────────────────────────────────────────
@dashboard_bp.route('/patient/result/<int:image_id>')
@login_required
def analysis_result(image_id):
    """
    Dedicated AI Analysis Results Page.
    Shows full diagnosis details, heatmap, probabilities chart, and recommendations
    for a specific medical image after analysis.
    """
    if not current_user.is_patient():
        flash('Access denied. Patient dashboard only.', 'error')
        return redirect(url_for('main.index'))

    medical_image = MedicalImage.query.get_or_404(image_id)

    # Verify ownership
    if medical_image.patient_id != current_user.id:
        flash('Access denied. You can only view your own results.', 'error')
        return redirect(url_for('dashboard.patient_history'))

    # Get the latest prediction for this image
    prediction = PredictionHistory.query.filter_by(
        medical_image_id=image_id
    ).order_by(PredictionHistory.prediction_date.desc()).first()

    if not prediction:
        flash('No analysis results found for this image. Please run Analyze first.', 'warning')
        return redirect(url_for('dashboard.patient_history'))

    return render_template(
        'dashboard/patient/analysis_result.html',
        user=current_user,
        medical_image=medical_image,
        prediction=prediction,
    )


@dashboard_bp.route('/patient/history')
@login_required
def patient_history():
    """
    Display patient's prediction history and medical images.
    Requires user to be logged in as patient.
    """
    if not current_user.is_patient():
        flash('Access denied. Patient dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    # Get patient's prediction history
    predictions = PredictionHistory.query.filter_by(patient_id=current_user.id).order_by(PredictionHistory.prediction_date.desc()).all()
    
    # Get patient's medical images with predictions loaded (eager loading)
    from sqlalchemy.orm import joinedload
    medical_images = MedicalImage.query.options(joinedload(MedicalImage.predictions)).filter_by(patient_id=current_user.id).order_by(MedicalImage.upload_date.desc()).all()
    
    return render_template('dashboard/patient/history.html', 
                          user=current_user,
                          predictions=predictions,
                          medical_images=medical_images)

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
    pending_appointments = Appointment.query.filter_by(doctor_id=current_user.id, status='pending').count()
    
    return render_template('dashboard/doctor/dashboard.html',
                          user=current_user,
                          total_patients=total_patients,
                          total_predictions=total_predictions,
                          recent_predictions=recent_predictions,
                          pending_appointments=pending_appointments)

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
    clinical_data = ClinicalData.query.filter_by(patient_id=patient_id).order_by(ClinicalData.created_at.desc()).all()
    medical_images = MedicalImage.query.filter_by(patient_id=patient_id).order_by(MedicalImage.upload_date.desc()).all()
    
    return render_template('dashboard/doctor/patient_detail.html',
                          user=current_user,
                          patient=patient,
                          predictions=predictions,
                          medical_records=reports,
                          clinical_data=clinical_data,
                          medical_images=medical_images)

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

@dashboard_bp.route('/doctor/prediction/<int:prediction_id>/add-note', methods=['POST'])
@login_required
def add_doctor_note(prediction_id):
    """
    Doctor adds a clinical review note for a patient's prediction.
    """
    if not current_user.is_doctor():
        flash('Access denied. Doctor access only.', 'error')
        return redirect(url_for('main.index'))

    prediction = PredictionHistory.query.get_or_404(prediction_id)
    note_text = request.form.get('note_text', '').strip()
    status = request.form.get('status', 'approved')

    if not note_text:
        flash('Please provide clinical notes.', 'error')
        return redirect(request.referrer or url_for('dashboard.doctor_patient_detail', patient_id=prediction.patient_id))

    note = DoctorNote(
        prediction_id=prediction.id,
        doctor_id=current_user.id,
        note_text=note_text,
        status=status
    )
    db.session.add(note)
    db.session.commit()

    flash('Clinical review note added successfully!', 'success')
    return redirect(request.referrer or url_for('dashboard.doctor_patient_detail', patient_id=prediction.patient_id))

