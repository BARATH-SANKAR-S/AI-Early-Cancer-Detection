from flask import Blueprint, render_template, redirect, url_for, request, flash, current_app, send_file, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import Patient, Doctor, ClinicalData, MedicalImage, PredictionHistory, Report, Appointment, DoctorNote
from app.services.vit_inference import predict_image
from app.services.multi_modal_fusion import get_fusion_service, create_modality_score
from app.services.dicom_service import get_dicom_service
from app.services.geographic_service import get_geographic_service
from app.services.annotation_service import get_annotation_service
from app.services.federated_learning import get_federated_learning_service
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from pathlib import Path

dashboard_bp = Blueprint('dashboard', __name__)

# Allowed file extensions for medical images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'dcm'}

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
            
            # Handle DICOM files
            if filename.lower().endswith('.dcm'):
                dicom_service = get_dicom_service()
                if dicom_service.is_available():
                    try:
                        # Read DICOM file
                        dicom_data = dicom_service.read_dicom_file(file_path)
                        if dicom_data:
                            # Convert to PNG for processing
                            converted_path = dicom_service.convert_to_standard_format(
                                dicom_data, 'png', 
                                output_path=file_path.replace('.dcm', '.png')
                            )
                            if converted_path:
                                # Update filename and file path to use converted file
                                filename = filename.replace('.dcm', '.png')
                                file_path = converted_path
                                flash('DICOM file converted successfully.', 'info')
                            else:
                                flash('DICOM conversion failed. Using original file.', 'warning')
                    except Exception as e:
                        flash(f'DICOM processing error: {str(e)}. Using original file.', 'warning')
                else:
                    flash('DICOM service not available. Please install pydicom.', 'warning')
            
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
            flash('Invalid file type. Allowed types: png, jpg, jpeg, gif, bmp, tiff, dcm', 'error')
    
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
            'skin_ct':     'lung_cancer',  # Backward compatibility if previously selected
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
            'raw_heatmap_b64': prediction_results.get('raw_heatmap_b64', None),
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

        # Trigger notifications: email patient, doctor alert if high risk, and in-app alerts
        try:
            from app.services.notification_service import notify_analysis_completed
            notify_analysis_completed(prediction, current_user)
        except Exception as notif_err:
            print(f"[NOTIFY] Non-fatal notification error: {notif_err}", file=sys.stderr)
        
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
        pdf_buf = generate_pdf_report(prediction, target_patient, base_url=request.host_url.rstrip('/'))
        filename = f'cancer_report_{prediction_id}_{datetime.utcnow().strftime("%Y%m%d")}.pdf'

        # Notify that report is generated and ready
        try:
            from app.services.notification_service import notify_report_ready
            notify_report_ready(prediction, target_patient)
        except Exception as notif_err:
            print(f"[NOTIFY] Report ready notice warning: {notif_err}", file=sys.stderr)

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


# ─── Public QR Verification Route ───────────────────────────────────────────
@dashboard_bp.route('/verify/report/<int:prediction_id>')
def verify_report(prediction_id):
    """
    Public QR code verification endpoint for clinical reports.
    Provides cryptographic tamper-proof validation of diagnostic findings.
    """
    import hashlib
    prediction = PredictionHistory.query.get_or_404(prediction_id)
    patient = prediction.patient

    # Generate matching verification hash
    verification_hash = hashlib.sha256(
        f"RPT-{prediction.id}-{patient.id}-{prediction.confidence_score}".encode()
    ).hexdigest()[:16].upper()

    return render_template(
        'dashboard/patient/report_view.html',
        prediction=prediction,
        patient=patient,
        verification_hash=verification_hash
    )


# ─── Longitudinal Patient Tracking Timeline ──────────────────────────────────
@dashboard_bp.route('/patient/timeline')
@login_required
def patient_timeline():
    """
    Longitudinal Patient Health Timeline.
    Tracks diagnostic scans, confidence scores, and risk progression over time.
    """
    if not current_user.is_patient():
        flash('Access denied. Patient dashboard only.', 'error')
        return redirect(url_for('main.index'))

    # Fetch predictions chronologically
    predictions = PredictionHistory.query.filter_by(patient_id=current_user.id)\
        .order_by(PredictionHistory.prediction_date.asc()).all()

    # Calculate trends
    total_scans = len(predictions)
    latest_prediction = predictions[-1] if predictions else None
    prev_prediction = predictions[-2] if total_scans >= 2 else None

    trend_delta = None
    trend_direction = 'stable'
    if latest_prediction and prev_prediction:
        trend_delta = round((latest_prediction.confidence_score - prev_prediction.confidence_score) * 100, 1)
        if trend_delta > 1.5:
            trend_direction = 'increasing'
        elif trend_delta < -1.5:
            trend_direction = 'decreasing'

    # Risk breakdown
    high_risk_count = sum(1 for p in predictions if str(p.risk_level).lower() == 'high')
    medium_risk_count = sum(1 for p in predictions if str(p.risk_level).lower() == 'medium')
    low_risk_count = sum(1 for p in predictions if str(p.risk_level).lower() == 'low')

    # Prepare timeline data for Chart.js
    timeline_labels = [
        p.prediction_date.strftime('%b %d, %Y') if p.prediction_date else f"Scan #{i+1}"
        for i, p in enumerate(predictions)
    ]
    timeline_confidences = [round(p.confidence_score * 100, 1) for p in predictions]
    timeline_risks = [str(p.risk_level or 'low').lower() for p in predictions]

    return render_template(
        'dashboard/patient/timeline.html',
        predictions=predictions,
        total_scans=total_scans,
        latest_prediction=latest_prediction,
        trend_delta=trend_delta,
        trend_direction=trend_direction,
        high_risk_count=high_risk_count,
        medium_risk_count=medium_risk_count,
        low_risk_count=low_risk_count,
        timeline_labels=timeline_labels,
        timeline_confidences=timeline_confidences,
        timeline_risks=timeline_risks
    )


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


@dashboard_bp.route('/patient/fusion-analysis/<int:prediction_id>')
@login_required
def fusion_analysis(prediction_id):
    """
    Multi-modal fusion analysis page.
    Shows combined fusion score with visual breakdown of modality contributions.
    """
    if not current_user.is_patient():
        flash('Access denied. Patient dashboard only.', 'error')
        return redirect(url_for('main.index'))

    prediction = PredictionHistory.query.get_or_404(prediction_id)

    # Verify ownership
    if prediction.patient_id != current_user.id:
        flash('Access denied. You can only view your own results.', 'error')
        return redirect(url_for('dashboard.patient_history'))

    # Get related clinical data if available
    clinical_data = None
    if prediction.clinical_data_id:
        clinical_data = ClinicalData.query.get(prediction.clinical_data_id)

    # Create modality scores for fusion
    modality_scores = []
    
    # Image modality score (from prediction)
    if prediction.medical_image_id:
        image_score = create_modality_score(
            modality_type='image',
            confidence=prediction.confidence_score,
            prediction=prediction.prediction_result,
            metadata={
                'cancer_type': prediction.cancer_type,
                'image_id': prediction.medical_image_id
            }
        )
        modality_scores.append(image_score)
    
    # Clinical modality score (if available)
    if clinical_data:
        # Calculate clinical risk score based on symptoms and family history
        clinical_risk = _calculate_clinical_risk_score(clinical_data)
        clinical_score = create_modality_score(
            modality_type='clinical',
            confidence=clinical_risk,
            prediction='high_risk' if clinical_risk > 0.6 else 'low_risk',
            metadata={
                'symptoms': clinical_data.symptoms,
                'family_history': clinical_data.family_history,
                'lifestyle_factors': clinical_data.lifestyle_factors
            }
        )
        modality_scores.append(clinical_score)

    # Calculate fusion score
    fusion_result = None
    if modality_scores:
        fusion_service = get_fusion_service()
        clinical_context = {
            'family_history': clinical_data.family_history if clinical_data else None,
            'age': current_user.age if hasattr(current_user, 'age') else None
        }
        fusion_result = fusion_service.calculate_fusion_score(
            modality_scores, clinical_context
        )

    return render_template(
        'dashboard/patient/fusion_analysis.html',
        user=current_user,
        prediction=prediction,
        clinical_data=clinical_data,
        fusion_result=fusion_result,
        modality_scores=modality_scores
    )


def _calculate_clinical_risk_score(clinical_data) -> float:
    """
    Calculate clinical risk score from clinical data.
    Simple heuristic based on symptoms, family history, and lifestyle factors.
    
    Args:
        clinical_data: ClinicalData object
        
    Returns:
        Risk score between 0.0 and 1.0
    """
    risk_score = 0.0
    
    # Family history contribution
    family_history = (clinical_data.family_history or '').lower()
    if family_history == 'multiple':
        risk_score += 0.3
    elif family_history in ['parent', 'sibling']:
        risk_score += 0.2
    elif family_history == 'grandparent':
        risk_score += 0.1
    
    # Lifestyle factors contribution
    lifestyle_factors = clinical_data.lifestyle_factors or []
    if 'smoking' in lifestyle_factors:
        risk_score += 0.2
    if 'alcohol' in lifestyle_factors:
        risk_score += 0.1
    if 'sedentary' in lifestyle_factors:
        risk_score += 0.05
    if 'poor_diet' in lifestyle_factors:
        risk_score += 0.05
    
    # Symptoms contribution (basic keyword analysis)
    symptoms = (clinical_data.symptoms or '').lower()
    high_risk_keywords = ['blood', 'pain', 'lump', 'weight loss', 'fatigue']
    for keyword in high_risk_keywords:
        if keyword in symptoms:
            risk_score += 0.1
    
    # Cap at 1.0
    return min(risk_score, 1.0)


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


@dashboard_bp.route('/patient/geographic-risk')
@login_required
def geographic_risk():
    """
    Geographic cancer risk heatmap for Erode district.
    Shows regional statistics and risk analysis.
    """
    if not current_user.is_patient():
        flash('Access denied. Patient dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    geographic_service = get_geographic_service()
    heatmap_data = geographic_service.generate_heatmap_data()
    
    # Get patient's specific region if available
    patient_region = getattr(current_user, 'district', 'Erode')
    regional_stats = geographic_service.get_regional_statistics(patient_region)
    regional_recommendations = geographic_service.get_regional_recommendations(patient_region)
    
    return render_template('dashboard/patient/geographic_risk.html',
                          user=current_user,
                          heatmap_data=heatmap_data,
                          regional_stats=regional_stats,
                          regional_recommendations=regional_recommendations,
                          patient_region=patient_region)


@dashboard_bp.route('/api/geographic/heatmap-data')
@login_required
def api_heatmap_data():
    """
    API endpoint to get heatmap data for client-side rendering.
    """
    try:
        geographic_service = get_geographic_service()
        heatmap_data = geographic_service.generate_heatmap_data()
        
        return jsonify({
            'success': True,
            'data': {
                'regions': heatmap_data.regions,
                'overall_stats': heatmap_data.overall_stats,
                'temporal_data': heatmap_data.temporal_data,
                'risk_factors': heatmap_data.risk_factors
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting heatmap data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@dashboard_bp.route('/api/geographic/regional-stats/<region_name>')
@login_required
def api_regional_stats(region_name):
    """
    API endpoint to get statistics for a specific region.
    """
    try:
        geographic_service = get_geographic_service()
        regional_stats = geographic_service.get_regional_statistics(region_name)
        
        if not regional_stats:
            return jsonify({
                'success': False,
                'error': f'Region {region_name} not found'
            }), 404
        
        recommendations = geographic_service.get_regional_recommendations(region_name)
        risk_score = geographic_service._calculate_regional_risk_score(regional_stats)
        risk_level = geographic_service.get_risk_level(risk_score)
        
        return jsonify({
            'success': True,
            'data': {
                'region_name': regional_stats.region_name,
                'total_predictions': regional_stats.total_predictions,
                'high_risk_count': regional_stats.high_risk_count,
                'medium_risk_count': regional_stats.medium_risk_count,
                'low_risk_count': regional_stats.low_risk_count,
                'avg_confidence': regional_stats.avg_confidence,
                'most_common_cancer_type': regional_stats.most_common_cancer_type,
                'trend_direction': regional_stats.trend_direction,
                'risk_score': risk_score,
                'risk_level': risk_level.value,
                'recommendations': recommendations
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting regional stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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


@dashboard_bp.route('/doctor/annotation/<int:prediction_id>', methods=['GET', 'POST'])
@login_required
def doctor_annotation(prediction_id):
    """
    Doctor annotation interface for AI predictions.
    Allows doctors to review and annotate AI predictions.
    """
    if not current_user.is_doctor():
        flash('Access denied. Doctor dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    prediction = PredictionHistory.query.get_or_404(prediction_id)
    
    if request.method == 'POST':
        annotation_service = get_annotation_service()
        
        try:
            annotation_type = request.form.get('annotation_type')
            corrected_label = request.form.get('corrected_label')
            confidence_level = request.form.get('confidence_level', 'medium')
            clinical_notes = request.form.get('clinical_notes', '')
            
            # Create annotation
            annotation = annotation_service.create_annotation(
                prediction_id=prediction_id,
                doctor_id=current_user.id,
                annotation_type=annotation_type,
                corrected_label=corrected_label,
                confidence_level=confidence_level,
                clinical_notes=clinical_notes
            )
            
            # Process for active learning
            active_learning_data = annotation_service.process_annotation_for_active_learning(
                prediction_id=prediction_id,
                original_prediction=prediction.prediction_result,
                prediction_confidence=prediction.confidence_score,
                doctor_annotation=annotation
            )
            
            # Save annotation to database
            doctor_note = DoctorNote(
                prediction_id=prediction_id,
                doctor_id=current_user.id,
                note_text=clinical_notes,
                status=annotation_type
            )
            db.session.add(doctor_note)
            db.session.commit()
            
            flash('Annotation saved successfully. Thank you for your feedback.', 'success')
            return redirect(url_for('dashboard.doctor_patients'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error saving annotation: {str(e)}', 'error')
    
    return render_template('dashboard/doctor/annotation.html',
                          user=current_user,
                          prediction=prediction)


@dashboard_bp.route('/doctor/annotation-dashboard')
@login_required
def annotation_dashboard():
    """
    Doctor annotation dashboard showing statistics and high-priority cases.
    Loads real stats from the DoctorNote database table.
    """
    if not current_user.is_doctor():
        flash('Access denied. Doctor dashboard only.', 'error')
        return redirect(url_for('main.index'))

    # ── Pull real statistics from DB ────────────────────────────────────
    from sqlalchemy import func
    from app.models import DoctorNote

    all_notes = DoctorNote.query.all()
    total_annotations = len(all_notes)

    # Count by annotation type (status field stores the type)
    by_type = {}
    for note in all_notes:
        t = note.status or 'unknown'
        by_type[t] = by_type.get(t, 0) + 1

    # We don't store confidence in DB — use placeholder breakdown
    by_confidence = {'high': 0, 'medium': 0, 'low': 0}

    # Build high-priority list: notes where doctor flagged a correction or rescan
    high_priority_notes = DoctorNote.query.filter(
        DoctorNote.status.in_(['correction', 'rescan'])
    ).order_by(DoctorNote.created_at.desc()).limit(20).all()

    # Format into display-friendly dicts
    high_priority_list = []
    for note in high_priority_notes:
        pred = note.prediction
        if pred:
            high_priority_list.append({
                'prediction_id': note.prediction_id,
                'original_prediction': pred.prediction_result or 'N/A',
                'annotation_type': note.status or 'correction',
                'disagreement_score': 0.8 if note.status == 'correction' else 0.7,
                'learning_value': 0.85 if note.status == 'correction' else 0.75,
                'clinical_notes': note.note_text or '',
                'created_at': note.created_at,
            })

    statistics = {
        'total_annotations': total_annotations,
        'by_type': by_type,
        'by_confidence': by_confidence,
        'avg_disagreement': round(
            sum(0.8 if n.status == 'correction' else
                0.7 if n.status == 'rescan' else
                0.5 if n.status == 'uncertain' else 0.1
                for n in all_notes) / total_annotations, 2
        ) if total_annotations else 0.0,
        'high_value_annotations': len([n for n in all_notes if n.status in ('correction', 'rescan')]),
        'learning_queue_size': total_annotations,
    }

    return render_template('dashboard/doctor/annotation_dashboard.html',
                           user=current_user,
                           statistics=statistics,
                           high_priority_annotations=high_priority_list)


@dashboard_bp.route('/api/doctor/annotation', methods=['POST'])
@login_required
def api_create_annotation():
    """
    API endpoint for creating doctor annotations.
    """
    if not current_user.is_doctor():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        data = request.get_json()
        annotation_service = get_annotation_service()
        
        annotation = annotation_service.create_annotation(
            prediction_id=data['prediction_id'],
            doctor_id=current_user.id,
            annotation_type=data['annotation_type'],
            corrected_label=data.get('corrected_label'),
            confidence_level=data.get('confidence_level', 'medium'),
            clinical_notes=data.get('clinical_notes', ''),
            additional_data=data.get('additional_data')
        )
        
        return jsonify({
            'success': True,
            'annotation_id': id(annotation),
            'message': 'Annotation created successfully'
        })
        
    except Exception as e:
        logger.error(f"Error creating annotation via API: {e}")
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/api/doctor/annotation-statistics')
@login_required
def api_annotation_statistics():
    """
    API endpoint for getting annotation statistics.
    """
    if not current_user.is_doctor():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        annotation_service = get_annotation_service()
        statistics = annotation_service.get_annotation_statistics()
        
        return jsonify({
            'success': True,
            'statistics': statistics
        })
        
    except Exception as e:
        logger.error(f"Error getting annotation statistics: {e}")
        return jsonify({'error': str(e)}), 500

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


@dashboard_bp.route('/admin/federated-learning')
@login_required
def federated_learning_dashboard():
    """
    Federated learning dashboard for administrators.
    Shows federated learning progress and hospital participation.
    """
    if not current_user.is_admin():
        flash('Access denied. Admin dashboard only.', 'error')
        return redirect(url_for('main.index'))
    
    federated_service = get_federated_learning_service()
    summary = federated_service.get_federated_learning_summary()
    
    return render_template('dashboard/admin/federated_learning.html',
                          user=current_user,
                          summary=summary)


@dashboard_bp.route('/admin/federated-learning/run-round', methods=['POST'])
@login_required
def run_federated_round():
    """
    Run a single federated learning round.
    """
    if not current_user.is_admin():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        federated_service = get_federated_learning_service()
        round_result = federated_service.run_federated_round()
        
        return jsonify({
            'success': True,
            'round_result': {
                'round_number': round_result.round_number,
                'participating_hospitals': round_result.participating_hospitals,
                'global_model_accuracy': round_result.global_model_accuracy,
                'privacy_loss': round_result.privacy_loss,
                'convergence_metrics': round_result.convergence_metrics
            }
        })
        
    except Exception as e:
        logger.error(f"Error running federated round: {e}")
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/api/admin/federated-learning/summary')
@login_required
def api_federated_summary():
    """
    API endpoint for federated learning summary.
    """
    if not current_user.is_admin():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        federated_service = get_federated_learning_service()
        summary = federated_service.get_federated_learning_summary()
        
        return jsonify({
            'success': True,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error getting federated summary: {e}")
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/api/admin/federated-learning/export')
@login_required
def export_federated_data():
    """
    Export federated learning data.
    """
    if not current_user.is_admin():
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        federated_service = get_federated_learning_service()
        data = federated_service.export_federated_data()
        
        return jsonify({
            'success': True,
            'data': data
        })
        
    except Exception as e:
        logger.error(f"Error exporting federated data: {e}")
        return jsonify({'error': str(e)}), 500

