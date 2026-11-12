from app import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

# Export all models for easy importing
__all__ = [
    'User', 'Patient', 'Doctor', 'Admin',
    'ClinicalData', 'MedicalImage', 'PredictionHistory',
    'Report', 'DoctorNote', 'Appointment', 'ChatMessage',
    'Notification'
]

# ==================== BASE USER MODEL ====================
class User(db.Model, UserMixin):
    """
    Base User model for authentication.
    This is the parent class for both Patient and Doctor models.
    Inherits from UserMixin for Flask-Login integration.
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # patient, doctor, admin
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Polymorphic relationship - allows accessing Patient or Doctor from User
    __mapper_args__ = {
        'polymorphic_on': role,
        'polymorphic_identity': 'user'
    }
    
    def set_password(self, password):
        """
        Hash and set the user's password.
        
        Args:
            password (str): Plain text password
        """
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """
        Verify if the provided password matches the stored hash.
        
        Args:
            password (str): Plain text password to verify
            
        Returns:
            bool: True if password matches, False otherwise
        """
        return check_password_hash(self.password_hash, password)
    
    def is_doctor(self):
        """Check if user has doctor role"""
        return self.role == 'doctor'
    
    def is_patient(self):
        """Check if user has patient role"""
        return self.role == 'patient'
    
    def is_admin(self):
        """Check if user has admin role"""
        return self.role == 'admin'
    
    def __repr__(self):
        return f'<User {self.username} - {self.role}>'

# ==================== PATIENT MODEL ====================
class Patient(User):
    """
    Patient model extending User.
    Contains patient-specific information and relationships.
    """
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    district = db.Column(db.String(100), default='Erode')
    blood_group = db.Column(db.String(5))
    emergency_contact = db.Column(db.String(100))
    emergency_phone = db.Column(db.String(20))
    
    # Relationships
    clinical_data = db.relationship('ClinicalData', backref='patient', lazy=True, cascade='all, delete-orphan')
    medical_images = db.relationship('MedicalImage', backref='patient', lazy=True, cascade='all, delete-orphan')
    prediction_history = db.relationship('PredictionHistory', backref='patient', lazy=True, cascade='all, delete-orphan')
    reports = db.relationship('Report', backref='patient', lazy=True, cascade='all, delete-orphan')
    
    __mapper_args__ = {
        'polymorphic_identity': 'patient'
    }
    
    def __repr__(self):
        return f'<Patient {self.full_name}>'

# ==================== DOCTOR MODEL ====================
class Doctor(User):
    """
    Doctor model extending User.
    Contains doctor-specific information and relationships.
    """
    __tablename__ = 'doctors'
    
    id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)
    specialization = db.Column(db.String(100))
    license_number = db.Column(db.String(50), unique=True)
    hospital_name = db.Column(db.String(200))
    hospital_address = db.Column(db.Text)
    phone = db.Column(db.String(20))
    years_of_experience = db.Column(db.Integer)
    qualification = db.Column(db.String(200))
    
    # Relationships
    reports_authored = db.relationship('Report', backref='doctor', lazy=True, cascade='all, delete-orphan')
    
    __mapper_args__ = {
        'polymorphic_identity': 'doctor'
    }
    
    def __repr__(self):
        return f'<Dr. {self.full_name} - {self.specialization}>'

# ==================== ADMIN MODEL ====================
class Admin(User):
    """
    Admin model extending User via single-table inheritance.
    Admins live solely in the 'users' table (no separate table needed).
    """
    __mapper_args__ = {
        'polymorphic_identity': 'admin'
    }

    def __repr__(self):
        return f'<Admin {self.username}>'

# ==================== CLINICAL DATA MODEL ====================
class ClinicalData(db.Model):
    """
    Model for storing clinical data submitted by patients.
    Includes symptoms, family history, and lifestyle factors.
    """
    __tablename__ = 'clinical_data'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    family_history = db.Column(db.String(50))  # none, parent, sibling, grandparent, multiple
    lifestyle_factors = db.Column(db.JSON)  # smoking, alcohol, sedentary, poor_diet
    additional_notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to prediction history
    predictions = db.relationship('PredictionHistory', backref='clinical_data', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<ClinicalData {self.id} - Patient {self.patient_id}>'

# ==================== MEDICAL IMAGES MODEL ====================
class MedicalImage(db.Model):
    """
    Model for storing medical image uploads.
    Includes X-rays, CT scans, MRI, etc.
    """
    __tablename__ = 'medical_images'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    image_type = db.Column(db.String(50), nullable=False)  # xray, ct_scan, mri, ultrasound, mammography
    file_path = db.Column(db.String(500), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)  # in bytes
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    description = db.Column(db.Text)
    
    # Relationship to prediction history
    predictions = db.relationship('PredictionHistory', backref='medical_image', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<MedicalImage {self.id} - {self.image_type}>'

# ==================== PREDICTION HISTORY MODEL ====================
class PredictionHistory(db.Model):
    """
    Model for storing cancer prediction results.
    Links to clinical data and/or medical images.
    """
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    cancer_type = db.Column(db.String(50), nullable=False)  # lung, breast, skin, etc.
    prediction_result = db.Column(db.String(20), nullable=False)  # benign, malignant
    confidence_score = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    risk_level = db.Column(db.String(20))  # low, medium, high
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys to clinical data and medical image (optional)
    clinical_data_id = db.Column(db.Integer, db.ForeignKey('clinical_data.id'))
    medical_image_id = db.Column(db.Integer, db.ForeignKey('medical_images.id'))
    
    # Additional data
    model_version = db.Column(db.String(50))  # AI model version used
    processing_time = db.Column(db.Float)  # in seconds
    additional_data = db.Column(db.JSON)
    
    @property
    def effective_risk_level(self):
        """
        Returns the resolved risk level ('low', 'medium', 'high').
        Infers clinically from prediction_result and confidence_score if not explicitly set.
        """
        if self.risk_level:
            return self.risk_level.lower().strip()
        if self.additional_data and isinstance(self.additional_data, dict) and self.additional_data.get('risk_level'):
            return str(self.additional_data['risk_level']).lower().strip()

        # Fallback inference
        result = (self.prediction_result or '').lower().strip()
        ct = (self.cancer_type or '').lower().strip()
        conf = float(self.confidence_score or 0.0)

        if 'brain' in ct:
            if result in ('glioma', 'malignant'):
                return 'high'
            elif result in ('meningioma', 'pituitary', 'benign', 'nodule'):
                return 'medium'
            elif result == 'tumor':
                return 'high' if conf >= 0.75 else 'medium'
            elif result in ('normal', 'notumor'):
                return 'low' if conf >= 0.70 else 'medium'
            return 'medium' if conf >= 0.60 else 'low'
        elif 'lung' in ct:
            if result == 'malignant':
                return 'high'
            elif result == 'benign':
                return 'medium'
            return 'low'
        elif 'skin' in ct:
            if result in ('melanoma', 'bcc'):
                return 'high'
            elif result == 'akiec':
                return 'medium'
            return 'low'
        return 'low'

    def __repr__(self):
        return f'<PredictionHistory {self.id} - {self.cancer_type}>'

# ==================== REPORTS MODEL ====================
class Report(db.Model):
    """
    Model for storing medical reports generated by doctors.
    Includes diagnosis, recommendations, and doctor notes.
    """
    __tablename__ = 'reports'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    report_type = db.Column(db.String(50), nullable=False)  # diagnosis, follow_up, consultation
    title = db.Column(db.String(200), nullable=False)
    diagnosis = db.Column(db.Text)
    recommendations = db.Column(db.Text)
    doctor_notes = db.Column(db.Text)
    report_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Link to prediction if applicable
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction_history.id'))
    
    # Report status
    status = db.Column(db.String(20), default='draft')  # draft, final, archived
    
    def __repr__(self):
        return f'<Report {self.id} - {self.title}>'

# ==================== DOCTOR NOTE MODEL ====================
class DoctorNote(db.Model):
    """
    Doctor's clinical review note on an AI prediction.
    Allows doctors to approve, flag, or add clinical observations.
    """
    __tablename__ = 'doctor_notes'

    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction_history.id'), nullable=False)
    doctor_id     = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    note_text     = db.Column(db.Text)
    status        = db.Column(db.String(20), default='pending')  # pending / approved / flagged / rescan
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at    = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    prediction = db.relationship('PredictionHistory', backref='doctor_notes', lazy=True)
    doctor     = db.relationship('Doctor', backref='notes_written', lazy=True)

    def __repr__(self):
        return f'<DoctorNote {self.id} - {self.status}>'


# ==================== APPOINTMENT MODEL ====================
class Appointment(db.Model):
    """
    Appointment booking between patient and doctor.
    Can be linked to a specific AI prediction for follow-up.
    """
    __tablename__ = 'appointments'

    id           = db.Column(db.Integer, primary_key=True)
    patient_id   = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    doctor_id    = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction_history.id'), nullable=True)
    scheduled_at = db.Column(db.DateTime, nullable=False)
    reason       = db.Column(db.Text)
    notes        = db.Column(db.Text)
    status       = db.Column(db.String(20), default='pending')  # pending / confirmed / cancelled / completed
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at   = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient     = db.relationship('Patient', backref='appointments', lazy=True)
    doctor      = db.relationship('Doctor', backref='appointments', lazy=True)
    prediction  = db.relationship('PredictionHistory', backref='appointments', lazy=True)

    def __repr__(self):
        return f'<Appointment {self.id} - {self.status}>'

# ==================== CHAT MESSAGE MODEL ====================
class ChatMessage(db.Model):
    """
    Model for storing chatbot conversation history.
    Maintains chat logs with retention policy for privacy.
    """
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    session_id = db.Column(db.String(100), nullable=True, index=True)
    query = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    sources = db.Column(db.JSON)  # Array of source citations
    citations = db.Column(db.Text)  # Formatted citation string
    is_cancer_related = db.Column(db.Boolean, default=True)
    has_sources = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # Relationship
    user = db.relationship('User', backref='chat_messages', lazy=True)

    def __repr__(self):
        return f'<ChatMessage {self.id} - User {self.user_id}>'


# ==================== NOTIFICATION MODEL ====================
class Notification(db.Model):
    """
    Model for storing in-app notifications for users.
    Supports different notification types and read status tracking.
    """
    __tablename__ = 'notifications'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    notification_type = db.Column(db.String(20), default='info')  # alert, info, success, warning
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Optional link to related content
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction_history.id'), nullable=True)
    appointment_id = db.Column(db.Integer, db.ForeignKey('appointments.id'), nullable=True)
    
    # Relationships
    user = db.relationship('User', backref='notifications', lazy=True)
    prediction = db.relationship('PredictionHistory', backref='notifications', lazy=True)
    appointment = db.relationship('Appointment', backref='notifications', lazy=True)

    def __repr__(self):
        return f'<Notification {self.id} - {self.title}>'
