from app import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

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
