from app import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

class User(db.Model, UserMixin):
    """
    User model for authentication and patient management.
    Inherits from UserMixin for Flask-Login integration.
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    district = db.Column(db.String(100), default='Erode')
    role = db.Column(db.String(20), default='patient')  # patient, doctor, admin
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    medical_records = db.relationship('MedicalRecord', backref='user', lazy=True, cascade='all, delete-orphan')
    
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

class Prediction(db.Model):
    """Model for storing cancer prediction results"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prediction_type = db.Column(db.String(50), nullable=False)  # lung, breast, skin, etc.
    prediction_result = db.Column(db.String(20), nullable=False)  # benign, malignant
    confidence_score = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20))  # low, medium, high
    image_path = db.Column(db.String(500))
    symptoms_data = db.Column(db.JSON)
    additional_notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Prediction {self.id} - {self.prediction_type}>'

class MedicalRecord(db.Model):
    """Model for storing patient medical history"""
    __tablename__ = 'medical_records'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    record_type = db.Column(db.String(50), nullable=False)  # lab_report, scan, prescription
    description = db.Column(db.Text)
    file_path = db.Column(db.String(500))
    doctor_notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<MedicalRecord {self.id}>'
