import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from app import create_app, db
from app.models import User, Patient, Doctor, Admin, ClinicalData, MedicalImage, PredictionHistory, Report

app = create_app(os.getenv('FLASK_CONFIG', 'development'))

@app.shell_context_processor
def make_shell_context():
    """Make objects available in Flask shell"""
    return {
        'db': db,
        'User': User,
        'Patient': Patient,
        'Doctor': Doctor,
        'ClinicalData': ClinicalData,
        'MedicalImage': MedicalImage,
        'PredictionHistory': PredictionHistory,
        'Report': Report
    }

@app.cli.command()
def init_db():
    """Initialize the database"""
    db.create_all()
    print('Database initialized successfully!')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
