from app import create_app, db
from app.models import User, Prediction, MedicalRecord
import os

app = create_app(os.getenv('FLASK_CONFIG', 'development'))

@app.shell_context_processor
def make_shell_context():
    """Make objects available in Flask shell"""
    return {'db': db, 'User': User, 'Prediction': Prediction, 'MedicalRecord': MedicalRecord}

@app.cli.command()
def init_db():
    """Initialize the database"""
    db.create_all()
    print('Database initialized successfully!')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
