from flask import Blueprint, render_template, redirect, url_for

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@main_bp.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

@main_bp.route('/services')
def services():
    """Services page route"""
    return render_template('services.html')

@main_bp.route('/cancer-data')
def cancer_data():
    """Cancer data and challenges page route"""
    return render_template('cancer_data.html')

@main_bp.route('/verify/report/<int:prediction_id>')
def verify_report_direct(prediction_id):
    """Direct alias for QR codes that access /verify/report/<id> without /dashboard prefix."""
    from app.controllers.dashboard import verify_report
    return verify_report(prediction_id)


