"""
Notification Service
Sends email alerts for high-risk cancer predictions.
Uses Flask-Mail for email delivery.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def notify_if_high_risk(prediction, patient, app=None):
    """
    Check if a prediction is high-risk and send notifications.

    Args:
        prediction: PredictionHistory object
        patient:    Patient user object
        app:        Flask app (for context, optional)
    """
    risk = (prediction.risk_level or '').lower()
    additional = prediction.additional_data or {}
    is_malignant = additional.get('is_malignant', False)

    # Trigger on high risk OR malignant result
    should_notify = risk == 'high' or is_malignant or prediction.confidence_score >= 0.85

    if not should_notify:
        logger.info(f'Prediction {prediction.id}: low risk, no notification sent.')
        return False

    # Send email
    try:
        _send_email_alert(prediction, patient)
    except Exception as e:
        logger.warning(f'Email notification failed for prediction {prediction.id}: {e}')

    return True


def _send_email_alert(prediction, patient):
    """Send high-risk email alert to patient (and optionally doctor)."""
    from flask import current_app
    from flask_mail import Message

    cancer_type  = (prediction.cancer_type or 'cancer').replace('_', ' ').title()
    confidence   = f'{prediction.confidence_score * 100:.1f}%'
    pred_result  = prediction.prediction_result or 'detected'
    pred_date    = prediction.prediction_date.strftime('%Y-%m-%d %H:%M UTC') if prediction.prediction_date else 'recently'
    report_url   = f'{current_app.config.get("SERVER_NAME", "http://localhost:5000")}/patient/history'

    subject = f'⚠️ AI Alert: High-Risk {cancer_type} Finding — Immediate Attention Required'

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; background: #f8fafc; margin: 0; padding: 20px; }}
            .card {{ background: white; border-radius: 12px; padding: 30px; max-width: 600px; margin: auto;
                     box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
            .header {{ background: #dc2626; color: white; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 24px; }}
            .header h1 {{ margin: 0; font-size: 22px; }}
            .badge {{ display: inline-block; padding: 6px 14px; border-radius: 20px;
                      background: #fee2e2; color: #dc2626; font-weight: bold; font-size: 14px; }}
            .info-row {{ display: flex; justify-content: space-between; border-bottom: 1px solid #e2e8f0;
                         padding: 10px 0; font-size: 14px; }}
            .label {{ color: #64748b; font-weight: 500; }}
            .value {{ color: #1e293b; font-weight: 600; }}
            .cta {{ background: #1a56db; color: white; text-decoration: none;
                    padding: 12px 24px; border-radius: 8px; display: inline-block;
                    margin-top: 20px; font-weight: bold; }}
            .disclaimer {{ color: #94a3b8; font-size: 12px; margin-top: 20px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="header">
                <h1>⚠️ High-Risk AI Finding Detected</h1>
                <p style="margin:8px 0 0 0; font-size:14px; opacity:0.9;">
                    Please consult a doctor immediately
                </p>
            </div>

            <p>Dear <strong>{patient.full_name}</strong>,</p>
            <p>Our AI system has detected a <strong>high-risk finding</strong> in your recent medical image analysis.
               Please review the details below and consult your doctor as soon as possible.</p>

            <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:16px; margin:16px 0;">
                <div class="info-row"><span class="label">Cancer Type Screened</span><span class="value">{cancer_type}</span></div>
                <div class="info-row"><span class="label">AI Prediction</span><span class="value">{pred_result.replace('_',' ').title()}</span></div>
                <div class="info-row"><span class="label">Confidence Score</span><span class="value">{confidence}</span></div>
                <div class="info-row"><span class="label">Risk Level</span><span class="value"><span class="badge">HIGH RISK</span></span></div>
                <div class="info-row"><span class="label">Analysis Date</span><span class="value">{pred_date}</span></div>
            </div>

            <p><strong>⚕️ Next Steps:</strong></p>
            <ol style="font-size:14px; color:#475569; line-height:1.8;">
                <li>Log in to view your full AI analysis report with Grad-CAM heatmap</li>
                <li>Book an appointment with your doctor through the platform</li>
                <li>Download your PDF medical report to share with your doctor</li>
            </ol>

            <div style="text-align:center;">
                <a href="{report_url}" class="cta">📋 View My Report</a>
            </div>

            <p class="disclaimer">
                ⚠️ DISCLAIMER: This is an AI-generated alert for informational purposes only.
                It does not constitute a medical diagnosis. Always consult a qualified healthcare
                professional for proper evaluation and treatment.
            </p>
        </div>
    </body>
    </html>
    """

    try:
        mail = current_app.extensions.get('mail')
        if mail and patient.email:
            msg = Message(
                subject=subject,
                recipients=[patient.email],
                html=html_body,
                sender=current_app.config.get('MAIL_DEFAULT_SENDER', 'noreply@cancerdetection.ai')
            )
            mail.send(msg)
            logger.info(f'High-risk alert email sent to {patient.email} for prediction {prediction.id}')
        else:
            logger.info(f'Mail not configured or no patient email. Skipping email for prediction {prediction.id}')
    except Exception as e:
        logger.warning(f'Failed to send email alert: {e}')
        raise
