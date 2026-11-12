"""
Notification Service
Sends email, SMS, and WhatsApp alerts for high-risk cancer predictions.
Uses Flask-Mail for email delivery and Twilio for SMS/WhatsApp.
"""

import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def notify_if_high_risk(prediction, patient, app=None):
    """
    Check if a prediction is high-risk and send notifications via multiple channels.

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

    # Send notifications via multiple channels
    notification_results = {
        'email': False,
        'sms': False,
        'whatsapp': False,
        'in_app': False
    }

    # Send email
    try:
        _send_email_alert(prediction, patient)
        notification_results['email'] = True
    except Exception as e:
        logger.warning(f'Email notification failed for prediction {prediction.id}: {e}')

    # Send SMS
    try:
        _send_sms_alert(prediction, patient)
        notification_results['sms'] = True
    except Exception as e:
        logger.warning(f'SMS notification failed for prediction {prediction.id}: {e}')

    # Send WhatsApp
    try:
        _send_whatsapp_alert(prediction, patient)
        notification_results['whatsapp'] = True
    except Exception as e:
        logger.warning(f'WhatsApp notification failed for prediction {prediction.id}: {e}')

    # Send in-app notification
    try:
        cancer_type = (prediction.cancer_type or 'cancer').replace('_', ' ').title()
        confidence = f'{prediction.confidence_score * 100:.1f}%'
        
        send_in_app_notification(
            user_id=patient.id,
            title=f'High-Risk {cancer_type} Alert',
            message=f'AI detected high-risk finding with {confidence} confidence. Please consult your doctor immediately.',
            notification_type='alert'
        )
        notification_results['in_app'] = True
    except Exception as e:
        logger.warning(f'In-app notification failed for prediction {prediction.id}: {e}')

    logger.info(f'Notification results for prediction {prediction.id}: {notification_results}')
    return any(notification_results.values())


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
        mail_username = current_app.config.get('MAIL_USERNAME')
        if mail and mail_username and patient.email:
            msg = Message(
                subject=subject,
                recipients=[patient.email],
                html=html_body,
                sender=current_app.config.get('MAIL_DEFAULT_SENDER', 'noreply@cancerdetection.ai')
            )
            mail.send(msg)
            logger.info(f'High-risk alert email sent to {patient.email} for prediction {prediction.id}')
        else:
            logger.info(f'Mail credentials not configured or no patient email. Skipping email for prediction {prediction.id}')
    except Exception as e:
        logger.warning(f'Failed to send email alert: {e}')


def _send_sms_alert(prediction, patient):
    """Send high-risk SMS alert to patient using Twilio."""
    try:
        from flask import current_app
        from twilio.rest import Client

        # Check if Twilio is configured
        account_sid = current_app.config.get('TWILIO_ACCOUNT_SID')
        auth_token = current_app.config.get('TWILIO_AUTH_TOKEN')
        from_number = current_app.config.get('TWILIO_PHONE_NUMBER')

        if not all([account_sid, auth_token, from_number]):
            logger.info('Twilio not configured. Skipping SMS notification.')
            return

        if not patient.phone:
            logger.info(f'No phone number for patient {patient.id}. Skipping SMS.')
            return

        cancer_type = (prediction.cancer_type or 'cancer').replace('_', ' ').title()
        confidence = f'{prediction.confidence_score * 100:.1f}%'
        report_url = f'{current_app.config.get("SERVER_NAME", "http://localhost:5000")}/patient/history'

        message = (
            f"⚠️ HIGH-RISK ALERT: AI detected high-risk {cancer_type} finding "
            f"with {confidence} confidence. Please consult your doctor immediately. "
            f"View report: {report_url}"
        )

        client = Client(account_sid, auth_token)
        client.messages.create(
            body=message,
            from_=from_number,
            to=patient.phone
        )

        logger.info(f'SMS alert sent to {patient.phone} for prediction {prediction.id}')

    except ImportError:
        logger.warning('Twilio library not installed. Install with: pip install twilio')
    except Exception as e:
        logger.warning(f'Failed to send SMS alert: {e}')
        raise


def _send_whatsapp_alert(prediction, patient):
    """Send high-risk WhatsApp alert to patient using Twilio WhatsApp API."""
    try:
        from flask import current_app
        from twilio.rest import Client

        # Check if Twilio is configured
        account_sid = current_app.config.get('TWILIO_ACCOUNT_SID')
        auth_token = current_app.config.get('TWILIO_AUTH_TOKEN')
        whatsapp_number = current_app.config.get('TWILIO_WHATSAPP_NUMBER')

        if not all([account_sid, auth_token, whatsapp_number]):
            logger.info('Twilio WhatsApp not configured. Skipping WhatsApp notification.')
            return

        if not patient.phone:
            logger.info(f'No phone number for patient {patient.id}. Skipping WhatsApp.')
            return

        cancer_type = (prediction.cancer_type or 'cancer').replace('_', ' ').title()
        confidence = f'{prediction.confidence_score * 100:.1f}%'
        report_url = f'{current_app.config.get("SERVER_NAME", "http://localhost:5000")}/patient/history'

        # WhatsApp message with formatted text
        message = (
            f"⚠️ *HIGH-RISK ALERT*\n\n"
            f"AI detected high-risk {cancer_type} finding with {confidence} confidence.\n\n"
            f"⚕️ *Next Steps:*\n"
            f"1. Log in to view your full AI analysis report\n"
            f"2. Book an appointment with your doctor\n"
            f"3. Download your PDF medical report\n\n"
            f"📋 View Report: {report_url}\n\n"
            f"⚠️ *Disclaimer:* This is an AI-generated alert for informational purposes only. "
            f"Always consult a qualified healthcare professional."
        )

        client = Client(account_sid, auth_token)
        client.messages.create(
            body=message,
            from_=f'whatsapp:{whatsapp_number}',
            to=f'whatsapp:{patient.phone}'
        )

        logger.info(f'WhatsApp alert sent to {patient.phone} for prediction {prediction.id}')

    except ImportError:
        logger.warning('Twilio library not installed. Install with: pip install twilio')
    except Exception as e:
        logger.warning(f'Failed to send WhatsApp alert: {e}')
        raise


def send_in_app_notification(user_id: int, title: str, message: str, notification_type: str = 'alert'):
    """
    Send in-app notification to user.
    
    Args:
        user_id: User ID to send notification to
        title: Notification title
        message: Notification message
        notification_type: Type of notification (alert, info, success, warning)
    """
    try:
        from app.models import Notification
        from app import db

        notification = Notification(
            user_id=user_id,
            title=title,
            message=message,
            notification_type=notification_type,
            is_read=False
        )
        db.session.add(notification)
        db.session.commit()

        logger.info(f'In-app notification sent to user {user_id}: {title}')
        return notification

    except Exception as e:
        logger.warning(f'Failed to send in-app notification: {e}')
        return None


def notify_analysis_completed(prediction, patient):
    """
    Send notifications when analysis is complete:
    - In-app notification for patient
    - Email summary to patient
    - If high-risk: SMS/WhatsApp + alerts to patient and doctors
    """
    try:
        cancer_type = (prediction.cancer_type or 'cancer').replace('_', ' ').title()
        risk = (prediction.risk_level or 'low').upper()
        confidence = f'{prediction.confidence_score * 100:.1f}%'

        # 1. In-app notification to patient
        send_in_app_notification(
            user_id=patient.id,
            title=f'Analysis Completed: {cancer_type}',
            message=f'Your {cancer_type} image analysis is complete. Result: {prediction.prediction_result} ({confidence}, Risk: {risk}).',
            notification_type='alert' if (prediction.risk_level or '').lower() == 'high' else 'info'
        )

        # 2. If high-risk, trigger high-risk alerts (Email, SMS, WhatsApp, and Doctor alerts)
        risk_lower = (prediction.risk_level or '').lower()
        additional = prediction.additional_data or {}
        is_malignant = additional.get('is_malignant', False)

        if risk_lower == 'high' or is_malignant or prediction.confidence_score >= 0.85:
            notify_if_high_risk(prediction, patient)
            _notify_doctors_high_risk(prediction, patient)
        else:
            # Low/Moderate risk: send standard completion email
            _send_completion_email(prediction, patient)

    except Exception as e:
        logger.warning(f'Failed in notify_analysis_completed: {e}')


def _notify_doctors_high_risk(prediction, patient):
    """Send high-risk alert (in-app and SMS) to all active doctors."""
    try:
        from app.models import Doctor
        from flask import current_app
        from twilio.rest import Client

        doctors = Doctor.query.filter_by(is_active=True).all()
        cancer_type = (prediction.cancer_type or 'cancer').replace('_', ' ').title()
        confidence = f'{prediction.confidence_score * 100:.1f}%'

        account_sid = current_app.config.get('TWILIO_ACCOUNT_SID')
        auth_token = current_app.config.get('TWILIO_AUTH_TOKEN')
        from_number = current_app.config.get('TWILIO_PHONE_NUMBER')
        twilio_ready = all([account_sid, auth_token, from_number])

        client = Client(account_sid, auth_token) if twilio_ready else None

        for doc in doctors:
            # In-app alert
            send_in_app_notification(
                user_id=doc.id,
                title=f'🚨 Urgent: High-Risk {cancer_type} Patient Alert',
                message=f'Patient {patient.full_name} has a high-risk {cancer_type} finding ({confidence} confidence). Immediate review advised.',
                notification_type='alert'
            )

            # Doctor SMS alert if Twilio configured and phone available
            if client and doc.phone:
                try:
                    sms_body = (
                        f"🚨 DOCTOR ALERT: Patient {patient.full_name} has a high-risk "
                        f"{cancer_type} finding ({confidence} confidence). Please review dashboard."
                    )
                    client.messages.create(
                        body=sms_body,
                        from_=from_number,
                        to=doc.phone
                    )
                    logger.info(f"Doctor SMS sent to {doc.phone} for patient {patient.id}")
                except Exception as sms_err:
                    logger.warning(f"Could not send SMS to doctor {doc.id}: {sms_err}")

    except Exception as e:
        logger.warning(f'Failed to notify doctors of high-risk case: {e}')


def _send_completion_email(prediction, patient):
    """Send standard analysis completion email to patient."""
    try:
        from flask import current_app
        from flask_mail import Message

        mail = current_app.extensions.get('mail')
        mail_username = current_app.config.get('MAIL_USERNAME')
        if not mail or not mail_username or not patient.email:
            logger.info('Mail not configured or no patient email. Skipping completion email.')
            return

        cancer_type = (prediction.cancer_type or 'cancer').replace('_', ' ').title()
        pred_result = prediction.prediction_result or 'Completed'
        confidence = f'{prediction.confidence_score * 100:.1f}%'
        report_url = f'{current_app.config.get("SERVER_NAME", "http://localhost:5000")}/dashboard/patient/history'

        subject = f'Your AI {cancer_type} Analysis Results Are Ready'
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background: #f8fafc; padding: 20px;">
            <div style="background: white; border-radius: 12px; padding: 25px; max-width: 550px; margin: auto; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                <h2 style="color: #2563eb; margin-top:0;">Analysis Results Available</h2>
                <p>Hello <strong>{patient.full_name}</strong>,</p>
                <p>Your recent {cancer_type} scan analysis has finished processing.</p>
                <div style="background:#f1f5f9; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <p style="margin:5px 0;"><strong>Result:</strong> {pred_result.replace('_', ' ').title()}</p>
                    <p style="margin:5px 0;"><strong>Confidence:</strong> {confidence}</p>
                    <p style="margin:5px 0;"><strong>Risk Level:</strong> {prediction.risk_level.title() if prediction.risk_level else 'Normal'}</p>
                </div>
                <div style="text-align: center; margin: 20px 0;">
                    <a href="{report_url}" style="background: #2563eb; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: bold;">View Details & Report</a>
                </div>
                <p style="font-size: 12px; color: #94a3b8;">Disclaimer: This AI analysis is intended as clinical decision support and does not replace medical consultation.</p>
            </div>
        </body>
        </html>
        """
        msg = Message(
            subject=subject,
            recipients=[patient.email],
            html=html_body,
            sender=current_app.config.get('MAIL_DEFAULT_SENDER', 'noreply@cancerdetection.ai')
        )
        mail.send(msg)
        logger.info(f'Analysis completion email sent to {patient.email}')
    except Exception as e:
        logger.warning(f'Failed to send completion email: {e}')


def notify_report_ready(prediction, patient):
    """Notify patient that their PDF medical report is ready to download."""
    try:
        cancer_type = (prediction.cancer_type or 'cancer').replace('_', ' ').title()
        send_in_app_notification(
            user_id=patient.id,
            title='Medical Report Ready',
            message=f'Your PDF report for {cancer_type} is generated and ready to download.',
            notification_type='success'
        )
    except Exception as e:
        logger.warning(f'Failed to send report ready notification: {e}')

