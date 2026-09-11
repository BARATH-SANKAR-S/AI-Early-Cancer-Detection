"""
Medical Report PDF Generator
Generates professional downloadable medical reports from AI prediction results.
Uses ReportLab for PDF generation with embedded Grad-CAM heatmaps.
"""

import io
import os
import base64
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics import renderPDF

import logging
logger = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
PRIMARY   = colors.HexColor('#1a56db')   # Deep blue
SECONDARY = colors.HexColor('#1e40af')
DANGER    = colors.HexColor('#dc2626')
WARNING   = colors.HexColor('#d97706')
SUCCESS   = colors.HexColor('#16a34a')
LIGHT_BG  = colors.HexColor('#f8fafc')
BORDER    = colors.HexColor('#e2e8f0')
TEXT_DARK = colors.HexColor('#1e293b')
TEXT_MID  = colors.HexColor('#475569')
TEXT_LITE = colors.HexColor('#94a3b8')
WHITE     = colors.white

CANCER_EMOJI = {
    'brain_tumor': '🧠',
    'lung_cancer':  '🫁',
    'skin_cancer':  '🔬',
}

RISK_COLORS = {
    'high':   DANGER,
    'medium': WARNING,
    'low':    SUCCESS,
}


# ── Style helpers ─────────────────────────────────────────────────────────────

def _make_styles():
    base = getSampleStyleSheet()
    s = {}
    s['hospital'] = ParagraphStyle('hospital',
        fontName='Helvetica-Bold', fontSize=18,
        textColor=WHITE, alignment=TA_CENTER, spaceAfter=2)
    s['hospital_sub'] = ParagraphStyle('hospital_sub',
        fontName='Helvetica', fontSize=9,
        textColor=colors.HexColor('#bfdbfe'), alignment=TA_CENTER)
    s['report_title'] = ParagraphStyle('report_title',
        fontName='Helvetica-Bold', fontSize=14,
        textColor=PRIMARY, alignment=TA_CENTER, spaceBefore=12, spaceAfter=4)
    s['section_header'] = ParagraphStyle('section_header',
        fontName='Helvetica-Bold', fontSize=11,
        textColor=WHITE, spaceBefore=10, spaceAfter=4, leftIndent=6)
    s['body'] = ParagraphStyle('body',
        fontName='Helvetica', fontSize=9,
        textColor=TEXT_DARK, spaceAfter=3, leading=13)
    s['small'] = ParagraphStyle('small',
        fontName='Helvetica', fontSize=8,
        textColor=TEXT_MID, spaceAfter=2, leading=11)
    s['disclaimer'] = ParagraphStyle('disclaimer',
        fontName='Helvetica-Oblique', fontSize=7.5,
        textColor=TEXT_MID, alignment=TA_CENTER, leading=11)
    s['centered'] = ParagraphStyle('centered',
        fontName='Helvetica-Bold', fontSize=12,
        textColor=TEXT_DARK, alignment=TA_CENTER)
    return s


def _section_header(title, styles):
    data = [[Paragraph(f'<b>{title}</b>', styles['section_header'])]]
    t = Table(data, colWidths=[17 * cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PRIMARY),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [PRIMARY]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ]))
    return t


def _info_table(rows, styles):
    data = []
    for label, value in rows:
        data.append([
            Paragraph(f'<b>{label}</b>', styles['small']),
            Paragraph(str(value), styles['body']),
        ])
    t = Table(data, colWidths=[5 * cm, 12 * cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.3, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return t


# ── Heatmap helper ────────────────────────────────────────────────────────────

def _decode_heatmap_to_temp(heatmap_b64):
    """
    Decode a base64 data-URI heatmap string and save as a temp PNG file.
    Returns temp file path or None on failure.
    """
    if not heatmap_b64:
        return None
    try:
        if ',' in heatmap_b64:
            heatmap_b64 = heatmap_b64.split(',', 1)[1]
        img_data = base64.b64decode(heatmap_b64)
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.write(img_data)
        tmp.close()
        return tmp.name
    except Exception as e:
        logger.warning(f'Could not decode heatmap: {e}')
        return None


# ── Main generator ────────────────────────────────────────────────────────────

def generate_pdf_report(prediction, patient, output_path=None):
    """
    Generate a professional PDF medical report.

    Args:
        prediction  (PredictionHistory): DB record with AI results.
        patient     (Patient): Patient DB record.
        output_path (str | None): If None, returns a BytesIO buffer.

    Returns:
        BytesIO buffer containing the PDF, or writes to output_path.
    """
    styles = _make_styles()
    buf = io.BytesIO() if output_path is None else output_path

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title='AI Cancer Detection Medical Report',
        author='AI Early Cancer Detection System',
    )

    elements = []
    add = elements.append
    extra_files = []   # temp files to clean up

    # ── Header banner ────────────────────────────────────────────────────────
    cancer_type = prediction.cancer_type or 'cancer'
    emoji = CANCER_EMOJI.get(cancer_type, '🏥')
    cancer_label = cancer_type.replace('_', ' ').title()

    header_data = [[
        Paragraph(f'AI Early Cancer Detection System', styles['hospital']),
        Paragraph('Powered by Deep Learning • EfficientNet / ResNet / ViT', styles['hospital_sub']),
    ]]
    # Flatten to single column
    header_data = [
        [Paragraph('AI Early Cancer Detection System', styles['hospital'])],
        [Paragraph('Powered by Deep Learning  •  EfficientNet / ResNet / ViT', styles['hospital_sub'])],
    ]
    h = Table([[Paragraph('AI Early Cancer Detection System', styles['hospital'])],
               [Paragraph('Powered by Deep Learning  •  EfficientNet / ResNet / ViT', styles['hospital_sub'])]],
              colWidths=[17 * cm])
    h.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PRIMARY),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('ROUNDEDCORNERS', [6, 6, 6, 6]),
    ]))
    add(h)
    add(Spacer(1, 8))

    # Report meta
    add(Paragraph(f'{emoji} {cancer_label} — AI Diagnostic Report', styles['report_title']))
    add(Paragraph(
        f'Report ID: RPT-{prediction.id:05d}  •  '
        f'Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}',
        ParagraphStyle('meta', fontName='Helvetica', fontSize=8,
                       textColor=TEXT_MID, alignment=TA_CENTER, spaceAfter=6)
    ))
    add(HRFlowable(width='100%', thickness=0.5, color=BORDER, spaceAfter=8))

    # ── Patient Information ───────────────────────────────────────────────────
    add(_section_header('👤  Patient Information', styles))
    add(Spacer(1, 4))
    add(_info_table([
        ('Full Name',   patient.full_name),
        ('Username',    patient.username),
        ('Email',       patient.email),
        ('Age',         patient.age or '—'),
        ('Gender',      patient.gender or '—'),
        ('Blood Group', patient.blood_group or '—'),
        ('Phone',       patient.phone or '—'),
        ('District',    patient.district or '—'),
    ], styles))
    add(Spacer(1, 10))

    # ── AI Analysis Results ───────────────────────────────────────────────────
    add(_section_header('🤖  AI Analysis Results', styles))
    add(Spacer(1, 4))

    additional = prediction.additional_data or {}
    label = additional.get('label', prediction.prediction_result) or prediction.prediction_result
    resolved_risk = prediction.risk_level or additional.get('risk_level') or getattr(prediction, 'effective_risk_level', None) or '—'
    risk_level = resolved_risk
    image_type_str = (additional.get('image_type') or '—').replace('_', ' ').upper()
    confidence_pct = f'{prediction.confidence_score * 100:.2f}%'

    risk_color = RISK_COLORS.get(str(risk_level).lower(), TEXT_MID)

    add(_info_table([
        ('Cancer Type',     cancer_label),
        ('Image Modality',  image_type_str),
        ('AI Prediction',   label.replace('_', ' ').title() if label else '—'),
        ('Confidence Score', confidence_pct),
        ('Risk Level',      risk_level.capitalize() if risk_level and risk_level != '—' else '—'),
        ('Model Used',      prediction.model_version or '—'),
        ('Analysis Date',   prediction.prediction_date.strftime('%Y-%m-%d %H:%M UTC') if prediction.prediction_date else '—'),
    ], styles))

    # Risk visual indicator
    add(Spacer(1, 8))
    risk_str = str(risk_level).lower()
    risk_display = risk_level.capitalize() if risk_level and risk_level != '—' else 'Unknown'
    risk_bg = RISK_COLORS.get(risk_str, TEXT_MID)
    risk_cell = Table([[Paragraph(f'<b>Risk Assessment: {risk_display}</b>',
                                   ParagraphStyle('risk', fontName='Helvetica-Bold',
                                                  fontSize=12, textColor=WHITE, alignment=TA_CENTER))]],
                      colWidths=[17 * cm])
    risk_cell.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), risk_bg),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ]))
    add(risk_cell)
    add(Spacer(1, 10))

    # ── Class Probabilities ───────────────────────────────────────────────────
    class_probs = additional.get('class_probabilities', {})
    if class_probs:
        add(_section_header('📊  Class Probabilities', styles))
        add(Spacer(1, 4))
        prob_rows = [['Class', 'Probability', 'Confidence Bar']]
        for cls_name, prob in sorted(class_probs.items(), key=lambda x: -x[1]):
            pct = prob * 100
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            prob_rows.append([
                Paragraph(cls_name.replace('_', ' ').capitalize(), styles['body']),
                Paragraph(f'<b>{pct:.1f}%</b>', styles['body']),
                Paragraph(f'<font color="#3b82f6">{bar[:int(pct/5)]}</font>'
                          f'<font color="#e2e8f0">{bar[int(pct/5):]}</font>', styles['small']),
            ])
        t = Table(prob_rows, colWidths=[6 * cm, 3 * cm, 8 * cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('GRID', (0, 0), (-1, -1), 0.3, BORDER),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        add(t)
        add(Spacer(1, 10))

    # ── Grad-CAM Heatmap ─────────────────────────────────────────────────────
    heatmap_b64 = additional.get('heatmap_b64')
    if heatmap_b64:
        tmp_path = _decode_heatmap_to_temp(heatmap_b64)
        if tmp_path:
            extra_files.append(tmp_path)
            add(_section_header('🔥  Grad-CAM Explainability Heatmap', styles))
            add(Spacer(1, 4))
            add(Paragraph(
                'The heatmap below shows which regions of the image the AI model focused on '
                'when making its prediction. Red/warm areas = highest attention.',
                styles['small']
            ))
            add(Spacer(1, 4))
            try:
                img_rl = RLImage(tmp_path, width=8 * cm, height=8 * cm)
                img_table = Table([[img_rl]], colWidths=[17 * cm])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                add(img_table)
            except Exception as e:
                logger.warning(f'Could not embed heatmap image: {e}')
            add(Spacer(1, 10))

    # ── Clinical Notes Section ───────────────────────────────────────────────
    add(_section_header("📝  Doctor's Clinical Notes & Review", styles))
    add(Spacer(1, 4))

    doctor_notes = getattr(prediction, 'doctor_notes', None)
    if doctor_notes:
        for dn in doctor_notes:
            doc_name = dn.doctor.full_name if (dn.doctor and dn.doctor.full_name) else 'Attending Physician'
            doc_spec = dn.doctor.specialization if (dn.doctor and dn.doctor.specialization) else 'Clinical Specialist'
            date_str = dn.created_at.strftime('%Y-%m-%d %H:%M') if dn.created_at else 'N/A'
            status_text = (dn.status or 'reviewed').upper()

            note_content = [
                [
                    Paragraph(f'<b>Reviewing Doctor:</b> Dr. {doc_name} ({doc_spec})', styles['small']),
                    Paragraph(f'<b>Status:</b> <b>[{status_text}]</b> &nbsp;&bull;&nbsp; {date_str}', styles['small']),
                ],
                [
                    Paragraph(f'<b>Clinical Assessment:</b><br/>{dn.note_text}', styles['body']),
                    '',
                ]
            ]
            nt = Table(note_content, colWidths=[11 * cm, 6 * cm])
            nt.setStyle(TableStyle([
                ('SPAN', (0, 1), (1, 1)),
                ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
                ('BOX', (0, 0), (-1, -1), 0.5, PRIMARY),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            add(nt)
            add(Spacer(1, 6))
    else:
        add(Paragraph('Clinical Findings:', styles['small']))
        blank_line_table = Table([[''] * 1] * 3, colWidths=[17 * cm], rowHeights=[14] * 3)
        blank_line_table.setStyle(TableStyle([
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]))
        add(blank_line_table)
        add(Spacer(1, 4))
        add(Paragraph('Recommendations:', styles['small']))
        add(blank_line_table)
        add(Spacer(1, 4))
        add(Paragraph('Doctor Signature & Stamp: _________________________  Date: __________', styles['small']))
    add(Spacer(1, 10))

    # ── Disclaimer ───────────────────────────────────────────────────────────
    add(HRFlowable(width='100%', thickness=0.5, color=BORDER, spaceAfter=6))
    add(Paragraph(
        '⚠️  IMPORTANT DISCLAIMER: This report is generated by an Artificial Intelligence system '
        'for research and educational purposes only. It does NOT constitute a medical diagnosis. '
        'Always consult a qualified healthcare professional for proper medical evaluation and treatment. '
        'The AI confidence scores are statistical probabilities and should not be interpreted as definitive diagnoses.',
        styles['disclaimer']
    ))
    add(Spacer(1, 4))
    add(Paragraph(
        f'Generated by AI Early Cancer Detection System  |  '
        f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC  |  Report ID: RPT-{prediction.id:05d}',
        styles['disclaimer']
    ))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc.build(elements)

    # Clean up temp heatmap files
    for f in extra_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    if output_path is None:
        buf.seek(0)
        return buf

    return None
