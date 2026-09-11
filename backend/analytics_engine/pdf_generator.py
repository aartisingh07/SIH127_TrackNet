import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_trajectory_pdf(trajectory_data, output_filepath):
    """Generates an enterprise PDF report for vehicle trajectory & camera detections using ReportLab."""
    doc = SimpleDocTemplate(
        output_filepath,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#0f172a") # Dark Slate
    secondary_color = colors.HexColor("#0284c7") # Sky Blue
    accent_color = colors.HexColor("#ef4444") # Crimson Alert
    light_bg = colors.HexColor("#f8fafc")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=primary_color,
        fontName='Helvetica-Bold',
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=secondary_color,
        fontName='Helvetica-Bold',
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1e293b")
    )
    
    story = []
    
    # Header Banner
    story.append(Paragraph("CITY-WIDE ANPR TRAJECTORY & TRAFFIC ANALYTICS REPORT", title_style))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Target Plate: <b>{trajectory_data.get('target_plate', 'N/A')}</b>", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=secondary_color, spaceAfter=15))
    
    # Summary Box
    summary_data = [
        [
            Paragraph("<b>Target License Plate:</b>", body_style), Paragraph(str(trajectory_data.get('target_plate')), body_style),
            Paragraph("<b>Total Detections:</b>", body_style), Paragraph(str(trajectory_data.get('total_detections')), body_style)
        ],
        [
            Paragraph("<b>Camera Sequence:</b>", body_style), Paragraph(str(trajectory_data.get('camera_sequence')), body_style),
            Paragraph("<b>Total Distance:</b>", body_style), Paragraph(f"{trajectory_data.get('total_distance_km')} km", body_style)
        ],
        [
            Paragraph("<b>Max Speed Observed:</b>", body_style), Paragraph(f"{trajectory_data.get('max_speed_kmh')} km/h", body_style),
            Paragraph("<b>Status:</b>", body_style), Paragraph("<font color='#10b981'><b>VERIFIED</b></font>", body_style)
        ]
    ]
    
    summary_table = Table(summary_data, colWidths=[120, 150, 120, 150])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # Trajectory Nodes Detail Table
    story.append(Paragraph("Spatial-Temporal Camera Trajectory Logs", heading_style))
    
    table_headers = ["Step", "Camera ID", "Location Name", "Timestamp", "Confidence", "Dist (km)", "Speed (km/h)"]
    table_rows = [[Paragraph(f"<b>{h}</b>", ParagraphStyle('TH', parent=body_style, textColor=colors.white, fontName='Helvetica-Bold')) for h in table_headers]]
    
    for node in trajectory_data.get('trajectory_nodes', []):
        row = [
            Paragraph(str(node['step']), body_style),
            Paragraph(str(node['camera_id']), body_style),
            Paragraph(str(node['location_name']), body_style),
            Paragraph(str(node['timestamp']), body_style),
            Paragraph(f"{int(node['confidence']*100)}%", body_style),
            Paragraph(str(node['segment_distance_km']), body_style),
            Paragraph(str(node['calculated_speed_kmh']), body_style),
        ]
        table_rows.append(row)
        
    logs_table = Table(table_rows, colWidths=[35, 65, 175, 110, 50, 50, 55])
    logs_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg])
    ]))
    story.append(logs_table)
    story.append(Spacer(1, 15))
    
    # Anomalies Section
    anomalies = trajectory_data.get('anomalies', [])
    if anomalies:
        story.append(Paragraph("Security & Route Anomalies Flagged", heading_style))
        for a in anomalies:
            story.append(Paragraph(f"• <font color='#ef4444'><b>ALERT:</b></font> {a}", body_style))
        story.append(Spacer(1, 10))

    # Footer Notice
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceBefore=15, spaceAfter=10))
    story.append(Paragraph("Confidential - Automated City ANPR Spatial-Temporal Tracking Report | Enterprise License", subtitle_style))
    
    doc.build(story)
    return output_filepath

if __name__ == "__main__":
    test_data = {
        'target_plate': 'MH12AB1234',
        'total_detections': 4,
        'camera_sequence': 'Camera 1 → Camera 3 → Camera 5 → Camera 6',
        'total_distance_km': 28.5,
        'max_speed_kmh': 72.4,
        'anomalies': ['Route consistency within normal limits'],
        'trajectory_nodes': [
            {'step': 1, 'camera_id': 'CAM-01', 'location_name': 'Connaught Place North Node', 'timestamp': '2026-09-06 20:15:00', 'confidence': 0.96, 'segment_distance_km': 0.0, 'calculated_speed_kmh': 0.0},
            {'step': 2, 'camera_id': 'CAM-03', 'location_name': 'AIIMS Flyover Intersection', 'timestamp': '2026-09-06 20:27:00', 'confidence': 0.94, 'segment_distance_km': 7.4, 'calculated_speed_kmh': 37.0},
            {'step': 3, 'camera_id': 'CAM-05', 'location_name': 'IGI Airport Terminal 3 Toll', 'timestamp': '2026-09-06 20:43:00', 'confidence': 0.95, 'segment_distance_km': 12.1, 'calculated_speed_kmh': 45.4},
            {'step': 4, 'camera_id': 'CAM-06', 'location_name': 'Cyber Hub Gurgaon Highway', 'timestamp': '2026-09-06 20:57:00', 'confidence': 0.93, 'segment_distance_km': 9.0, 'calculated_speed_kmh': 38.6},
        ]
    }
    os.makedirs(r"c:\SIH127-I\reports", exist_ok=True)
    pdf_p = generate_trajectory_pdf(test_data, r"c:\SIH127-I\reports\sample_report.pdf")
    print("Generated PDF at:", pdf_p)
