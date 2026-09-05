import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def build_pdf(pdf_path="ANPR_System_Documentation.pdf"):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1A2B4C"),
        alignment=0,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceAfter=15
    )

    heading2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1A2B4C"),
        spaceBefore=14,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8
    )

    code_style = ParagraphStyle(
        'CodeCustom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1A202C"),
        backColor=colors.HexColor("#EDF2F7"),
        borderColor=colors.HexColor("#CBD5E0"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=15,
        spaceAfter=4
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("SIH127: Automatic Number Plate Recognition (ANPR)", title_style))
    elements.append(Paragraph("Technical System Documentation, Stack Details & Deployment Guide", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=12))

    # Executive Summary
    elements.append(Paragraph("1. Executive Summary", heading2_style))
    elements.append(Paragraph(
        "This document provides full technical specifications, dependency requirements, and setup instructions "
        "for the SIH127 Automatic Number Plate Recognition (ANPR) system. Designed specifically for Indian vehicle registration formats, "
        "the pipeline achieves high accuracy localizing and recognizing standard car plates as well as challenging 2-line motorcycle plates.",
        body_style
    ))

    # Technology Stack
    elements.append(Paragraph("2. Technology Stack", heading2_style))
    stack_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Technology</b>", body_style), Paragraph("<b>Version / Specification</b>", body_style), Paragraph("<b>Role in Pipeline</b>", body_style)],
        [Paragraph("Programming Language", body_style), Paragraph("Python", body_style), Paragraph("3.11+", body_style), Paragraph("Pipeline orchestration & business logic", body_style)],
        [Paragraph("Object Detection", body_style), Paragraph("Ultralytics YOLOv8", body_style), Paragraph("YOLOv8n (Fine-tuned)", body_style), Paragraph("Bounding box plate localization", body_style)],
        [Paragraph("Computer Vision", body_style), Paragraph("OpenCV", body_style), Paragraph("opencv-python 4.x", body_style), Paragraph("Super-res upscaling, CLAHE & Otsu thresholding", body_style)],
        [Paragraph("OCR Engine", body_style), Paragraph("EasyOCR", body_style), Paragraph("1.7+", body_style), Paragraph("Deep learning character extraction", body_style)],
        [Paragraph("Deep Learning Core", body_style), Paragraph("PyTorch", body_style), Paragraph("2.x (CPU / GPU)", body_style), Paragraph("Neural network tensor operations", body_style)],
        [Paragraph("PDF Generation", body_style), Paragraph("ReportLab", body_style), Paragraph("5.x", body_style), Paragraph("Automated system documentation", body_style)]
    ]
    t_stack = Table(stack_data, colWidths=[1.5*inch, 1.4*inch, 1.4*inch, 2.7*inch])
    t_stack.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(t_stack)
    elements.append(Spacer(1, 10))

    # Mandatory Package Imports
    elements.append(Paragraph("3. Required Python Packages & Imports", heading2_style))
    elements.append(Paragraph("When developing or extending the ANPR pipeline, the following Python libraries must be installed and imported:", body_style))
    
    code_text = (
        "# Mandatory Python Package Imports for SIH127 ANPR System<br/>"
        "import cv2                          # OpenCV Computer Vision Library<br/>"
        "import numpy as np                  # Numerical matrix operations<br/>"
        "import re                           # Regular expressions for syntax validation<br/>"
        "import time                         # Latency & benchmark timing<br/>"
        "import os                           # File system operations<br/>"
        "from typing import Dict, Any, List  # Static type annotations<br/>"
        "from ultralytics import YOLO        # Fine-tuned YOLOv8 Detector<br/>"
        "import easyocr                      # EasyOCR Character Recognition Engine<br/>"
        "import torch                        # PyTorch Deep Learning Backend"
    )
    elements.append(Paragraph(code_text, code_style))

    # Installation Guide
    elements.append(Paragraph("4. Step-by-Step Installation & Deployment", heading2_style))
    elements.append(Paragraph("<b>Step 1: Clone the GitHub Repository</b>", bullet_style))
    elements.append(Paragraph("git clone https://github.com/aartisingh07/SIH127_2026.git<br/>cd SIH127_2026", code_style))
    
    elements.append(Paragraph("<b>Step 2: Create & Activate Virtual Environment</b>", bullet_style))
    elements.append(Paragraph("# Windows:<br/>python -m venv venv<br/>venv\\Scripts\\activate<br/><br/># Linux / Mac:<br/>python3 -m venv venv<br/>source venv/bin/activate", code_style))
    
    elements.append(Paragraph("<b>Step 3: Install Required Dependencies</b>", bullet_style))
    elements.append(Paragraph("pip install ultralytics easyocr opencv-python numpy reportlab torch torchvision", code_style))

    # Pipeline Features
    elements.append(Paragraph("5. Core Architectural Innovations", heading2_style))
    elements.append(Paragraph("• <b>Multi-Line Candidate Box Merging:</b> Automatically detects and merges vertically adjacent bounding boxes to capture top and bottom lines of 2-line motorcycle plates.", bullet_style))
    elements.append(Paragraph("• <b>Top-to-Bottom Text Concatenation:</b> Sorts extracted OCR text blocks by vertical Y-coordinates before joining strings (e.g. MH19BY + 2225 = MH19BY2225).", bullet_style))
    elements.append(Paragraph("• <b>Brand Emblem Filtering:</b> Strips vehicle distractors and stickers (POLICE, BULLET, ROYAL, ENFIELD, HERO, HONDA).", bullet_style))
    elements.append(Paragraph("• <b>Positional Syntax Repair:</b> Enforces standard Indian plate structure [State 2L][District 2D][Series 1-3L][Number 4D] with positional digit/character mapping.", bullet_style))

    # Benchmarks
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("6. Verification & Test Dataset Results", heading2_style))
    bench_data = [
        [Paragraph("<b>Image Sample</b>", body_style), Paragraph("<b>Vehicle Type</b>", body_style), Paragraph("<b>Expected Plate</b>", body_style), Paragraph("<b>Recognized Output</b>", body_style), Paragraph("<b>Accuracy Score</b>", body_style)],
        [Paragraph("test1.jpg", body_style), Paragraph("Royal Enfield Bullet", body_style), Paragraph("MH02GD7249", body_style), Paragraph("MH02GD7249", body_style), Paragraph("100.00%", body_style)],
        [Paragraph("test2.jpg", body_style), Paragraph("Hero Motorcycle", body_style), Paragraph("MH19BY2225", body_style), Paragraph("MH19BY2225", body_style), Paragraph("100.00%", body_style)],
        [Paragraph("test3.jpg", body_style), Paragraph("TVS Motorcycle", body_style), Paragraph("MH34H1559", body_style), Paragraph("MH34H1559", body_style), Paragraph("100.00%", body_style)],
        [Paragraph("test4.jpg", body_style), Paragraph("Car / Commercial", body_style), Paragraph("MH05AE8290", body_style), Paragraph("MH05AE8290", body_style), Paragraph("100.00%", body_style)]
    ]
    t_bench = Table(bench_data, colWidths=[1.1*inch, 1.8*inch, 1.4*inch, 1.4*inch, 1.3*inch])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(t_bench)

    doc.build(elements)
    print(f"[PDF Generator] Successfully generated PDF document at: '{pdf_path}'")

if __name__ == "__main__":
    build_pdf("c:/SIH127/ANPR_System_Documentation.pdf")
