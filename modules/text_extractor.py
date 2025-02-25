import pytesseract
import cv2
import numpy as np
from PIL import Image
import fitz
import streamlit as st
import io
import os
import re

# Set Tesseract path for Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image, enhance_resolution=True):
    """
    Preprocess image to improve OCR results
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove noise
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    # Increase resolution if requested
    if enhance_resolution:
        return cv2.resize(denoised, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    return denoised

def clean_ocr_text(text):
    """Clean up common OCR errors and formatting issues"""
    if not text:
        return ""
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove isolated single characters (likely OCR errors)
    text = re.sub(r'(?<!\w)([a-zA-Z])(?!\w)', ' ', text)
    
    # Fix common OCR errors
    text = text.replace('|', 'I').replace('0', 'O')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix broken sentences (period followed by lowercase)
    text = re.sub(r'(\.)([a-z])', r'\1 \2', text)
    
    return text.strip()

def extract_text_from_pdf(pdf_file, use_ocr=True, enhance_resolution=True):
    """
    Extract text from a PDF using a hybrid approach:
    1. Try to extract embedded text with PyMuPDF
    2. Fall back to OCR for pages with little or no text
    """
    try:
        pdf_file.seek(0)
        pdf_content = pdf_file.read()
        
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        extracted_text = []
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # First, try to extract embedded text
            embedded_text = page.get_text()
            
            # If there's enough embedded text, use it
            if len(embedded_text.strip()) > 50 and not use_ocr:
                extracted_text.append(embedded_text)
                continue
            
            # Otherwise, fall back to OCR
            pix = page.get_pixmap(alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Preprocess the image
            processed_img = preprocess_image(opencv_img, enhance_resolution)
            
            # Run OCR with custom configuration
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            
            # Clean the OCR text
            text = clean_ocr_text(text)
            
            # If OCR text is too short, try the embedded text
            if len(text.strip()) < 20 and embedded_text.strip():
                extracted_text.append(embedded_text)
            else:
                extracted_text.append(text)
        
        pdf_document.close()
        return "\n\n".join(extracted_text)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return ""

def render_text_tab(uploaded_file):
    st.subheader("Extracted Text")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.write("OCR Settings")
        force_ocr = st.checkbox("Force OCR", value=False, 
                               help="Use OCR even if embedded text is found")
        enhance_res = st.checkbox("Enhance Resolution", value=True,
                                help="Increase image resolution for better OCR")
    
    with col1:
        with st.spinner("Extracting text..."):
            extracted_text = extract_text_from_pdf(
                uploaded_file, 
                use_ocr=force_ocr,
                enhance_resolution=enhance_res
            )
        
        st.text_area("OCR Result", extracted_text, height=300)
    
    return extracted_text 