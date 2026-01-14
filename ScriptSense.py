import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2
import easyocr
import os
from difflib import SequenceMatcher

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

def preprocess_image(image):
    """Apply preprocessing to enhance handwritten text recognition."""
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    
    # Ensure the image is in RGB format
    if len(image_np.shape) == 2:  # If grayscale, convert to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
    # Convert RGB to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    
    return denoised

def pdf_to_text(input_pdf_path):
    """Extract text from a handwritten PDF using OCR."""
    # Open the PDF file
    pdf = fitz.open(input_pdf_path)
    extracted_text = ""
    
    # Extract text from each page
    for page_num in range(pdf.page_count):
        page = pdf[page_num]
        pix = page.get_pixmap()  # Convert page to image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Preprocess the image
        preprocessed_img = preprocess_image(img)
        
        # Use EasyOCR to do OCR on the preprocessed image
        page_text = reader.readtext(preprocessed_img, detail=0)
        
        # Join text lines
        page_text_combined = " ".join(page_text)
        extracted_text += f"Page {page_num + 1}:\n{page_text_combined}\n\n"
    
    pdf.close()
    return extracted_text

def create_text_pdf(text, output_pdf_path):
    """Create a PDF document with the extracted text."""
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)  # Insert text on page
    pdf.save(output_pdf_path)
    pdf.close()

def compare_text_lines(generated_text, reference_text):
    """Compare lines from generated text and reference text."""
    generated_lines = generated_text.split('\n')
    reference_lines = reference_text.split('\n')
    
    evaluation_report = []
    total_score = 0

    for idx, gen_line in enumerate(generated_lines):
        if idx < len(reference_lines):
            ref_line = reference_lines[idx]
            match_ratio = SequenceMatcher(None, gen_line.strip(), ref_line.strip()).ratio()
            
            # Determine score based on matching ratio
            score = max(0, int((match_ratio - 0.5) // 0.12) + 1)
            total_score += score
            evaluation_report.append(f"Question {idx + 1} - Score: {score}\n")
        else:
            evaluation_report.append(f"Question {idx + 1} - No matching reference line.\n")
    
    evaluation_report.append(f"\nTotal Score: {total_score}")
    
    return "\n".join(evaluation_report)

def display_evaluation_pdf(evaluation_text):
    """Display the evaluation report in PDF format."""
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), evaluation_text)  # Insert evaluation text on page
    pdf_bytes = pdf.write()  # Save to bytes for in-memory display
    pdf.close()
    
    return pdf_bytes

# Streamlit Web App
st.title("Handwritten PDF to Text PDF Converter and Evaluator")

# File uploads
handwritten_pdf = st.file_uploader("Upload a Handwritten PDF", type=["pdf"])
reference_pdf = st.file_uploader("Upload a Reference PDF (not handwritten)", type=["pdf"])

if handwritten_pdf and reference_pdf:
    # Convert the uploaded PDFs to a format we can work with
    with open("uploaded_handwritten.pdf", "wb") as f:
        f.write(handwritten_pdf.read())
    
    with open("uploaded_reference.pdf", "wb") as f:
        f.write(reference_pdf.read())

    # OCR conversion of handwritten PDF
    st.write("Converting the handwritten PDF to text...")
    generated_text = pdf_to_text("uploaded_handwritten.pdf")
    
    # Extract text from reference PDF
    st.write("Extracting text from the reference PDF...")
    reference_text = pdf_to_text("uploaded_reference.pdf")
    
    # Generate a new PDF with the extracted text
    st.write("Generating the converted PDF...")
    create_text_pdf(generated_text, "generated_text.pdf")
    
    # Evaluation
    st.write("Evaluating the generated text against the reference...")
    evaluation_report = compare_text_lines(generated_text, reference_text)
    
    # Display the evaluation result
    pdf_bytes = display_evaluation_pdf(evaluation_report)
    st.download_button(label="Download Evaluation Report PDF", data=pdf_bytes, file_name="evaluation_report.pdf")

    # Show evaluation report in the web app
    st.write("### Evaluation Report")
    st.text_area("Evaluation Results", evaluation_report, height=400)
