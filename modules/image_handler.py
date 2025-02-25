import fitz
from PIL import Image
import streamlit as st
import io

def extract_images_from_pdf(pdf_file):
    try:
        pdf_file.seek(0)
        pdf_content = pdf_file.read()
        
        images = []
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)
        
        pdf_document.close()
        return images
    except Exception as e:
        st.error(f"Error extracting images: {str(e)}")
        return []

def render_image_tab(uploaded_file):
    st.subheader("Extracted Images")
    images = extract_images_from_pdf(uploaded_file)
    for idx, img in enumerate(images):
        st.image(img, caption=f"Image {idx + 1}") 