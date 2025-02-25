import streamlit as st
from modules.text_extractor import render_text_tab
from modules.image_handler import render_image_tab
from modules.qa_analyzer import render_qa_tab
import os

# Fix for torch.classes error
os.environ["STREAMLIT_WATCH_MODULES"] = "false"

def main():
    st.title("PDF Analysis with OCR and Q&A")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        tab1, tab2, tab3 = st.tabs(["Extracted Text", "Images", "Q&A"])
        
        with tab1:
            extracted_text = render_text_tab(uploaded_file)
        
        with tab2:
            render_image_tab(uploaded_file)
        
        with tab3:
            render_qa_tab(extracted_text)

if __name__ == "__main__":
    main() 