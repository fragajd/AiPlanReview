# OCR Fraga - PDF Analysis Tool

A powerful PDF analysis tool that combines OCR (Optical Character Recognition), text extraction, and question-answering capabilities. Built with Streamlit, this application provides an intuitive interface for analyzing PDF documents.

## Features

- **Text Extraction**: Extract text from PDF documents using advanced OCR technology
- **Image Analysis**: View and analyze images contained within PDF files
- **Q&A System**: Ask questions about the extracted text and get intelligent answers
- **User-Friendly Interface**: Clean and intuitive web interface built with Streamlit
- **Multi-Tab Interface**: Organized view with separate tabs for text, images, and Q&A

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine installed on your system
  - Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ocrfraga.git
   cd ocrfraga
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv env
   .\env\Scripts\activate

   # Linux/macOS
   python -m venv env
   source env/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload a PDF file using the file uploader

4. Navigate through the tabs to:
   - View extracted text
   - Analyze images from the PDF
   - Ask questions about the content

## Project Structure

```
ocrfraga/
├── app.py              # Main application file
├── modules/            # Application modules
│   ├── text_extractor.py  # Text extraction functionality
│   ├── image_handler.py   # Image processing functionality
│   └── qa_analyzer.py     # Q&A system functionality
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Dependencies

- streamlit - Web application framework
- pytesseract - OCR engine Python wrapper
- opencv-python - Image processing library
- Pillow - Python Imaging Library
- PyMuPDF - PDF processing library
- transformers - Hugging Face Transformers for Q&A
- torch - PyTorch for deep learning
- spacy - Natural Language Processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers. 