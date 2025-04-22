# OCR Fraga - AI Architectural & Engineering Plan Reviewer

A powerful AI-driven tool for analyzing Architectural and Engineering PDF documents. It combines advanced OCR, text extraction, image analysis, and a sophisticated question-answering system built on Retrieval-Augmented Generation (RAG). Developed with Streamlit, this application offers an intuitive web interface for comprehensive plan review.

## Features

- **Project Management**: Upload new PDF plan sets or load previously processed projects from a persistent database (Supabase).
- **Advanced Text Extraction**:
    - Extracts text from PDFs using Optical Character Recognition (OCR via Tesseract).
    - Performs specialized pre-processing and cleaning tailored for technical A&E documents (standardizing measurements, code references, terminology).
    - Generates semantic embeddings for extracted text, enabling intelligent search and analysis.
- **Image Handling**:
    - Extracts images embedded within PDF documents.
    - Stores images and associated metadata (source file, page number, position) in cloud storage (Supabase Storage).
    - Provides an interface to view extracted images.
    - **(Experimental/Requires Setup)** Offers AI-powered analysis of image content using multimodal models (e.g., local Ollama LLaVA or potentially Gemini).
- **Retrieval-Augmented Generation (RAG) Q&A System**:
    - Ask natural language questions about the content of the uploaded plans.
    - Performs semantic search across the extracted text embeddings to find relevant context.
    - **Includes Florida Building Code Integration**: Optionally searches an embedded Florida Building Code database to find relevant code sections alongside document context.
    - Generates context-aware answers using Large Language Models (LLMs).
    - Supports compliance-focused queries.
- **Flexible AI Model Selection**: Choose between using:
    - Local models via Ollama (e.g., Gemma for text generation, potentially LLaVA for image analysis). Requires Ollama installation and model setup.
    - Google Gemini API (Requires a Google API key).
- **Intuitive Multi-Tab Interface**: Built with Streamlit, providing separate tabs for:
    - **Text/Project Info**: View extracted text, processing status, and project details.
    - **Images**: View and analyze extracted images.
    - **Q&A**: Interact with the AI assistant.
- **Database Persistence**: Utilizes Supabase (PostgreSQL database and Storage) to save project data, extracted text, embeddings, image metadata, and analysis results.

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine installed on your system
  - Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki). Ensure `tesseract.exe` is in your system's PATH or update the path in `modules/text_extractor.py`.
  - Linux: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`
- **(Optional but Recommended)** Ollama installed and running for local AI model support. Download from [Ollama.ai](https://ollama.ai/). Pull desired models (e.g., `ollama pull gemma`, `ollama pull llava`).
- **(Optional)** Supabase account and project setup for data persistence. Create a project at [Supabase.io](https://supabase.io/). You will need API URL and Key. Configure database schema (see project setup details - *to be added*).
- **(Optional)** Google API Key for using the Gemini model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ocrfraga.git # Replace with actual repo URL
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

4. **Environment Configuration**:
   - Create a `.env` file in the project root directory.
   - Add the following variables, replacing placeholders with your actual credentials if using Supabase/Gemini:
     ```dotenv
     SUPABASE_URL="YOUR_SUPABASE_URL"
     SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
     # Optional: Google API Key if using Gemini
     # GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
     # Optional: Specify local model if not using default
     # OLLAMA_MODEL_NAME="gemma:latest"
     # TESSERACT_PATH="C:\Program Files\Tesseract-OCR\tesseract.exe" # Example for Windows if not in PATH
     ```
   - *Note: The application attempts to function without Supabase/Gemini for basic local processing, but persistence and API models require configuration.*

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`).

3. **Select Mode**:
   - **Upload New Files**: Use the file uploader in the main panel.
   - **Load Existing Project**: Select a project from the dropdown in the sidebar (requires Supabase setup).

4. **Configure AI Settings (Sidebar)**:
   - Choose between 'Local (Ollama/Gemma)' or 'API (Google Gemini)'.
   - If using API, enter your Google API Key.

5. **Analyze**:
   - Navigate through the "Text/Project Info", "Images", and "Q&A" tabs.
   - Processing (text extraction, embedding, image extraction) happens automatically upon file upload or project load. Progress indicators will be shown.
   - Use the Q&A tab to ask questions about the loaded project content.

## Project Structure

```
ocrfraga/
├── app.py                     # Main Streamlit application file
├── modules/                   # Core application logic
│   ├── __init__.py
│   ├── text_extractor.py      # PDF text extraction, OCR, cleaning, embedding
│   ├── image_handler.py       # PDF image extraction, storage, analysis
│   ├── qa_analyzer.py         # RAG Q&A system, context retrieval, AI interaction
│   └── shared_resources.py    # Shared utilities (DB connection, model loading)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .env.example               # Example environment file structure
└── (Optional) database_schema.sql # Example SQL for Supabase setup
```

## Key Dependencies

- `streamlit`: Web application framework
- `pytesseract` & `opencv-python-headless` & `Pillow`: OCR and image processing
- `PyMuPDF` (fitz): PDF parsing and manipulation
- `transformers`: Foundational library (though specific pipelines might be replaced by direct Ollama/Gemini calls)
- `sentence-transformers`: Generating text embeddings for semantic search
- `torch` & `torchvision` & `torchaudio`: Required by sentence-transformers/Hugging Face models
- `supabase`: Interacting with Supabase database and storage
- `ollama`: Interacting with a local Ollama instance
- `google-generativeai`: Interacting with the Google Gemini API
- `numpy`: Numerical operations
- `python-dotenv`: Loading environment variables

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Consider updating tests and documentation along with code changes.

## License

This project is licensed under the MIT License - see the `LICENSE` file (if available) for details.

## Support

For support, feature requests, or bug reports, please open an issue in the GitHub repository. 