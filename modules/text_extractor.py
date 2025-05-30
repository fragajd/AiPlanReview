import pytesseract
import cv2
import numpy as np
from PIL import Image
import fitz
import streamlit as st
import io
import os
import re
import logging
import hashlib
import time
from typing import List, Dict, Any, Tuple, Optional

# Import shared resources
from .shared_resources import ( # Use relative import within the package
    get_supabase_client,
    get_sentence_transformer_model,
    get_initialization_errors,
    get_ollama_client,
    get_ollama_model_name
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get resources from shared module
supabase = get_supabase_client()
model = get_sentence_transformer_model()
ollama_client = get_ollama_client()
ollama_model_name = get_ollama_model_name()
initialization_errors = get_initialization_errors()

# Set Tesseract path for Windows
if os.name == 'nt':
    # Verify the path exists before setting it
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        logger.warning(f"Tesseract executable not found at {tesseract_path}. OCR might fail.")
        # Optionally provide instructions to the user via Streamlit
        # st.warning("Tesseract not found. Please install Tesseract-OCR and ensure it's in your PATH or update the path in text_extractor.py")

# --- File Identification Helpers ---
def generate_file_hash(file_bytes: bytes) -> str:
    """Generate a hash to uniquely identify a file by its contents"""
    return hashlib.md5(file_bytes).hexdigest()

def get_file_identifier(pdf_file: io.BytesIO) -> str:
    """Get a unique identifier for a file based on name and content hash"""
    file_name = getattr(pdf_file, 'name', 'Unknown')
    
    # Get current position
    current_pos = pdf_file.tell()
    
    # Go to beginning and read content for hashing
    pdf_file.seek(0)
    file_content = pdf_file.read(8192)  # Read first 8KB for hashing
    file_hash = generate_file_hash(file_content)
    
    # Restore position
    pdf_file.seek(current_pos)
    
    return f"{file_name}_{file_hash[:10]}"

# --- Init Session State for Text Extraction ---
def init_extraction_state():
    """Initialize session state variables for text extraction tracking"""
    if "extraction_state_initialized" not in st.session_state:
        st.session_state.extraction_state_initialized = True
        st.session_state.processed_files = {}  # Dict to track processed files
        st.session_state.file_extraction_results = {}  # Store extraction results
        logger.info("Initialized extraction session state")

@st.cache_data(ttl=3600, show_spinner=False)
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

@st.cache_data(ttl=3600, show_spinner=False)
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

# --- New Technical Text Pre-processing Functions ---
@st.cache_data(ttl=3600, show_spinner=False)
def preprocess_technical_text(text: str) -> str:
    """Apply specialized pre-processing for technical architectural/engineering text."""
    if not text:
        return ""
        
    processed_text = text
    
    # Standardize units of measurement
    processed_text = standardize_measurements(processed_text)
    
    # Fix common building code references
    processed_text = standardize_code_references(processed_text)
    
    # Improve technical terminology
    processed_text = fix_technical_terminology(processed_text)
    
    # Format lists and specifications consistently
    processed_text = format_specifications(processed_text)
    
    return processed_text

@st.cache_data(ttl=3600, show_spinner=False)
def standardize_measurements(text: str) -> str:
    """Standardize and fix common measurement formats in architectural text."""
    if not text:
        return ""
        
    # Convert fraction notations to decimal (e.g., 1-1/2" → 1.5")
    text = re.sub(r'(\d+)-(\d+)/(\d+)"', lambda m: f"{int(m.group(1)) + int(m.group(2))/int(m.group(3))}\"", text)
    
    # Fix spacing in dimensions (e.g., 2' 6" → 2'-6")
    text = re.sub(r'(\d+)\'(\s*)(\d+)"', r"\1'-\3\"", text)
    
    # Standardize unit spacing (e.g., "50 mm" → "50mm", "20 psf" → "20psf")
    units = ['mm', 'cm', 'in', 'ft', 'psf', 'psi', 'ksi', 'pcf', 'sq ft', 'kg']
    for unit in units:
        text = re.sub(rf'(\d+)\s+{unit}', rf'\1{unit}', text)
    
    return text
    
@st.cache_data(ttl=3600, show_spinner=False)
def standardize_code_references(text: str) -> str:
    """Standardize building code references."""
    if not text:
        return ""
        
    # Standardize code references (e.g., "ASTM C 90" → "ASTM C90")
    text = re.sub(r'(ASTM|ANSI|ACI|AISI|IBC|IPC)\s+([A-Z])\s+(\d+)', r'\1 \2\3', text)
    
    # Standardize section references (e.g., "Sec. 4.2.1" → "Section 4.2.1")
    text = re.sub(r'(?i)(sec|sect)\.?\s+(\d+\.\d+)', r'Section \2', text)
    
    return text
    
@st.cache_data(ttl=3600, show_spinner=False)
def fix_technical_terminology(text: str) -> str:
    """Fix common technical terms that might be misspelled by OCR."""
    if not text:
        return ""
        
    # Dictionary of common OCR errors in technical terms
    corrections = {
        "relnforced": "reinforced",
        "concreie": "concrete",
        "concrele": "concrete",
        "structurai": "structural",
        "sieel": "steel",
        "steei": "steel",
        "specificaiions": "specifications",
        "lnsulation": "insulation",
        "lnstallation": "installation",
        "fastenlng": "fastening"
    }
    
    for error, correction in corrections.items():
        text = re.sub(rf'\b{error}\b', correction, text, flags=re.IGNORECASE)
    
    return text
    
@st.cache_data(ttl=3600, show_spinner=False)
def format_specifications(text: str) -> str:
    """Format specification lists and numbered items consistently."""
    if not text:
        return ""
        
    # Format numbered specifications (e.g., "1 Steel shall..." → "1. Steel shall...")
    text = re.sub(r'(?<!\d)(\d+)(?!\d|\.)(\s+[A-Z])', r'\1.\2', text)
    
    # Format bullet points consistently
    text = re.sub(r'(?<=\n)[\*\-•⦁◦] ?', '• ', text)
    
    return text

# --- Enhanced Chunking Function for Better Context ---
@st.cache_data(ttl=3600, show_spinner=False)
def chunk_text_with_context(text, chunk_size=1500, overlap=300):
    """
    Splits text into overlapping chunks with improved context preservation.
    Tries to split at paragraph or sentence boundaries when possible.
    
    Args:
        text: The text to split
        chunk_size: Target size for each chunk
        overlap: Minimum overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Split by paragraphs first (preserve paragraph structure)
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = []
    current_size = 0
    
    # Process paragraphs, preserving whole paragraphs when possible
    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        paragraph_size = len(paragraph_words)
        
        # If a single paragraph is too large, we need to split it
        if paragraph_size > chunk_size:
            # If we have content in the current chunk, finish it first
            if current_size > 0:
                chunks.append(" ".join(current_chunk))
                # Keep overlap with previous chunk for context
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:]
                current_size = len(current_chunk)
            
            # Now split the large paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sentence_words = sentence.split()
                sentence_size = len(sentence_words)
                
                # If adding this sentence exceeds the chunk size
                if current_size + sentence_size > chunk_size and current_size > 0:
                    chunks.append(" ".join(current_chunk))
                    # Keep overlap with previous chunk for context
                    overlap_start = max(0, len(current_chunk) - overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_size = len(current_chunk)
                
                # Add sentence to current chunk
                current_chunk.extend(sentence_words)
                current_size += sentence_size
        else:
            # If adding this paragraph exceeds the chunk size
            if current_size + paragraph_size > chunk_size and current_size > 0:
                chunks.append(" ".join(current_chunk))
                # Keep overlap with previous chunk for context
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:]
                current_size = len(current_chunk)
            
            # Add paragraph to current chunk
            current_chunk.extend(paragraph_words)
            current_size += paragraph_size
    
    # Add the last chunk if it has content
    if current_size > 0:
        chunks.append(" ".join(current_chunk))
    
    # Filter out very short chunks
    return [chunk for chunk in chunks if len(chunk.split()) > 30]

# --- Cached Text Enhancement Function ---
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_enhance_text(text_to_enhance: str) -> str:
    """Cached version of text enhancement using either local Ollama or Google Gemini API."""
    
    model_preference = st.session_state.get('model_preference', 'local') # Default to local
    google_api_key = st.session_state.get('google_api_key', None)

    if not text_to_enhance or len(text_to_enhance.strip()) < 20: # Don't enhance very short texts
        return text_to_enhance

    # Enhanced detailed prompt (remains the same for both models)
    prompt = f"""You are an expert assistant specialized in enhancing and correcting text from architectural and engineering plans specifically for building permit applications and structural engineering documentation.

Task: Improve the OCR-extracted text below to enhance readability and accuracy while preserving all technical information. 

Focus on the following:
1. Fix formatting and structure of technical specifications, building codes, and material requirements
2. Standardize measurement formats (e.g., "1'-6"" instead of "1 ft 6 in")
3. Correct technical terminology and specialized engineering vocabulary
4. Ensure building code references are properly formatted (e.g., "ASTM C90" not "ASTM C 90")
5. Preserve all numerical values, dimensions, and technical specifications exactly
6. Format lists, bullet points, and numbered specifications consistently
7. Maintain paragraph breaks and section structure
8. Preserve all compliance statements and regulatory references

IMPORTANT: Do NOT add explanatory text, commentary, or alter the meaning. Return ONLY the enhanced version of the text.

Text to enhance:

{text_to_enhance}"""

    enhanced_text = text_to_enhance # Default to original text

    try:
        if model_preference == 'local':
            # --- Ollama Logic --- 
            if not ollama_client:
                logger.warning("Ollama client not initialized. Skipping local text enhancement.")
                return text_to_enhance
                
            logger.info(f"Sending text (length: {len(text_to_enhance)}) to local Ollama model {ollama_model_name} for enhancement...")
            
            # Chunking logic for Ollama (remains the same)
            if len(text_to_enhance.split()) > 1000:
                chunks = chunk_text_with_context(text_to_enhance)
                enhanced_chunks = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Ollama: Processing chunk {i+1}/{len(chunks)} ({len(chunk.split())} words)...")
                    chunk_prompt = prompt.replace(text_to_enhance, chunk)
                    response = ollama_client.chat(
                        model="gemma3:4b", 
                        messages=[{'role': 'user', 'content': chunk_prompt}],
                        stream=False
                    )
                    enhanced_chunk = response['message']['content'].strip()
                    enhanced_chunks.append(enhanced_chunk)
                enhanced_text = "\n\n".join(enhanced_chunks)
            else:
                # Process shorter text as a single chunk with Ollama
                response = ollama_client.chat(
                    model="gemma3:4b", 
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=False
                )
                enhanced_text = response['message']['content'].strip()
            
            logger.info(f"Received enhanced text (length: {len(enhanced_text)}) from Ollama.")

        elif model_preference == 'api':
            # --- Google Gemini API Logic --- 
            if not google_api_key:
                logger.warning("Google API Key not provided. Skipping API text enhancement.")
                st.warning("Google API Key needed for enhancement. Please provide it in the sidebar.")
                return text_to_enhance
            
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_api_key)
                gemini_model_name = 'gemini-2.5-flash-preview-05-20' # Or choose another appropriate model
                model = genai.GenerativeModel(gemini_model_name)
                
                logger.info(f"Sending text (length: {len(text_to_enhance)}) to Google Gemini model {gemini_model_name} for enhancement...")
                
                # Chunking logic for Gemini (similar structure)
                if len(text_to_enhance.split()) > 1000: # Adjust threshold if needed for API
                    chunks = chunk_text_with_context(text_to_enhance)
                    enhanced_chunks = []
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Gemini: Processing chunk {i+1}/{len(chunks)} ({len(chunk.split())} words)...")
                        chunk_prompt = prompt.replace(text_to_enhance, chunk)
                        # Note: Gemini API might handle longer prompts directly, adjust if needed
                        response = model.generate_content(chunk_prompt)
                        # TODO: Add more robust error checking for Gemini response (e.g., response.prompt_feedback)
                        enhanced_chunks.append(response.text)
                    enhanced_text = "\n\n".join(enhanced_chunks)
                else:
                    # Process shorter text as a single call with Gemini
                    response = model.generate_content(prompt)
                    # TODO: Add more robust error checking for Gemini response
                    enhanced_text = response.text
                    
                logger.info(f"Received enhanced text (length: {len(enhanced_text)}) from Gemini.")
                
            except ImportError:
                 logger.error("google.generativeai library not installed. Cannot use Gemini API.")
                 st.error("Google AI library not found. Please install it (`pip install google-generativeai`).")
                 return text_to_enhance # Fallback to original text
            except Exception as api_error:
                 logger.error(f"Error calling Google Gemini API: {api_error}", exc_info=True)
                 st.error(f"Error connecting to Google Gemini API: {api_error}")
                 return text_to_enhance # Fallback to original text
        else:
             logger.warning(f"Unknown model preference: {model_preference}. Defaulting to original text.")
             return text_to_enhance

        # Sanity check (remains the same)
        if not enhanced_text or len(enhanced_text) < len(text_to_enhance) * 0.5:
             logger.warning("Enhancement resulted in empty or significantly shorter text. Falling back to original.")
             return text_to_enhance
             
        return enhanced_text
        
    except Exception as e:
        logger.error(f"Generic error during text enhancement ({model_preference}): {e}", exc_info=True)
        return text_to_enhance # Fallback in case of unexpected errors

# --- Text Enhancement Wrapper Function --- 
def enhance_text(text_to_enhance: str) -> str:
    """Uses the configured model (Ollama or Gemini) to enhance extracted OCR text."""
    
    # Check session state for preference (needed for warnings/guidance)
    model_preference = st.session_state.get('model_preference', 'local')
    google_api_key = st.session_state.get('google_api_key', None)

    if model_preference == 'api' and not google_api_key:
        st.warning("API mode selected, but no Google API Key found. Skipping enhancement.")
        return text_to_enhance
        
    if not text_to_enhance or len(text_to_enhance.strip()) < 20:  # Don't enhance very short texts
        return text_to_enhance
    
    # First apply technical preprocessing (remains the same)
    preprocessed_text = preprocess_technical_text(text_to_enhance)
    
    # Use cached version for the actual enhancement (which now handles model choice)
    return _cached_enhance_text(preprocessed_text)

# --- Check if file has been processed ---
def is_file_processed(file_id: str, project_id: int) -> bool:
    """Check if a file has already been processed for the current project"""
    if file_id and project_id:
        processed_key = f"{file_id}_{project_id}"
        return processed_key in st.session_state.processed_files
    return False

# --- Store file processing result ---
def mark_file_as_processed(file_id: str, project_id: int, extracted_text: str) -> None:
    """Mark a file as processed and store its extracted text"""
    if file_id and project_id:
        processed_key = f"{file_id}_{project_id}"
        st.session_state.processed_files[processed_key] = True
        st.session_state.file_extraction_results[processed_key] = extracted_text
        logger.info(f"Marked file {file_id} as processed for project {project_id}")

# --- Get stored extraction result ---
def get_stored_extraction_result(file_id: str, project_id: int) -> Optional[str]:
    """Get previously stored extraction result for a file"""
    if file_id and project_id:
        processed_key = f"{file_id}_{project_id}"
        return st.session_state.file_extraction_results.get(processed_key)
    return None

# --- Supabase Project Creation ---
def _create_supabase_project(project_name: str = "Default Project") -> int | None:
    """Creates a new project entry in Supabase and returns its ID."""
    if not supabase:
        st.error("Supabase client not available. Cannot create project.")
        logger.error("Supabase client not initialized, cannot create project.")
        return None
    try:
        logger.info(f"Creating Supabase project with name: {project_name}")
        response = supabase.table('projects').insert({"name": project_name}).execute()
        
        if hasattr(response, 'data') and response.data and len(response.data) > 0:
            project_id = response.data[0]['id']
            logger.info(f"Successfully created project with ID: {project_id}")
            # Store in session state IMMEDIATELY upon successful creation
            st.session_state.current_project_id = project_id 
            return project_id
        else:
            error_info = getattr(response, 'error', 'Unknown error')
            st.error(f"Failed to create project in Supabase: {error_info}")
            logger.error(f"Failed to create Supabase project. Response: {response}")
            # Ensure session state is clear if creation fails
            if 'current_project_id' in st.session_state: 
                del st.session_state['current_project_id']
            return None
    except Exception as e:
        st.error(f"Error creating Supabase project: {e}")
        logger.error(f"Exception during Supabase project creation: {e}", exc_info=True)
        # Ensure session state is clear if creation fails
        if 'current_project_id' in st.session_state: 
            del st.session_state['current_project_id']
        return None

# Cache embedding storage
@st.cache_data(ttl=1800, show_spinner=False)
def _cached_store_embeddings(project_id: int, chunks_key: str, chunks: List[str], file_name: str) -> bool:
    """Cached version of embedding storage"""
    if not project_id:
        logger.error("_cached_store_embeddings called without project_id.")
        return False
    if not file_name:
        logger.error("_cached_store_embeddings called without file_name.")
        return False
    if not supabase or not model or not chunks:
        return False

    try:
        logger.info(f"Generating embeddings for {len(chunks)} chunks (Project ID: {project_id}, File: {file_name})...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        logger.info(f"Generated {len(embeddings)} embeddings for Project ID: {project_id}, File: {file_name}.")

        data_to_insert = [
            {
                "content": chunk,
                "embedding": embedding.tolist(),
                "project_id": project_id,
                "file_name": file_name
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        logger.info(f"Storing {len(data_to_insert)} chunks and embeddings in Supabase (Project ID: {project_id}, File: {file_name})...")
        response = supabase.table('documents').insert(data_to_insert).execute()

        if hasattr(response, 'data') and response.data:
             logger.info(f"Successfully inserted {len(response.data)} items into Supabase for Project ID: {project_id}, File: {file_name}.")
             return True
        elif hasattr(response, 'error') and response.error:
             logger.error(f"Supabase insertion error (Project ID: {project_id}, File: {file_name}): {response.error}")
             return False
        else:
             logger.warning(f"Unexpected Supabase response (Project ID: {project_id}, File: {file_name}): {response}")
             return False

    except Exception as e:
        logger.error(f"Error storing embeddings (Project ID: {project_id}, File: {file_name}): {e}", exc_info=True)
        return False

# --- Modified store_embeddings --- 
def store_embeddings(project_id: int, text_chunks: List[str], file_name: str) -> bool:
    """Generates embeddings and stores them in Supabase, associated with a project_id and file_name."""
    if not project_id:
        st.error("Project ID is missing. Cannot store embeddings.")
        logger.error("store_embeddings called without project_id.")
        return False
    if not file_name:
        st.error("File name is missing. Cannot store embeddings.")
        logger.error("store_embeddings called without file_name.")
        return False
    if not supabase or not model or not text_chunks:
        # Logging handled internally by the check
        return False

    try:
        st.info(f"Generating embeddings for {len(text_chunks)} chunks from summary (Project ID: {project_id}, File: {file_name})...")
        
        # Create a chunks key for caching (based on summary chunks)
        chunks_key = hashlib.md5("".join(text_chunks[:5]).encode()).hexdigest()
        
        # Call cached version, passing file_name
        success = _cached_store_embeddings(project_id, chunks_key, text_chunks, file_name)
        
        if success:
            st.success(f"Successfully stored {len(text_chunks)} summary chunks and embeddings for {file_name}!")
        else:
            st.error(f"Failed to store summary embeddings in the database for {file_name}.")
            
        return success

    except Exception as e:
        logger.error(f"Error in store_embeddings wrapper (Project ID: {project_id}, File: {file_name}): {e}", exc_info=True)
        st.error(f"Error storing summary embeddings for {file_name}: {e}")
        return False

# --- Cached PDF extraction function ---
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_extract_text_from_pdf(
    file_hash: str,
    pdf_content: bytes,
    use_ocr: bool = True,
    enhance_resolution: bool = True
) -> List[str]:
    """
    Cached function to extract text from PDF content.
    Returns a list of text from each page.
    """
    extracted_pages = []
    
    try:
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        num_pages = pdf_document.page_count
        
        for page_num in range(num_pages):
            page = pdf_document[page_num]
            embedded_text = page.get_text().strip()
            page_text = ""

            if len(embedded_text) > 50 and not use_ocr:
                page_text = clean_ocr_text(embedded_text)
            else:
                # OCR process
                try:
                    pix = page.get_pixmap(alpha=False, dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    processed_img = preprocess_image(opencv_img, enhance_resolution)
                    custom_config = r'--oem 3 --psm 6'
                    ocr_text = pytesseract.image_to_string(processed_img, config=custom_config, lang='eng')
                    ocr_text = clean_ocr_text(ocr_text)
                    if len(ocr_text) < 20 and len(embedded_text) > len(ocr_text):
                        page_text = clean_ocr_text(embedded_text)
                    else:
                        page_text = ocr_text
                except Exception as ocr_error:
                    logger.error(f"OCR Error (File hash: {file_hash}, Page: {page_num + 1}): {ocr_error}")
                    if embedded_text:
                        page_text = clean_ocr_text(embedded_text)
                    else:
                        page_text = ""
            
            if page_text:
                extracted_pages.append(page_text)

        pdf_document.close()
        return extracted_pages
        
    except Exception as e:
        logger.error(f"Error in _cached_extract_text_from_pdf (File hash: {file_hash}): {e}", exc_info=True)
        return []

# --- Modified extract_text_from_single_pdf function ---
def _extract_text_from_single_pdf(
    pdf_file: io.BytesIO, 
    project_id: int, 
    use_ocr=True, 
    enhance_resolution=True, 
    process_embeddings=True,
    enhance_with_gemma=False
) -> Tuple[str, bool]:
    """
    Extracts text from a SINGLE PDF, optionally enhances with Gemma, chunks it,
    generates embeddings, and stores them in Supabase under the given project_id.
    Returns the (potentially enhanced) extracted text and a boolean indicating embedding success.
    """
    full_extracted_text = ""
    embedding_success = False
    file_name = getattr(pdf_file, 'name', 'Unknown File')
    status_container = st.container()

    # Initialize session state for extraction tracking
    init_extraction_state()
    
    # Generate file identifier
    file_id = get_file_identifier(pdf_file)
    
    # Check if this file has already been processed for this project
    if is_file_processed(file_id, project_id):
        logger.info(f"File {file_name} already processed for project {project_id}, using cached result")
        status_container.success(f"Using cached extraction for {file_name}")
        
        # Get stored result
        stored_text = get_stored_extraction_result(file_id, project_id)
        if stored_text:
            return stored_text, True
    
    # If not processed or no stored result, continue with extraction
    try:
        status_container.info(f"Processing file: {file_name}...")
        pdf_file.seek(0)
        pdf_content = pdf_file.read()
        pdf_file.seek(0)
        
        # Get file hash for caching
        file_hash = generate_file_hash(pdf_content)
        
        # Use cached extraction
        extracted_pages = _cached_extract_text_from_pdf(
            file_hash=file_hash,
            pdf_content=pdf_content,
            use_ocr=use_ocr,
            enhance_resolution=enhance_resolution
        )
        
        # Join pages and apply basic processing
        initial_extracted_text = "\n\n".join(extracted_pages) if extracted_pages else ""
        
        if not initial_extracted_text:
            status_container.warning(f"No text extracted from {file_name}")
            return "", False
        
        # Apply basic technical preprocessing right after extraction
        preprocessed_text = preprocess_technical_text(initial_extracted_text)
        full_extracted_text = preprocessed_text # Start with preprocessed text
        
        logger.info(f"File: {file_name} - Initial extracted text length: {len(initial_extracted_text)} chars.")
        logger.info(f"File: {file_name} - After basic preprocessing: {len(preprocessed_text)} chars.")

        # --- Gemma Enhancement Step --- (Step 11)
        if enhance_with_gemma and preprocessed_text: # Use preprocessed text for check
            # Determine which model is being used for logging/display
            model_used_for_enhancement = st.session_state.get('model_preference', 'local')
            status_container.info(f"File: {file_name} - Enhancing text with {model_used_for_enhancement.upper()}...")
            
            # Call the updated enhance_text function
            enhanced_text = enhance_text(preprocessed_text) 
            
            # Display comparison
            with status_container.expander(f"Show {model_used_for_enhancement.upper()} Enhancement Comparison", expanded=True):
                col1_comp, col2_comp = st.columns(2)
                with col1_comp:
                    st.text_area("Before Enhancement", preprocessed_text, height=200, key=f"before_{file_id}")
                with col2_comp:
                    st.text_area("After Enhancement", enhanced_text, height=200, key=f"after_{file_id}")
            
            if enhanced_text != preprocessed_text:
                 logger.info(f"File: {file_name} - Text enhanced by {model_used_for_enhancement.upper()}. New length: {len(enhanced_text)} chars.")
                 full_extracted_text = enhanced_text # Update text to be used for embedding
            else:
                 logger.info(f"File: {file_name} - Text enhancement by {model_used_for_enhancement.upper()} did not change the text or failed.")
                 # Keep full_extracted_text as preprocessed_text
        elif enhance_with_gemma:
             logger.warning(f"File: {file_name} - Skipping enhancement as no initial text was extracted.")
        
        # --- Generate Document Summary ---
        # Always generate a document summary, regardless of text enhancement
        status_container.info(f"Generating document summary for {file_name}...")
        document_summary = generate_document_summary(full_extracted_text, file_name)
        
        # Display generated summary
        with status_container.expander("Document Summary", expanded=True):
            st.markdown(document_summary)
            
        # --- Chunking, Embedding, and Storage of SUMMARY ---
        if process_embeddings and document_summary and not document_summary.startswith("Summary generation") and supabase and model:
            status_container.info(f"File: {file_name} - Chunking SUMMARY...") 
            # Chunk the summary with smaller settings
            summary_chunks = chunk_text_with_context(document_summary, chunk_size=250, overlap=50) 
            if summary_chunks:
                # Pass file_name to store_embeddings
                embedding_success = store_embeddings(project_id, summary_chunks, file_name)
            else:
                status_container.warning(f"File: {file_name} - Summary was too short to chunk.")
                embedding_success = False # No chunks to embed
        elif not process_embeddings: 
            logger.info(f"File: {file_name} - Embedding skipped by settings.")
            embedding_success = False # Embeddings not processed
        elif not document_summary or document_summary.startswith("Summary generation"): 
            status_container.warning(f"File: {file_name} - No valid summary generated, skipping embedding.")
            embedding_success = False # No summary to embed
        else: 
            status_container.warning(f"File: {file_name} - Supabase/Model not ready, skipping embedding.")
            embedding_success = False # Cannot process embeddings
        
        # Mark file as processed and store result (storing full text for potential future use, not embedding)
        mark_file_as_processed(file_id, project_id, full_extracted_text)

    except fitz.fitz.FileDataError:
        status_container.error(f"File: {file_name} - Invalid or corrupted PDF file.")
    except pytesseract.TesseractNotFoundError:
         status_container.error("Tesseract not installed or not found. Please check installation.")
         logger.error("Tesseract executable not found.")
    except Exception as e:
        status_container.error(f"File: {file_name} - Error during processing: {str(e)}")
        logger.error(f"Error processing PDF (File: {file_name}): {str(e)}", exc_info=True)

    return full_extracted_text, embedding_success

# --- Modified render_text_tab --- Step 8 (Part 2)
def render_text_tab(uploaded_files: List[io.BytesIO]) -> str:
    """Processes a list of uploaded PDF files (if any), creates project/embeddings,
       or displays info for a loaded project."""
    
    # Initialize session state
    init_extraction_state()
    
    # Determine mode based on session state (set in app.py)
    app_mode = st.session_state.get('app_mode', 'Upload')
    project_id = st.session_state.get('current_project_id', None)
    
    combined_text = ""
    all_texts = []

    col1, col2 = st.columns([3, 1])

    with col2:
        # Only show settings in Upload mode
        if app_mode == 'Upload':
            st.write("Processing Settings")
            force_ocr = st.checkbox("Force OCR", value=False, key="force_ocr_checkbox",
                                   help="Use OCR even if embedded text is found")
            enhance_res = st.checkbox("Enhance Resolution", value=True, key="enhance_res_checkbox",
                                    help="Increase image resolution for better OCR")
            process_embeddings = st.checkbox("Process & Store Embeddings", value=True, key="process_embed_checkbox",
                                    help="Generate embeddings and store in Supabase")
            enhance_with_gemma_cb = st.checkbox("Enhance Text with AI Model", value=True, key="enhance_gemma_checkbox",
                                              help="Use the selected AI model (Ollama/Gemini) to clean/enhance OCR text (can be slow).")
        else:
            st.write("Project Info")
            if project_id:
                st.info(f"Currently loaded Project ID: {project_id}")
            else:
                st.warning("No project selected.")

    with col1:
        # --- Upload Mode Logic ---
        if app_mode == 'Upload':
            st.subheader("Text Extraction & Processing")
            if not uploaded_files:
                st.info("Upload files using the sidebar widget to begin processing.")
                return ""
                
            # Project creation logic
            if project_id is None:
                 st.info("No active project found, creating a new one...")
                 project_name = f"Project - {uploaded_files[0].name}" if uploaded_files else "Default Project"
                 created_id = _create_supabase_project(project_name)
                 if created_id is None:
                     st.error("Failed to create a project. Cannot proceed.")
                     return ""
                 else:
                     project_id = created_id
                     st.success(f"Created and using new project (ID: {project_id}).")
            # Ensure we definitely have a project ID 
            if project_id is None: return "" # Abort if still None

            overall_progress = st.progress(0)
            files_processed_count = 0
            total_files = len(uploaded_files)
            all_texts = []

            st.write(f"Processing {total_files} file(s) for Project ID: {project_id}")

            for pdf_file in uploaded_files:
                file_name = getattr(pdf_file, 'name', f'File {files_processed_count+1}')
                with st.spinner(f"Processing {file_name}..."):
                    extracted_text, _ = _extract_text_from_single_pdf(
                        pdf_file,
                        project_id=project_id, # Pass the confirmed project ID
                        use_ocr=force_ocr,
                        enhance_resolution=enhance_res,
                        process_embeddings=process_embeddings,
                        enhance_with_gemma=enhance_with_gemma_cb
                    )
                    if extracted_text:
                        all_texts.append(extracted_text)
                
                files_processed_count += 1
                overall_progress.progress(files_processed_count / total_files)

            overall_progress.empty() # Clear progress bar
            combined_text = "\n\n--- End of File ---\n\n".join(all_texts)
            st.subheader("Combined Extracted Text (Current Upload)")
            st.text_area("Combined Text Preview", combined_text, height=400)
            logger.info(f"Processing complete for Project ID: {project_id}. Combined text length: {len(combined_text)}")

        # --- Load Mode Logic ---
        elif app_mode == 'Load':
            st.subheader("Project Information")
            if project_id:
                 st.success(f"Loaded Project ID: {project_id}")
                 st.write("You can now use the Q&A tab to chat with the documents associated with this project.")
                 # We are not displaying extracted text in load mode for now.
                 combined_text = "" # Explicitly set to empty
            else:
                 st.warning("No project is currently loaded. Please select one from the sidebar.")
                 combined_text = ""
            
    # Return combined text (relevant for Upload mode, empty for Load mode)
    return combined_text 

# --- Cached Document Summarization Function ---
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_generate_document_summary(text_to_summarize: str, document_name: str) -> str:
    """
    Cached function to generate a document summary using AI.
    Uses either local Ollama/Gemma or Google Gemini API based on user preferences.
    
    Args:
        text_to_summarize: Document text to summarize
        document_name: Name of the document for context
        
    Returns:
        Summary text that provides a high-level overview of the document
    """
    if not text_to_summarize or len(text_to_summarize.strip()) < 100:
        return "Insufficient text to generate a meaningful summary."
        
    model_preference = st.session_state.get('model_preference', 'local')
    google_api_key = st.session_state.get('google_api_key', None)
    
    # Detailed, section-based information extraction prompt
    prompt = f"""You are an expert technical analyst tasked with creating a DETAILED INFORMATION EXTRACTION from an architectural or engineering document named "{document_name}".

Your goal is to process the document section by section and extract the most critical information.

Instructions:
1.  First, try to identify the main sections of the document (e.g., Introduction, Specifications, Load Calculations, Material Requirements, Compliance Statements, Conclusion, Appendices, etc.).
2.  For EACH identified section, provide a concise summary of that section's purpose AND extract the most important technical details, data, specifications, measurements, code references, and key findings presented within that section.
3.  Structure your output clearly, perhaps using headings for each section you identify.
4.  Be comprehensive. The goal is NOT a brief overview, but a detailed extraction of core information that would be useful for answering specific technical questions about the document later.
5.  If the document is short or does not have clearly defined sections, then provide a detailed extraction of all key information found.
6.  Preserve all numerical values, dimensions, units, and technical terminology accurately.
7.  Focus on factual information extraction. Avoid interpretation or adding information not present in the text.

Document text:
{text_to_summarize}

DETAILED INFORMATION EXTRACTION:"""
    
    try:
        if model_preference == 'local':
            # Use Ollama/Gemma
            if not ollama_client:
                logger.warning("Ollama client not initialized. Cannot generate document summary.")
                return "Summary generation unavailable: Ollama client not initialized."
                
            logger.info(f"Generating document summary for '{document_name}' with Ollama...")
            
            # Chunk if necessary (for very large documents)
            if len(text_to_summarize) > 12000: # Adjusted threshold for potentially longer prompt/output
                # Only use the beginning and end portions for summary if very large
                beginning = text_to_summarize[:6000]
                end = text_to_summarize[-6000:]
                # Ensure the prompt is applied to the shortened text
                current_prompt = prompt.replace(text_to_summarize, beginning + "\n\n[...middle content omitted...]\n\n" + end)
            else:
                current_prompt = prompt
                
            response = ollama_client.chat(
                model="gemma3:4b",
                messages=[{'role': 'user', 'content': current_prompt}],
                stream=False
            )
            
            summary = response['message']['content'].strip()
            logger.info(f"Generated detailed information extraction with Ollama for '{document_name}'")
            return summary
            
        elif model_preference == 'api':
            # Use Google Gemini API
            if not google_api_key:
                logger.warning("Google API Key not provided. Cannot generate document summary.")
                return "Summary generation unavailable: Google API Key not provided."
                
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_api_key)
                
                gemini_model_name = 'gemini-1.5-flash-latest'
                model = genai.GenerativeModel(gemini_model_name)
                
                logger.info(f"Generating document summary for '{document_name}' with Gemini API...")
                
                # Chunk if necessary (for very large documents)
                if len(text_to_summarize) > 12000: # Adjusted threshold
                    # Only use the beginning and end portions for summary if very large
                    beginning = text_to_summarize[:6000]
                    end = text_to_summarize[-6000:]
                    # Ensure the prompt is applied to the shortened text
                    current_prompt = prompt.replace(text_to_summarize, beginning + "\n\n[...middle content omitted...]\n\n" + end)
                else:
                    current_prompt = prompt
                
                response = model.generate_content(current_prompt)
                summary = response.text.strip()
                logger.info(f"Generated detailed information extraction with Gemini API for '{document_name}'")
                return summary
                
            except ImportError:
                logger.error("google.generativeai library not installed. Cannot use Gemini API.")
                return "Summary generation unavailable: Google AI library not installed."
            except Exception as api_error:
                logger.error(f"Error calling Google Gemini API for document summary: {api_error}", exc_info=True)
                return f"Summary generation failed: {api_error}"
                
        else:
            logger.warning(f"Unknown model preference: {model_preference}. Cannot generate document summary.")
            return "Summary generation unavailable: Unknown model preference."
            
    except Exception as e:
        logger.error(f"Error generating document summary: {e}", exc_info=True)
        return f"Summary generation failed: {e}"
        
# --- Public Document Summarization Function ---
def generate_document_summary(text: str, document_name: str) -> str:
    """
    Public function to generate a document summary.
    Uses cached implementation for efficiency.
    
    Args:
        text: Document text to summarize
        document_name: Name of the document
        
    Returns:
        Document summary
    """
    if not text or len(text.strip()) < 100:
        return "Insufficient text to generate a meaningful summary."
        
    return _cached_generate_document_summary(text, document_name) 