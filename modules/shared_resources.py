import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import streamlit as st
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Client Initializations ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Clean the model name read from environment variable
raw_model_name = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_MODEL_NAME = raw_model_name.split('#')[0].strip()

supabase: Client = None
model: SentenceTransformer = None
ollama_client: ollama.Client = None
initialization_errors = []

# Use Streamlit caching for the sentence transformer model
@st.cache_resource
def load_sentence_transformer():
    try:
        logger.info("Attempting to load Sentence Transformer model...")
        loaded_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("Sentence Transformer model loaded successfully.")
        return loaded_model
    except Exception as e:
        logger.error(f"Error loading Sentence Transformer model: {e}", exc_info=True)
        return e

# --- Initialize Supabase ---
logger.info("Attempting to initialize Supabase client...")
supabase_initialized = False

# Try with anonymous key first
if SUPABASE_URL and SUPABASE_KEY:
    try:
        logger.info("Attempting to initialize Supabase with anonymous key...")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully with anonymous key.")
        supabase_initialized = True
    except Exception as e:
        logger.warning(f"Error initializing Supabase with anonymous key: {e}")
        # Don't add to initialization_errors yet, we'll try service key next

# If anonymous key failed, try service key
if not supabase_initialized and SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        logger.info("Attempting to initialize Supabase with service role key...")
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully with service role key.")
        supabase_initialized = True
    except Exception as e:
        error_msg = f"Error initializing Supabase with service role key: {e}"
        logger.error(error_msg)
        initialization_errors.append(error_msg)

# If both failed or no credentials provided
if not supabase_initialized:
    if not (SUPABASE_URL and (SUPABASE_KEY or SUPABASE_SERVICE_KEY)):
        error_msg = "Supabase credentials (SUPABASE_URL and either SUPABASE_KEY or SUPABASE_SERVICE_KEY) not found in .env file."
        logger.warning(error_msg)
        initialization_errors.append(error_msg)
    supabase = None  # Ensure supabase is None if init fails

logger.info(f"Supabase client status: {'Initialized' if supabase_initialized else 'Failed'}")

# --- Initialize Ollama Client ---
logger.info(f"Attempting to initialize Ollama client at {OLLAMA_BASE_URL}...")
try:
    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
    logger.info("Ollama client object created. Verifying connection...")
    ollama_client.list() # Verify connection
    logger.info("Ollama client initialized and connection verified.")
except Exception as e:
    error_msg = f"Error initializing or connecting to Ollama at {OLLAMA_BASE_URL}: {e}"
    logger.error(error_msg, exc_info=True)
    ollama_client = None
    initialization_errors.append(error_msg)
logger.info(f"Ollama client status: {'Initialized' if ollama_client else 'Failed'}")

# --- Load Sentence Transformer Model ---
logger.info("Triggering Sentence Transformer model loading (cached)...")
model_load_result = load_sentence_transformer()
if isinstance(model_load_result, Exception):
    model = None
    error_msg = f"Model Loading Error: {model_load_result}"
    logger.error(error_msg)
    initialization_errors.append(error_msg)
else:
    model = model_load_result
logger.info(f"Sentence Transformer model status: {'Loaded' if model else 'Failed'}")

# --- Getter Functions ---
def get_supabase_client() -> Client | None:
    return supabase

def get_sentence_transformer_model() -> SentenceTransformer | None:
    return model

def get_ollama_client() -> ollama.Client | None:
    return ollama_client

def get_ollama_model_name() -> str:
    """Returns the configured Ollama model name."""
    return OLLAMA_MODEL_NAME

def get_initialization_errors() -> list[str]:
    return initialization_errors

# --- New Function to Fetch Projects --- Step 1
def get_projects_from_db() -> list[dict]:
    """Fetches existing projects from the Supabase database."""
    projects = []
    if not supabase:
        logger.error("Cannot fetch projects: Supabase client not available.")
        return projects

    try:
        logger.info("Fetching projects from Supabase...")
        # Chain methods directly without backslashes
        response = (
            supabase.table('projects')
            .select('id, name, created_at')
            .order('created_at', desc=True)
            .limit(100) # Limit to avoid fetching too many
            .execute()
        )

        if hasattr(response, 'data') and response.data:
            logger.info(f"Fetched {len(response.data)} projects.")
            for item in response.data:
                display_name = item.get('name') or f"Project {item.get('id')}"
                created_date = item.get('created_at', '').split('T')[0]
                if created_date:
                     display_name += f" ({created_date})"
                projects.append({'id': item.get('id'), 'display_name': display_name})
            return projects
        elif hasattr(response, 'error') and response.error:
            logger.error(f"Supabase error fetching projects: {response.error}")
            return []
        else:
            logger.warning(f"No projects found or unexpected response: {response}")
            return []

    except Exception as e:
        logger.error(f"Error fetching projects from Supabase: {e}", exc_info=True)
        return []

# --- Check Resources Function ---
def check_resources_ready():
    ready = True
    if not supabase:
        logger.warning("Supabase client not ready.")
        ready = False
    if not model:
        logger.warning("Sentence Transformer model not ready.")
        ready = False
    if not ollama_client:
        logger.warning("Ollama client not ready.")
        ready = False
    
    if ready:
        logger.info("All resources (Supabase, Sentence Transformer, Ollama) are ready.")
    else:
        logger.warning(f"One or more resources are not ready. Errors: {initialization_errors}")
        
    return ready

# Remove Streamlit call from here
# startup_errors = get_initialization_errors()
# if startup_errors:
#     st.sidebar.error("Initialization Errors:\\n" + "\\n".join(f"- {e}" for e in startup_errors))

# Optional: Call check on import
# check_resources_ready() 