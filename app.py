import logging
import os

# Set environment variable BEFORE importing streamlit
os.environ["STREAMLIT_WATCH_MODULES"] = "false"

# Import streamlit first
import streamlit as st

# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide") 

# Now setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Starting app.py execution...")

# Import other modules AFTER st.set_page_config
import io
from modules.text_extractor import render_text_tab
logger.info("Imported render_text_tab")
from modules.image_handler import render_image_tab
logger.info("Imported render_image_tab")
from modules.qa_analyzer import render_qa_tab
logger.info("Imported render_qa_tab")
from modules.shared_resources import get_projects_from_db, get_initialization_errors
logger.info("Imported shared_resources functions")
logger.info("Imports completed.")


def main():
    logger.info("Entered main() function.")
    
    # Set title (this is not the page config)
    st.title("Architectural & Engineering AI Plan Reviewer")
    logger.info("Called st.title().")

    # --- Display Initialization Errors --- 
    logger.info("Checking for initialization errors...")
    startup_errors = get_initialization_errors()
    if startup_errors:
        st.sidebar.error("Initialization Errors:\\n" + "\\n".join(f"- {e}" for e in startup_errors))

    # --- Session State Initialization ---
    # Initialize session state variables if they don't exist
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'Upload' # Default mode
    if 'uploaded_files_list' not in st.session_state:
        st.session_state.uploaded_files_list = []
    if 'combined_text' not in st.session_state:
        st.session_state.combined_text = ""
    if 'processed_file_names' not in st.session_state:
        st.session_state.processed_file_names = set()
    if '_last_uploaded_file_names' not in st.session_state:
        st.session_state._last_uploaded_file_names = set()
    if 'current_project_id' not in st.session_state:
        st.session_state.current_project_id = None
    if 'available_projects' not in st.session_state:
        st.session_state.available_projects = [] # Store fetched projects

    # --- Mode Selection --- 
    st.sidebar.title("Mode")
    mode = st.sidebar.radio(
        "Select Operation Mode:", 
        ('Upload New Files', 'Load Existing Project'), 
        key='app_mode_radio',
        # Use index=0 for Upload, index=1 for Load to map to session state
        index=0 if st.session_state.app_mode == 'Upload' else 1,
        on_change=lambda: st.session_state.update(app_mode='Upload' if st.session_state.app_mode_radio == 'Upload New Files' else 'Load')
    )
    
    st.sidebar.markdown("---") # Separator

    # --- AI Model Preference --- 
    st.sidebar.title("AI Settings")
    model_options = {'Local (Ollama/Gemma)': 'local', 'API (Google Gemini)': 'api'}
    # Find the index corresponding to the current session state value
    current_pref_value = st.session_state.get('model_preference', 'local')
    options_list = list(model_options.keys())
    pref_index = 0
    try:
        pref_index = options_list.index([k for k, v in model_options.items() if v == current_pref_value][0])
    except IndexError:
        pass # Keep index 0 if value not found
    
    selected_model_label = st.sidebar.radio(
        "Select AI Model:", 
        options=options_list,
        key='model_preference_radio',
        index=pref_index,
        help="Choose between running a local model (requires Ollama) or using the Google Gemini API (requires API key)."
    )
    # Update session state when radio button changes
    st.session_state.model_preference = model_options[selected_model_label]

    # --- Google API Key Input (Conditional) ---
    if st.session_state.model_preference == 'api':
        api_key = st.sidebar.text_input(
            "Google API Key:", 
            type="password",
            key='google_api_key_input',
            help="Enter your Google API key for Gemini.",
            value=st.session_state.get('google_api_key', '') # Pre-fill if already in state
        )
        # Update session state if the input changes
        if api_key != st.session_state.get('google_api_key', ''):
            st.session_state.google_api_key = api_key
    
    st.sidebar.markdown("---") # Separator
    
    # --- UI based on Mode --- 
    if st.session_state.app_mode == 'Upload':
        st.sidebar.info("Upload one or more PDF files below.")
        # --- File Uploader (Only in Upload mode) ---
        uploaded_files = st.file_uploader(
            "Upload your plan(s) here (PDF format):",
            type="pdf",
            accept_multiple_files=True
        )
        
        # Reset project ID if files are uploaded in Upload mode
        if uploaded_files:
            current_file_names = set(f.name for f in uploaded_files)
            if current_file_names != st.session_state._last_uploaded_file_names:
                st.sidebar.warning("New files detected. Previous project context cleared.")
                if 'current_project_id' in st.session_state: 
                    del st.session_state['current_project_id']
                st.session_state.uploaded_files_list = uploaded_files
                st.session_state._last_uploaded_file_names = current_file_names
                st.session_state.combined_text = "" # Reset text
                st.session_state.processed_file_names = set()
                # Trigger rerun to process the new files
                # st.experimental_rerun()
        else:
            # Clear state if uploader becomes empty while in Upload mode
            st.session_state.uploaded_files_list = []
            st.session_state._last_uploaded_file_names = set()
            # Optionally clear project ID too, or keep it if user might re-upload same batch?
            # For now, let's keep project ID unless explicitly cleared by loading/new upload.
            
    elif st.session_state.app_mode == 'Load':
        st.sidebar.info("Select a previously processed project.")
        uploaded_files = None # Ensure this is None in Load mode
        st.session_state.uploaded_files_list = [] # Clear uploaded files list
        
        # Fetch projects if not already fetched or if mode just changed
        if not st.session_state.available_projects:
             with st.spinner("Loading project list..."):
                 st.session_state.available_projects = get_projects_from_db()
        
        if st.session_state.available_projects:
            project_options = {p['display_name']: p['id'] for p in st.session_state.available_projects}
            # Add a placeholder option
            options_list = ["Select a Project..."] + list(project_options.keys())
            
            selected_project_name = st.selectbox(
                "Load Project:", 
                options=options_list,
                index=0 # Default to placeholder
            )
            
            if selected_project_name != "Select a Project...":
                selected_project_id = project_options[selected_project_name]
                # Update session state only if the selection changed
                if st.session_state.current_project_id != selected_project_id:
                    st.session_state.current_project_id = selected_project_id
                    st.session_state.combined_text = "" # Clear any old text
                    st.session_state.processed_file_names = set() # Clear processed files
                    st.session_state.messages = [] # Clear chat history when loading project
                    st.success(f"Loaded Project ID: {selected_project_id}")
                    # st.experimental_rerun() # Rerun might be needed to update tabs
            else:
                # If placeholder is selected, clear current project ID
                if st.session_state.current_project_id is not None:
                     st.session_state.current_project_id = None
                     st.session_state.messages = [] # Clear chat history

        else:
            st.warning("No previous projects found in the database.")

    # --- Determine if Tabs Should Be Shown --- 
    show_tabs = False
    if st.session_state.app_mode == 'Upload' and st.session_state.uploaded_files_list:
        show_tabs = True
    elif st.session_state.app_mode == 'Load' and st.session_state.current_project_id is not None:
        show_tabs = True

    # --- Display Tabs --- 
    if show_tabs:
        tab_names = ["Text Extraction & Processing", "Images", "Q&A"]
        # Adjust tab names/content based on mode maybe?
        if st.session_state.app_mode == 'Load':
             tab_names[0] = "Project Info" # Rename first tab in Load mode
             
        tab1, tab2, tab3 = st.tabs(tab_names)

        with tab1:
            if st.session_state.app_mode == 'Upload':
                # Pass files to process
                render_text_tab(st.session_state.uploaded_files_list)
            elif st.session_state.app_mode == 'Load':
                # Display info, don't process files
                render_text_tab([]) # Pass empty list

        with tab2:
            if st.session_state.app_mode == 'Upload':
                 # Pass files to process
                 render_image_tab(st.session_state.uploaded_files_list)
            elif st.session_state.app_mode == 'Load':
                 # Display info, don't process files
                 render_image_tab([]) # Pass empty list

        with tab3:
            # Q&A tab relies on project_id in session state, which is set in both modes
            # It doesn't need the combined_text argument anymore if context comes from DB
            render_qa_tab() # Remove the argument as the function doesn't expect any
            
    else:
        if st.session_state.app_mode == 'Upload':
            st.info("Please upload one or more PDF files to begin analysis.")
        elif st.session_state.app_mode == 'Load':
             st.info("Please select a project to load from the sidebar.")

if __name__ == "__main__":
    main() 