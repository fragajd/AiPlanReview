import streamlit as st
from transformers import pipeline
import re
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
import uuid
import random

# Import shared resources
from .shared_resources import (
    get_supabase_client,
    get_sentence_transformer_model,
    get_initialization_errors,
    get_ollama_client,      # Import ollama client
    get_ollama_model_name   # Import ollama model name
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get resources from shared module
supabase = get_supabase_client()
model = get_sentence_transformer_model() # Sentence transformer for embeddings
ollama_client = get_ollama_client()      # Ollama client for generation
ollama_model_name = get_ollama_model_name() # Gemma model name
initialization_errors = get_initialization_errors()

def clean_text(text):
    try:
        # Handle None or empty text
        if not text or not isinstance(text, str):
            return ""
            
        # Remove special characters but keep more punctuation for context
        text = re.sub(r'[^\w\s.,!?;:()-]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove very long words (likely OCR errors)
        text = ' '.join(word for word in text.split() if len(word) < 45)
        
        # Ensure minimum length
        if len(text.strip()) < 10:  # arbitrary minimum length
            return ""
            
        return text
    except Exception as e:
        st.error(f"Error in clean_text: {str(e)}")
        return ""

@st.cache_data(ttl=3600, show_spinner=False)
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        # If this is not the last chunk, try to find a good breaking point
        if end < len(words):
            # Look for the last period in the overlap region
            overlap_start = max(end - overlap, start)
            overlap_text = ' '.join(words[overlap_start:end])
            last_period = overlap_text.rfind('.')
            
            if last_period != -1:
                # Found a period in the overlap, adjust end to break at this sentence
                end = overlap_start + last_period + 1
        
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

# Cache context retrieval results to avoid redundant database queries
@st.cache_data(ttl=1800, show_spinner=False)
def _cached_find_relevant_chunks(
    question: str, 
    project_id: Optional[int], 
    threshold: float, 
    match_count: int,
    include_building_code: bool = True,
    code_ratio: float = 0.4  # % of results to come from building code
) -> Dict[str, List[Dict[str, Any]]]:
    """Cached version of finding relevant chunks from Supabase.
       Searches the 'documents' table (containing summary chunks) and optionally 'florida_building_code'.
    Args:
        question: User's question
        project_id: Project ID to filter results
        threshold: Similarity threshold
        match_count: Maximum number of matches to return from documents table
        include_building_code: Whether to include Florida Building Code sections
        code_ratio: Percentage of total results to aim for from building code (0.0-1.0)
    
    Returns:
        Dictionary with 'summary_chunks' and 'code_chunks' lists
    """
    if not supabase or not model:
        logger.warning("Supabase client or sentence transformer model not available for search.")
        return {"summary_chunks": [], "code_chunks": []}
    if not question or not project_id:
        logger.warning("_cached_find_relevant_chunks called without question or project_id.")
        return {"summary_chunks": [], "code_chunks": []}

    try:
        logger.info(f"Generating embedding for question: '{question[:50]}...'" )
        question_embedding = model.encode(question).tolist() # Convert to list for Supabase

        results = {"summary_chunks": [], "code_chunks": []}
        
        # --- Search Document Summary Chunks (in 'documents' table) ---
        # Determine number of document matches needed
        doc_match_count = match_count
        if include_building_code:
            # Adjust doc count if code is included to maintain total near match_count
            doc_match_count = int(match_count * (1.0 - code_ratio))
            doc_match_count = max(1, doc_match_count) # Ensure at least 1 doc match
            
        logger.info(f"Searching for {doc_match_count} relevant document summary chunks...")
        doc_response = supabase.rpc('match_documents', { # Assuming this RPC queries 'documents' table
            'query_embedding': question_embedding,
            'match_threshold': threshold,
            'match_count': doc_match_count,
            'filter_project_id': project_id
        }).execute()
        
        if hasattr(doc_response, 'data') and doc_response.data:
            # IMPORTANT: Ensure the RPC returns 'file_name' along with 'content'
            results["summary_chunks"] = doc_response.data 
            logger.info(f"Found {len(doc_response.data)} relevant document summary chunks.")
        elif hasattr(doc_response, 'error') and doc_response.error:
            logger.error(f"Supabase RPC error for match_documents: {doc_response.error}")
        
        # --- Search Florida Building Code --- 
        if include_building_code:
            code_match_count = int(match_count * code_ratio)
            code_match_count = max(1, code_match_count) # Ensure at least 1 code match
            
            logger.info(f"Searching for {code_match_count} relevant Florida Building Code chunks...")
            code_response = supabase.rpc('match_florida_building_code', {
                'query_embedding': question_embedding,
                'match_threshold': threshold,
                'match_count': code_match_count
            }).execute()
            
            if hasattr(code_response, 'data') and code_response.data:
                results["code_chunks"] = code_response.data
                logger.info(f"Found {len(code_response.data)} relevant Florida Building Code chunks.")
            elif hasattr(code_response, 'error') and code_response.error:
                logger.error(f"Supabase RPC error for building code: {code_response.error}")
        
        return results

    except Exception as e:
        logger.error(f"Error finding relevant chunks: {e}", exc_info=True)
        return {"summary_chunks": [], "code_chunks": []}

def find_relevant_chunks(
    query: str,
    top_k: int = 5, # Increased default slightly
    use_code: bool = True,
    min_score_threshold: float = 0.20  # Minimum relevance score
) -> Tuple[str, str]: # Returns (summary_context, code_context)
    """
    Find and format the most relevant content chunks based on the query.
    Retrieves chunks from document summaries (stored in 'documents' table) and optionally Florida Building Code.
    
    Args:
        query: User's question
        top_k: Total number of results to aim for across sources
        use_code: Whether to search in code
        min_score_threshold: Minimum similarity score threshold
        
    Returns:
        Tuple of (summary_context, code_context)
    """
    # Get the current project ID from session state
    project_id = st.session_state.get('current_project_id')
    if not project_id:
        logger.warning("No project ID in session state. Cannot find relevant chunks.")
        return "", ""
    
    # Use cached function to find relevant chunks
    # Removed use_summaries and use_documents as they are implicit now
    results = _cached_find_relevant_chunks(
        question=query,
        project_id=project_id,
        threshold=min_score_threshold,
        match_count=top_k, # Pass top_k as the target total match count
        include_building_code=use_code,
        code_ratio=0.4  # Keep code ratio
    )
    
    # Extract chunks from results
    summary_chunks = results.get("summary_chunks", []) # Renamed from document_chunks for clarity
    code_chunks = results.get("code_chunks", [])
    
    # Format summary chunk context
    summary_context = ""
    if summary_chunks:
        summary_context = "RELEVANT DOCUMENT SUMMARY CHUNKS:\n\n"
        for chunk_data in summary_chunks:
            # Ensure 'file_name' and 'content' are retrieved by the RPC
            file_name = chunk_data.get('file_name', 'Unknown Document') 
            content = chunk_data.get('content', '')
            similarity = chunk_data.get('similarity', None) # Optional: display similarity
            context_line = f"Source File: {file_name}\nContent: {content}"
            if similarity is not None:
                 context_line += f" (Similarity: {similarity:.3f})"
            summary_context += context_line + "\n\n"
    
    # Format code content
    code_context = ""
    if code_chunks:
        code_context = "FLORIDA BUILDING CODE SECTIONS:\n\n"
        for chunk_data in code_chunks:
            content = chunk_data.get('content', '')
            section = chunk_data.get('source_section', 'Unknown Section')
            similarity = chunk_data.get('similarity', None)
            context_line = f"Section: {section}\nContent: {content}"
            if similarity is not None:
                 context_line += f" (Similarity: {similarity:.3f})"
            code_context += context_line + "\n\n"
            
    return summary_context, code_context

# Cache summarization to avoid regenerating the same summary
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_summarize_with_gemma(text_to_summarize: str, max_length_approx_words: int = 150) -> str:
    """Cached version of Gemma summarization"""
    if not ollama_client:
        logger.warning("Ollama client not initialized. Skipping Gemma summarization.")
        return "Summarization unavailable: Ollama client not ready."
    if not text_to_summarize or len(text_to_summarize.split()) < 50:  # Min length check
        logger.info("Text too short to summarize effectively.")
        return "Text too short to summarize effectively."

    try:
        # Simple approach: Summarize entire text.
        # Consider adding chunking logic here if very long inputs become common.

        prompt = f"""Summarize the following text concisely, focusing on the main technical points found in architectural or engineering plans. Aim for a summary around {max_length_approx_words} words. Output only the summary itself, without any preamble like "Here is the summary:".

Text to summarize:
---
{text_to_summarize}
---

Summary:"""

        logger.info(f"Sending text (length: {len(text_to_summarize)}) to Ollama model {ollama_model_name} for summarization...")
        response = ollama_client.chat(
            model="gemma3:4b",  # Use literal string for consistency
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        summary = response['message']['content'].strip()
        # Post-process: Remove potential unwanted preamble if model ignores instructions
        summary = re.sub(r"^(Here's a summary|The summary is|Okay, here's the summary):?\s*", "", summary, flags=re.IGNORECASE).strip()
        logger.info("Received summary from Ollama.")
        return summary

    except Exception as e:
        logger.error(f"Error calling Ollama for summarization: {e}", exc_info=True)
        return f"Error generating summary: {e}"

# --- New Gemma Summarization Function ---
def summarize_with_gemma(text_to_summarize: str, max_length_approx_words=150) -> str:
    """Uses Gemma via Ollama to summarize the provided text. Now uses caching."""
    if not ollama_client:
        st.warning("Ollama client not available. Skipping summarization.")
        return "Summarization unavailable: Ollama client not ready."
        
    # Use the cached version
    return _cached_summarize_with_gemma(
        text_to_summarize=text_to_summarize,
        max_length_approx_words=max_length_approx_words
    )

# Generate a hash for a question to help track similar questions
def _get_question_hash(question: str) -> str:
    """Generate a hash for the question for caching purposes"""
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

# Cache Gemma/Gemini responses to avoid regenerating answers for similar questions
@st.cache_data(ttl=1800, show_spinner=False)
def _cached_get_ai_response(
    question: str, 
    context: str, 
    history_fingerprint: str,
    has_document_matches: bool, # Note: This now represents if summary chunks were found
    has_code_matches: bool,
    check_compliance: bool = False,
    use_gemini: bool = False
) -> str:
    """Cached version of AI response generation.
    Works with both Ollama (Gemma) and Google API (Gemini) models.
    Assumes context contains document summary chunks and/or code chunks.
    """
    if use_gemini:
        # Check if we can use Gemini API
        google_api_key = st.session_state.get('google_api_key', None)
        if not google_api_key:
            logger.error("Gemini API selected but no API key provided.")
            return "Error: Google API Key needed for Gemini. Please provide it in the sidebar."
    else:
        # Check if we can use Ollama/Gemma
        if not ollama_client:
            logger.error("_cached_get_ai_response called but Ollama client is not available.")
            return "Sorry, I cannot process your request right now. The required language model connection is unavailable."

    # --- Construct the system prompt --- 
    if check_compliance:
        # Compliance-focused prompt (updated context description)
        system_prompt = (
            "You are a specialized architectural and engineering compliance assistant with expertise in the Florida Building Code. "
            "Your primary goal is to analyze construction documents and identify potential compliance issues or errors. "
            "Use the provided CONTEXT (containing relevant chunks from document summaries and potentially Florida Building Code sections) "
            "to answer the user's QUESTION, with special focus on identifying any compliance problems. "
            "\n\n"
            "When responding, follow these rules:\n"
            "1. Clearly identify any potential code compliance issues based on the provided summary chunks and code sections.\n"
            "2. Reference specific sections of the Florida Building Code if relevant context is provided.\n"
            "3. Suggest possible remediation approaches for any issues found.\n"
            "4. Be explicit about what aspects comply and what potentially doesn't comply according to the context.\n"
            "5. When uncertain, indicate the limitations of your analysis (e.g., 'Based on the summary provided...').\n"
            "\n"
            "Your analysis should be thorough but concise. Be direct and professional in your assessment."
        )
    else:
        # Standard assistance prompt (updated context description)
        system_prompt = (
            "You are a helpful AI assistant specialized in analyzing technical documents like architectural or engineering plans. "
            "Use the provided CONTEXT (extracted from relevant document summary chunks and potentially Florida Building Code sections) "
            "to answer the user's QUESTION. Give accurate and concise answers based *only* on the context provided. "
            "If the context does not contain the answer, state that clearly."
        )
    
    # Create different message templates depending on what context we have
    context_prompt_part = ""
    no_context_message = "(No specific context was found in the document summaries or building code for this question.)"
    
    if context:
        context_prompt_part = f"\n\nRelevant CONTEXT:\n{context}"
    else:
        context_prompt_part = f"\n\nCONTEXT: {no_context_message}"
    
    # Add information about what type of context was found
    context_sources = []
    if has_document_matches: # This flag now indicates summary chunks were found
        context_sources.append("document summary chunks")
    if has_code_matches:
        context_sources.append("Florida Building Code")
    
    if context_sources:
        source_info = f"\n\nNote: This answer draws from: {', '.join(context_sources)}"
    else:
        source_info = ""
        
    # --- Get response from selected model ---
    try:
        if use_gemini:
            # --- Use Google Gemini API ---
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            
            google_api_key = st.session_state.get('google_api_key', None)
            genai.configure(api_key=google_api_key)
            
            # Use a suitable Gemini model 
            gemini_model_name = 'gemini-2.5-flash-preview-05-20'
            model = genai.GenerativeModel(gemini_model_name)
            
            logger.info(f"Sending request to Google Gemini API model {gemini_model_name} for Q&A.")
            
            # Format the message for Gemini
            gemini_prompt = f"{system_prompt}\n\nQUESTION: {question}{context_prompt_part}"
            
            response = model.generate_content(
                gemini_prompt,
                stream=False,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            if response.text:
                answer = response.text.strip()
                logger.info(f"Received response from Gemini model {gemini_model_name}.")
                return answer + source_info
            else:
                logger.error(f"Gemini API returned empty response. Feedback: {response.prompt_feedback}")
                return f"Error: Failed to generate a response. Feedback: {response.prompt_feedback}"
                
        else:
            # --- Use local Ollama/Gemma ---
            logger.info(f"Sending request to Ollama model {ollama_model_name} for Q&A.")
        
            # Prepare messages for Ollama chat endpoint
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"QUESTION: {question}{context_prompt_part}"}
            ]
    
            # Use stream=False for a single response object
            response = ollama_client.chat(
                model="gemma3:4b",  # Use literal string for consistency
                messages=messages,
                stream=False 
            )
        
            assistant_response = response['message']['content'].strip()
            logger.info(f"Received response from Ollama model {ollama_model_name}.")
            return assistant_response + source_info
        
    except Exception as e:
        logger.error(f"Error calling model for Q&A: {e}", exc_info=True)
        return f"Sorry, I encountered an error while trying to generate a response. ({e})"

# --- Unified Q&A Function with model selection ---
def get_ai_response(
    question: str, 
    chat_history: list, 
    context: str,
    has_document_matches: bool,
    has_code_matches: bool,
    check_compliance: bool = False
) -> str:
    """
    Generates a response using Gemma via Ollama or Gemini via API, with caching.
       Includes chat history in responses but uses caching for efficiency.
    """
    # Determine which model to use based on user preference
    model_preference = st.session_state.get('model_preference', 'local') # Default to local
    use_gemini = (model_preference == 'api')
    
    # Check if we can use the selected model
    if use_gemini and not st.session_state.get('google_api_key'):
        return "Error: Google API Key needed for Gemini. Please provide it in the sidebar."
    if not use_gemini and not ollama_client:
        return "Sorry, I cannot process your request right now. The Ollama client is unavailable."

    # Only use last 5 messages of history for context
    recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
    
    # Generate a fingerprint of the history to use in cache key
    history_items = []
    for msg in recent_history:
        # Only include role and first 100 chars of content to keep fingerprint reasonable size
        history_items.append(f"{msg['role']}:{msg['content'][:100]}")
    history_fingerprint = hashlib.md5("|".join(history_items).encode()).hexdigest()
    
    # Get cached response with appropriate model
    cached_response = _cached_get_ai_response(
        question=question,
        context=context,
        history_fingerprint=history_fingerprint,
        has_document_matches=has_document_matches,
        has_code_matches=has_code_matches,
        check_compliance=check_compliance,
        use_gemini=use_gemini
    )
    
    # If history is short or empty, use the cached response directly
    if len(chat_history) <= 2:  # Just the current question or first exchange
        return cached_response
        
    # For longer conversations, add context about chat history
    historical_context = ""
    
    # Format history as a more concise context
    if len(recent_history) > 2:
        # Skip the current question which should be the last message
        condensed_history = recent_history[:-1]
        # Format history as a simple dialogue
        formattted_history = []
        for msg in condensed_history:
            formattted_history.append(f"{msg['role'].capitalize()}: {msg['content']}")
        historical_context = "\n\nThis question is part of an ongoing conversation:\n" + "\n".join(formattted_history)
    
    # If we have historical context, add a note about it to the response
    if historical_context:
        # Check if we need to add reference to chat history
        if "previous message" not in cached_response.lower() and "earlier question" not in cached_response.lower():
            # Add a simple note if the response doesn't already reference history
            final_response = f"{cached_response}\n\n(Note: I've considered our conversation history in this response.)"
            return final_response
            
    return cached_response

# --- Function to check if embeddings are available ---
def check_embeddings_status() -> bool:
    """
    Check if document embeddings are available for the current project.
    
    Returns:
        bool: True if embeddings are available, False otherwise
    """
    if not supabase:
        logger.warning("Supabase client not available. Cannot check embeddings status.")
        return False
        
    # Get current project ID from session state
    project_id = st.session_state.get('current_project_id')
    if not project_id:
        logger.warning("No project ID in session state. Cannot check embeddings status.")
        return False
        
    try:
        # Query for at least one document chunk for the current project
        response = supabase.table('documents') \
            .select('id') \
            .eq('project_id', project_id) \
            .limit(1) \
            .execute()
            
        # If we get at least one result, embeddings are available
        if hasattr(response, 'data') and len(response.data) > 0:
            logger.info(f"Found document embeddings for project ID: {project_id}")
            return True
        else:
            logger.warning(f"No document embeddings found for project ID: {project_id}")
            return False
    except Exception as e:
        logger.error(f"Error checking embeddings status: {e}", exc_info=True)
        return False

def render_qa_tab():
    """Renders the Q&A tab in the Streamlit interface for document analysis."""
    st.header("Document Q&A")

    # Sidebar for model and settings
    with st.sidebar:
        st.subheader("Q&A Settings")
        use_gemini = st.checkbox("Use Google Gemini API", value=False, 
                                help="Use Google's Gemini API instead of local Gemma model")
        
        check_compliance = st.checkbox("Check for Code Compliance", value=False,
                                    help="Focus on identifying potential Florida Building Code compliance issues")
                                    
        # Only show API key input if Gemini is selected
        if use_gemini:
            google_api_key = st.text_input("Google API Key", 
                                        value=st.session_state.get('google_api_key', ''),
                                        type="password",
                                        help="Required for using Google Gemini API")
            st.session_state['google_api_key'] = google_api_key
    
    # Get embeddings status
    embeddings_available = check_embeddings_status()
    
    if not embeddings_available:
        st.warning("No document embeddings found. Please upload and process documents in the Document Management tab first.")
        return
    
    # Display information about document summaries
    st.info("The Q&A system is optimized to prioritize document summaries, providing more accurate and comprehensive answers about your documents.")
    
    # Create a unique key for each session to track question history
    if 'qa_session_id' not in st.session_state:
        st.session_state['qa_session_id'] = str(uuid.uuid4())
    
    if 'qa_history' not in st.session_state:
        st.session_state['qa_history'] = []
    
    # Input for user's question
    user_question = st.text_input("Ask a question about the documents:")
    
    if st.button("Submit Question", key="submit_question") and user_question:
        with st.spinner("Finding relevant information and generating response..."):
            # Find relevant chunks (updated call)
            summary_context, code_context = find_relevant_chunks(user_question, use_code=check_compliance)
            
            # Combine contexts
            combined_context = summary_context + code_context # Simpler combination
            has_document_matches = bool(summary_context)
            has_code_matches = bool(code_context)
            
            # Generate a fingerprint of the question history to use for caching
            history_fingerprint = hashlib.md5(
                str(st.session_state['qa_history'][-5:] if st.session_state['qa_history'] else "").encode()
            ).hexdigest()
            
            # Get AI response (updated call)
            response = _cached_get_ai_response(
                user_question, 
                combined_context, 
                history_fingerprint,
                has_document_matches=has_document_matches,
                has_code_matches=has_code_matches,
                check_compliance=check_compliance,
                use_gemini=use_gemini
            )
            
            # Store the Q&A pair in history
            st.session_state['qa_history'].append({"question": user_question, "answer": response})
    
    # Display conversation history in reverse (newest at the top)
    if st.session_state['qa_history']:
        for i, qa_pair in enumerate(reversed(st.session_state['qa_history'])):
            with st.expander(f"Q: {qa_pair['question']}", expanded=(i == 0)):
                st.markdown(qa_pair['answer'])
                
                # Add option to regenerate the answer
                if st.button("Regenerate Answer", key=f"regenerate_{i}"):
                    with st.spinner("Regenerating response..."):
                        # Remove the last answer to force regeneration
                        question_to_regenerate = qa_pair['question']
                        
                        # Find the index of this QA pair in the history
                        idx = len(st.session_state['qa_history']) - 1 - i
                        
                        # Update the session state to remove this answer
                        st.session_state['qa_history'].pop(idx)
                        
                        # Rerun with the same question
                        st.session_state['regenerate_question'] = question_to_regenerate
                        st.experimental_rerun()
    
    # Check if we have a question to regenerate from a previous button click
    if 'regenerate_question' in st.session_state and st.session_state['regenerate_question']:
        user_question = st.session_state['regenerate_question']
        del st.session_state['regenerate_question']
        
        with st.spinner("Finding relevant information and regenerating response..."):
            # Find relevant chunks (updated call)
            summary_context, code_context = find_relevant_chunks(user_question, use_code=check_compliance)
            
            # Combine contexts
            combined_context = summary_context + code_context # Simpler combination
            has_document_matches = bool(summary_context)
            has_code_matches = bool(code_context)

            # Generate a fingerprint of the question history to use for caching
            history_fingerprint = hashlib.md5(
                str(st.session_state['qa_history'][-5:] if st.session_state['qa_history'] else "").encode()
            ).hexdigest()

            # Get AI response (updated call with randomness)
            response = _cached_get_ai_response(
                user_question, 
                combined_context, 
                history_fingerprint + str(random.randint(1, 10000)),  # Add randomness to force regeneration
                has_document_matches=has_document_matches,
                has_code_matches=has_code_matches,
                check_compliance=check_compliance,
                use_gemini=use_gemini
            )
            
            # Store the Q&A pair in history
            st.session_state['qa_history'].append({"question": user_question, "answer": response})
            
            # Rerun to show the updated history
            st.experimental_rerun()

# --- Placeholder for generate_response function ---
async def generate_response(query_text, context, bot_instructions=None, model="gpt-3.5-turbo"):
    """
    Placeholder for the generate_response function.
    This function would normally generate a response using an external API like OpenAI.
    For now, we'll use the cached_get_ai_response function instead.
    
    Args:
        query_text: User's question
        context: Context information for answering the question
        bot_instructions: Optional special instructions for the bot
        model: Model name to use
        
    Returns:
        Generated response text
    """
    logger.warning("Using placeholder generate_response function")
    
    # Since this is a placeholder, we'll create a fingerprint for caching
    fingerprint = hashlib.md5((query_text + context[:100]).encode()).hexdigest()
    
    # Determine if using Google API based on session state
    use_gemini = st.session_state.get('model_preference', 'local') == 'api'
    
    # Call our cached function to generate a response
    response = _cached_get_ai_response(
        question=query_text,
        context=context,
        history_fingerprint=fingerprint,
        has_document_matches=bool(context),
        has_code_matches="FLORIDA BUILDING CODE" in context,
        check_compliance=False,
        use_gemini=use_gemini
    )
    
    return response

async def analyze_query(
    query_text,
    project_id=None,
    use_code=True,
    bot_instructions=None,
    model="gpt-3.5-turbo",
):
    """
    Analyze a query using GPT with context from relevant document chunks.
    
    Args:
        query_text: The user's query
        project_id: Optional project ID to filter results
        use_code: Whether to include code snippets in context
        bot_instructions: Optional custom instructions for the bot
        model: The OpenAI model to use for generating the response
        
    Returns:
        Dict containing the query, response, and other metadata
    """
    logger.info(f"Analyzing query: {query_text}")
    start_time = time.time()
    
    if not query_text.strip():
        return {
            "query": query_text,
            "response": "I need a question to be able to help you.",
            "has_context": False,
            "processing_time": 0
        }
    
    # Get relevant chunks for the query (updated call)
    summary_context, code_context = find_relevant_chunks(
        query=query_text,
        top_k=5, # Use the default or adjust as needed
        use_code=use_code
    )
    
    # Combine contexts
    combined_context = summary_context + code_context # Simpler combination
    has_document_matches = bool(summary_context)
    has_code_matches = bool(code_context)
    
    # Flag if we have any context at all
    has_context = has_document_matches or has_code_matches
    
    # If no context found, use a generic response approach
    if not has_context:
        logger.info("No context found in documents. Using generic response.")
        # Note: generate_response placeholder internally calls _cached_get_ai_response
        response = await generate_response(
            query_text, 
            "", 
            bot_instructions, 
            model=model
        )
    else:
        # Generate a response with the retrieved context
        # Note: generate_response placeholder internally calls _cached_get_ai_response
        response = await generate_response(
            query_text, 
            combined_context, 
            bot_instructions, 
            model=model
        )
    
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    
    logger.info(f"Query processed in {processing_time} seconds")
    
    return {
        "query": query_text,
        "response": response,
        "has_context": has_context,
        "has_document_matches": has_document_matches, # Represents summary chunks found
        "has_code_matches": has_code_matches,
        "processing_time": processing_time
    }