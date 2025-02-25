import streamlit as st
from transformers import pipeline
import spacy
import re

@st.cache_resource
def load_models():
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    nlp = spacy.load("en_core_web_sm")
    return qa_pipeline, summarizer, nlp

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

def safe_summarize(summarizer, text, max_length=130, min_length=30):
    try:
        # Clean and validate the text
        cleaned_text = clean_text(text)
        
        # Debug logging using expander
        with st.expander("Debug Information", expanded=False):
            st.write("Text length before cleaning:", len(text))
            st.write("Text length after cleaning:", len(cleaned_text))
        
        # Check if text is too short
        if len(cleaned_text.split()) < min_length:
            st.info("Text is too short to summarize")
            return None
        
        # Split text into chunks if it's too long
        words = cleaned_text.split()
        if len(words) > 500:  # If text is longer than 500 words
            st.info("Text is long, processing in chunks...")
            chunks = chunk_text(cleaned_text)
            summaries = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                try:
                    chunk_summary = summarizer(chunk,
                                            max_length=max_length,
                                            min_length=min_length,
                                            do_sample=False,
                                            truncation=True)
                    if chunk_summary and len(chunk_summary) > 0:
                        summaries.append(chunk_summary[0]['summary_text'])
                except Exception as e:
                    st.warning(f"Skipping chunk {i+1} due to error: {str(e)}")
                    continue
            
            # Combine summaries if we got any
            if summaries:
                # Join the summaries and create a final summary
                combined_summary = " ".join(summaries)
                if len(combined_summary.split()) > max_length:
                    # Create a final summary of the summaries
                    final_summary = summarizer(combined_summary,
                                            max_length=max_length,
                                            min_length=min_length,
                                            do_sample=False,
                                            truncation=True)
                    return final_summary[0]['summary_text']
                return combined_summary
            return None
        
        # For shorter texts, process normally
        summary = summarizer(cleaned_text,
                           max_length=max_length,
                           min_length=min_length,
                           do_sample=False,
                           truncation=True)
        
        if summary and len(summary) > 0:
            return summary[0]['summary_text']
        else:
            st.error("Summarizer returned empty result")
            return None
            
    except Exception as e:
        st.error(f"Error in safe_summarize: {str(e)}")
        return None

def render_qa_tab(extracted_text):
    qa_pipeline, summarizer, nlp = load_models()
    
    st.subheader("Document Summary")
    if not extracted_text or len(extracted_text.strip()) < 100:
        st.info("Not enough text for summarization (minimum 100 characters needed)")
    else:
        with st.spinner("Generating summary..."):
            summary = safe_summarize(summarizer, extracted_text)
            if summary:
                st.success("Summary generated successfully")
                st.write(summary)
            else:
                st.info("Could not generate summary for this document.")
    
    st.subheader("Ask Questions About the Document")
    if not extracted_text or len(extracted_text.strip()) < 10:
        st.warning("Please upload a document with text content first")
        return
        
    question = st.text_input("Enter your question:")
    
    if question:
        try:
            with st.spinner("Finding answer..."):
                answer = qa_pipeline(question=question, context=extracted_text)
                st.write("Answer:", answer["answer"])
                st.write("Confidence:", f"{answer['score']:.2%}")
                
                # Use spaCy for additional analysis
                doc = nlp(extracted_text)
                
                # Show named entities
                st.subheader("Named Entities in Context")
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                if entities:
                    st.write(entities)
                else:
                    st.info("No named entities found in the context")
        except Exception as e:
            st.error(f"Error processing question: {str(e)}") 