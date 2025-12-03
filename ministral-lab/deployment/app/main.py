import streamlit as st
from openai import OpenAI
import os
import json
import base64
import uuid
from datetime import datetime
from io import BytesIO
from pypdf import PdfReader
from docx import Document

# --- Configuration ---
VLLM_URL = os.getenv("VLLM_API_URL", "http://backend:8000/v1")
MODEL_NAME = "/model"
LOG_FILE = "/app/logs/feedback.jsonl"
SESSIONS_DIR = "/app/sessions"

# Ensure directories exist
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

st.set_page_config(page_title="Climate Expert AI", layout="wide", page_icon="🌍")

# --- Helper Functions ---

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def save_session():
    """Saves the current chat history to a JSON file."""
    session_path = os.path.join(SESSIONS_DIR, f"{st.session_state.session_id}.json")
    with open(session_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages
        }, f)

def load_session(filename):
    """Loads a chat history from disk."""
    with open(os.path.join(SESSIONS_DIR, filename), "r") as f:
        data = json.load(f)
        st.session_state.messages = data["messages"]
        st.session_state.session_id = filename.replace(".json", "")

def encode_image(uploaded_file):
    """Encodes image to Base64 for the model."""
    bytes_data = uploaded_file.getvalue()
    return base64.b64encode(bytes_data).decode('utf-8')

def extract_text_from_file(uploaded_file):
    """Extracts text from PDF or DOCX."""
    text = ""
    try:
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif uploaded_file.name.endswith(".docx"):
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            # Fallback for txt/md
            text = uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text

def update_feedback(message_index, score, text_correction=None):
    """Updates the feedback log. Handles re-clicking."""
    entry = {
        "session_id": st.session_state.session_id,
        "message_index": message_index,
        "timestamp": datetime.now().isoformat(),
        "user_prompt": st.session_state.messages[message_index-1]["content"],
        "model_response": st.session_state.messages[message_index]["content"],
        "score": score, # 1 for thumbs up, 0 for thumbs down
        "correction": text_correction
    }
    
    # In a real production app, you'd use a DB. For V1 JSONL, we just append.
    # To "update", we can just append a new entry with the same session_id/message_index 
    # and use the latest one during training data processing.
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    st.toast("Feedback recorded!")

# --- UI Layout ---

# Sidebar: History & Settings
with st.sidebar:
    st.title("🗄️ History")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Load past sessions
    files = sorted(os.listdir(SESSIONS_DIR), reverse=True)
    for f in files:
        if f.endswith(".json"):
            # Simple label: "Chat 2024-10..."
            if st.button(f"📄 {f[:15]}...", key=f):
                load_session(f)
                st.rerun()

# Main Chat
st.title("🌍 Climate Expert AI")
st.caption("Analyzing Policy, Laws, and IPCC Visuals.")

# Initialize Chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    get_session_id()

# Display Messages
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Handle Multi-modal content display
        if isinstance(msg["content"], list):
            # It's a complex message (Image + Text)
            for item in msg["content"]:
                if item["type"] == "text":
                    st.markdown(item["text"])
                elif item["type"] == "image_url":
                    st.image(item["image_url"]["url"], width=300)
        else:
            st.markdown(msg["content"])
    
    # Feedback Widget (Only for Assistant messages)
    if msg["role"] == "assistant":
        # Streamlit 1.37+ feedback widget
        score = st.feedback("thumbs", key=f"feedback_{idx}")
        if score is not None:
            update_feedback(idx, score)

# --- Input Area ---
with st.container():
    # File Uploader
    uploaded_file = st.file_uploader("Attach Context (PDF, DOCX, IMG)", type=["pdf", "docx", "png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if prompt := st.chat_input("Ask about climate policy..."):
        # 1. Prepare User Message
        user_content = []
        display_content = [] # What we show in UI vs what we send to model might differ slightly in structure
        
        # Handle File Attachment
        if uploaded_file:
            if uploaded_file.type.startswith("image"):
                # Image Logic
                base64_image = encode_image(uploaded_file)
                image_url = f"data:{uploaded_file.type};base64,{base64_image}"
                user_content.append({"type": "text", "text": prompt})
                user_content.append({"type": "image_url", "image_url": {"url": image_url}})
                # For display
                st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]})
            else:
                # Document Logic (Context Stuffing)
                doc_text = extract_text_from_file(uploaded_file)
                full_prompt = f"CONTEXT FROM DOCUMENT ({uploaded_file.name}):\n{doc_text}\n\nUSER QUESTION:\n{prompt}"
                user_content = full_prompt # Send as simple string
                st.session_state.messages.append({"role": "user", "content": full_prompt})
        else:
            # Text Only
            user_content = prompt
            st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Render User Message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
            if uploaded_file and uploaded_file.type.startswith("image"):
                 st.image(uploaded_file)

        # 3. Generate Response
        with st.chat_message("assistant"):
            text_placeholder = st.empty()
            full_response = ""
            client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")
            
            # Construct messages for API
            api_messages = [
                {"role": "system", "content": "You are a Climate Expert. You analyze climate laws, IPCC reports, and scientific data with precision."},
            ]
            # Add history (simple version: last 5 turns to save context)
            for m in st.session_state.messages[-5:]:
                # Ensure we format history correctly for the API
                api_messages.append({"role": m["role"], "content": m["content"]})
            
            try:
                stream = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=api_messages,
                    temperature=0.2, # Lower temp for "Expert" accuracy
                    stream=True,
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        text_placeholder.markdown(full_response + "▌")
                
                text_placeholder.markdown(full_response)
                
            except Exception as e:
                st.error(f"Model Error: {e}")
                full_response = "Error generating response."

        # 4. Save Turn
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        save_session()
        st.rerun()