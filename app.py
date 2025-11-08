import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

import streamlit as st

# --- Google Gemini SDK ---
try:
    import google.generativeai as genai
except ImportError:
    st.error("Please install the Google Generative AI SDK:\n\n`pip install google-generativeai`")
    st.stop()


# ---------------------------------
# Helper: get API key
# ---------------------------------
def get_api_key() -> str:
    load_dotenv()
    return os.environ.get("GOOGLE_API_KEY", "")


# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(page_title="Gemini Chatbot (Streamlit)", page_icon="💬", layout="wide")
st.title("💬 Gemini Chatbot with Memory")
st.caption("Streamlit + Google Gemini • retains conversation context & history")


# ---------------------------------
# Sidebar (set after model discovery)
# ---------------------------------
DEFAULT_SYSTEM_CONTEXT = (
    "You are a helpful, concise assistant. "
    "Use the chat history and the provided context to keep responses consistent. "
    "If the user asks for code, provide complete, runnable snippets."
)

# ---------------------------------
# Init SDK
# ---------------------------------
api_key = get_api_key()
if not api_key:
    st.error("Missing Google API key. Add **GOOGLE_API_KEY** to Streamlit secrets or environment.")
    st.stop()

genai.configure(api_key=api_key)


# ---------------------------------
# Fetch supported models (once)
# ---------------------------------
@st.cache_resource(show_spinner=False)
def get_supported_models() -> List[str]:
    """
    Return model names that support `generateContent`.
    We check `supported_generation_methods` defensively.
    """
    try:
        models = genai.list_models()
    except Exception as e:
        st.error(f"Unable to list models: {e}")
        return []

    valid = []
    for m in models:
        methods = getattr(m, "supported_generation_methods", []) or []
        # Some SDK versions may expose a dict-like; normalize to list[str]
        if isinstance(methods, dict):
            methods = list(methods.keys())
        if "generateContent" in methods:
            valid.append(m.name)
    return sorted(valid)


available_models = get_supported_models()
if not available_models:
    st.error(
        "No available models support `generateContent` with your API key.\n"
        "Verify your key/quotas or try a different project."
    )
    st.stop()


# ---------------------------------
# Sidebar controls (now that we know models)
# ---------------------------------
with st.sidebar:
    st.subheader("⚙️ Settings")

    model_name = st.selectbox(
        "Model",
        options=available_models,
        index=0,
        help="Only models available to your API key and supporting `generateContent` are shown."
    )

    memory_window = st.slider(
        "Messages to keep in memory (per side)",
        min_value=4, max_value=50, value=20, step=2,
        help="How many latest user/assistant messages to include in each prompt."
    )

    system_context = st.text_area(
        "Persistent context / system prompt",
        value=DEFAULT_SYSTEM_CONTEXT,
        height=150,
        help="Sent with every request to keep responses on-topic and consistent."
    )

    if st.button("🧹 Clear conversation", type="secondary"):
        # Clear only chat-related state so model dropdown etc. remain
        st.session_state.pop("messages", None)
        st.session_state.pop("last_model_name", None)
        st.rerun()

   


# ---------------------------------
# Instantiate model (cached)
# ---------------------------------
@st.cache_resource(show_spinner=False)
def get_model(name: str):
    return genai.GenerativeModel(name)

model = get_model(model_name)


# ---------------------------------
# Session state for messages
# ---------------------------------
if "messages" not in st.session_state:
    # Store history as [{"role": "user"|"model", "content": str}, ...]
    st.session_state.messages: List[Dict[str, Any]] = []

if "last_model_name" not in st.session_state:
    st.session_state.last_model_name = model_name

# Note a mid-chat model switch
if st.session_state.last_model_name != model_name:
    st.session_state.messages.append({
        "role": "model",
        "content": f"_Switched model to **{model_name}**. I'll continue with the same context._"
    })
    st.session_state.last_model_name = model_name


# ---------------------------------
# Build payload for Gemini
# ---------------------------------
def build_history_payload(
    history: List[Dict[str, str]],
    sys_context: str,
    window: int
) -> List[Dict[str, Any]]:
    """
    Gemini expects a list of messages with roles 'user' or 'model'.
    We inject a '[SYSTEM CONTEXT]' as the very first 'user' message.
    Then include the last `window` turns per side (approx via 2*window).
    """
    trimmed = history[-(window * 2):] if window > 0 else history

    payload: List[Dict[str, Any]] = []
    if sys_context.strip():
        payload.append({
            "role": "user",
            "parts": [f"[SYSTEM CONTEXT]\n{sys_context.strip()}"]
        })

    for msg in trimmed:
        role = msg.get("role", "user")  # "user" or "model"
        text = msg.get("content", "")
        payload.append({"role": role, "parts": [text]})
    return payload


# ---------------------------------
# Render chat history
# ---------------------------------
for msg in st.session_state.messages:
    # Streamlit chat UI expects 'assistant' or 'user'
    ui_role = "assistant" if msg["role"] == "model" else "user"
    with st.chat_message(ui_role):
        st.markdown(msg["content"])


# ---------------------------------
# Generate reply
# ---------------------------------
def generate_reply(user_text: str) -> str:
    payload = build_history_payload(
        st.session_state.messages + [{"role": "user", "content": user_text}],
        sys_context=system_context,
        window=memory_window,
    )
    try:
        response = model.generate_content(payload)
        # Prefer response.text; fall back defensively
        text = getattr(response, "text", None)
        if text:
            return text
        return "(No response text returned.)"
    except Exception as e:
        return f"⚠️ Error from model: {e}"


# ---------------------------------
# Chat input (THIS is the message box)
# ---------------------------------
user_input = st.chat_input("Type your message…")  # <-- the input box you type into

if user_input:
    # Show and store the user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")

        reply = generate_reply(user_input)

        # Optional typing effect
        shown = ""
        for ch in reply:
            shown += ch
            if len(shown) % 5 == 0:
                placeholder.markdown(shown)
            time.sleep(0.002)
        placeholder.markdown(shown)

    # Persist assistant reply (Gemini expects role 'model')
    st.session_state.messages.append({"role": "model", "content": reply})
