import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="Gemini 2.0 Flash Thinking Chat (Streaming)",
    page_icon="🧠💬",
    layout="wide",
)

st.title("🧠 Gemini 2.0 Flash Thinking Chat")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Please create a .env file with `GEMINI_API_KEY=YOUR_KEY`.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Layout using columns ---
chat_col, thought_col = st.columns([4, 1])

with chat_col:
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Set up the thought process area ahead of time
with thought_col:
    st.subheader("Thinking Process:")
    thinking_placeholder = st.empty()

# Chat input
prompt = st.chat_input("What's on your mind?")
if prompt:
    # Add user's message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_col:
        with st.chat_message("user"):
            st.markdown(prompt)

    # Initialize placeholders for the assistant's response
    with chat_col:
        assistant_placeholder = st.chat_message("assistant")
        assistant_response_container = assistant_placeholder.container()
    full_thinking_process = ""
    full_response_text = ""
    processing_thinking = True

    try:
        response_stream = model.generate_content(prompt, stream=True)
        for chunk in response_stream:
            if chunk.parts:
                # If we have multiple parts, that means we've hit the final response section
                # The first part of that chunk is the remainder of the thinking, 
                # and subsequent parts (if any) are the start of the final response.
                if len(chunk.parts) > 1:
                    # Finish off the thinking process if still processing it
                    if processing_thinking:
                        full_thinking_process += chunk.parts[0].text
                        thinking_placeholder.markdown(full_thinking_process)
                        processing_thinking = False
                        # Append any remaining parts from this chunk to the final response
                        for part in chunk.parts[1:]:
                            if part.text:
                                full_response_text += part.text
                                assistant_response_container.markdown(full_response_text)
                    else:
                        # Already in final response mode
                        for part in chunk.parts:
                            if part.text:
                                full_response_text += part.text
                                assistant_response_container.markdown(full_response_text)
                else:
                    # Single part chunk
                    if processing_thinking:
                        # Still processing thinking mode
                        full_thinking_process += chunk.parts[0].text
                        thinking_placeholder.markdown(full_thinking_process)
                    else:
                        # Already in final response mode
                        full_response_text += chunk.parts[0].text
                        assistant_response_container.markdown(full_response_text)

    except Exception as e:
        full_thinking_process = f"Error fetching thinking process: {e}"
        full_response_text = f"Error generating response: {e}"
        thinking_placeholder.markdown(full_thinking_process)
        assistant_response_container.markdown(full_response_text)

    # Add the assistant's final response to session state and show it
    st.session_state.messages.append({"role": "assistant", "content": full_response_text})

# If there are no messages yet, display a hint
if not st.session_state.messages and api_key:
    with thought_col:
        thinking_placeholder.info("Start a conversation to see the model's thinking process here.")
elif not api_key:
    with thought_col:
        st.warning("API Key is missing. Cannot access the thinking model.")
