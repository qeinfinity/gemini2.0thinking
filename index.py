import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="LLM Interface",
    page_icon="🤖",
)

st.title("Gemini 2.0")

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
else:
    st.error("Gemini API key not found. Please create a .env file with GEMINI_API_KEY=YOUR_KEY.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's on your mind?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if api_key:
        try:
            response_stream = model.generate_content(prompt, stream=True)
            full_response = ""  # Accumulate the response
            message_placeholder = st.empty() # Create an empty placeholder

            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌") # Display with a "typing" indicator

            message_placeholder.markdown(full_response) # Final full response

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_message = f"Error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.markdown(error_message)
    else:
        st.warning("Please enter your API key to get a response.")