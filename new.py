import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import re

# Title
st.title("🎨 AI-Powered Art Generator")

# Hugging Face API setup
API_TOKEN = "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"
TEXT2IMG_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
API_URL = f"https://api-inference.huggingface.co/models/{TEXT2IMG_MODEL}"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def sanitize_filename(prompt):
    """Convert prompt to safe filename"""
    # Remove special characters and limit length
    clean = re.sub(r'[^a-zA-Z0-9_]', '', prompt.replace(' ', '_'))[:40]
    return clean or "generated_art"

# Text-to-Image Section
st.subheader("📝 Generate Art from Text Prompt")
text_prompt = st.text_input(
    "Enter your prompt here", 
    value="sunset",
    key="text_input"
)

if st.button("Generate Image"):
    if text_prompt:
        with st.spinner("✨ Creating your masterpiece..."):
            try:
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json={
                        "inputs": text_prompt,
                        "parameters": {
                            "height": 768,
                            "width": 768,
                            "num_inference_steps": 50,
                            "guidance_scale": 7.5
                        }
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    image_bytes = response.content
                    image = Image.open(BytesIO(image_bytes))
                    
                    # Display image
                    st.image(image, 
                            caption="Generated Art", 
                            use_container_width=True)
                    
                    # Create download button
                    filename = f"{sanitize_filename(text_prompt)}.png"
                    st.download_button(
                        label="⬇️ Download Image",
                        data=image_bytes,
                        file_name=filename,
                        mime="image/png",
                        key="download_btn"
                    )
                    
                else:
                    error_msg = response.json().get("error", response.text)
                    st.error(f"Error {response.status_code}: {error_msg}")
                    if "loading" in error_msg.lower():
                        st.info("Model is loading... Please try again in 45 seconds!")
                    
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
    else:
        st.warning("Please enter a text prompt")

st.markdown("---")
st.info("ℹ️ First request may take 45-60 seconds while model loads. For best results, use descriptive prompts like 'vibrant sunset over ocean with dramatic clouds'")