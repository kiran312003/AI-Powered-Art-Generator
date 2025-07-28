import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import warnings
from io import BytesIO
import gc

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

# Streamlit page setup
st.set_page_config(page_title="Text to Image Generator", layout="centered")
st.title("🎨✨ Text-to-Image Generator")

# Device setting
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.success(f"Running on: {device.upper()}")
# Load model with proper error handling
@st.cache_resource
def load_model():
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        )
        
        pipe = pipe.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            pipe.enable_attention_slicing()
            # Optionally enable xformers for even more memory efficiency if available
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
        
        return pipe
    except Exception as e:
        return None

# Create sidebar controls for advanced options
st.sidebar.header("Advanced Options")
use_cached_model = st.sidebar.checkbox("Use cached model", value=True, 
                                     help="Uncheck to reload model (useful if encountering issues)")

# Show loading message
with st.spinner("Loading Stable Diffusion model (this may take a minute)..."):
    if use_cached_model:
        pipe = load_model()
    else:
        # Clear cache and reload model
        st.cache_resource.clear()
        pipe = load_model()

# Check if model loaded successfully
if pipe is None:
    st.error("Failed to initialize the model. Please try again.")
    st.stop()

# UI for user inputs
st.subheader("Create your image")
prompt = st.text_area("Enter your prompt:", 
                     placeholder="e.g., A colorful futuristic city at sunset with flying cars and neon lights 🌇",
                     help="Be detailed and descriptive for better results")

col1, col2 = st.columns(2)
with col1:
    num_inference_steps = st.slider("Quality (Steps)", 20, 100, 50, 
                                  help="Higher values = better quality but slower generation")
with col2:
    guidance_scale = st.slider("Creativity", 1.0, 20.0, 7.5, 
                             help="Lower values = more creative, higher values = more prompt-adherent")

# Add negative prompt option
negative_prompt = st.text_area("Negative prompt:", 
                              placeholder="What you DON'T want in the image (e.g., blurry, bad anatomy, low quality)",
                              help="Specify elements to avoid in the generated image")

# Add a seed option for reproducibility
use_random_seed = st.checkbox("Use random seed", value=True)
seed = None
if not use_random_seed:
    seed = st.number_input("Seed", value=42, help="Same seed produces similar images for the same prompt")

generate = st.button("Generate Image 🎨", use_container_width=True)

# Initialize session state for storing generation history
if "history" not in st.session_state:
    st.session_state.history = []

if generate and prompt:
    with st.spinner("Generating your masterpiece... Please wait ⏳"):
        try:
            # Set the seed if specified
            generator = None
            if not use_random_seed and seed is not None:
                generator = torch.Generator(device).manual_seed(seed)
            
            # Generate image with progress bar
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
            
            # Check if result contains images
            if not result.images or len(result.images) == 0:
                st.error("Failed to generate image. Please try again with a different prompt.")
            else:
                image = result.images[0]
                
                # Save current generation to history
                buf = BytesIO()
                image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                # Append to history with prompt info
                st.session_state.history.append({
                    "prompt": prompt,
                    "image": byte_im,
                    "negative_prompt": negative_prompt,
                    "steps": num_inference_steps,
                    "guidance": guidance_scale,
                    "seed": seed if not use_random_seed else "random"
                })
                
                # Display image
                st.image(image, caption=f"Generated Art: {prompt[:50]}...", use_container_width=True)
                
                # Download button
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="generated_image.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                # Clean up to prevent memory issues
                del result
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                st.error("GPU ran out of memory. Try reducing the quality steps or restart the application.")
            else:
                st.error(f"Error during generation: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# Show generation history
if st.session_state.history:
    st.subheader("Generation History")
    for i, item in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5 generations
        with st.expander(f"Generation {len(st.session_state.history) - i}: {item['prompt'][:30]}..."):
            st.image(item["image"], caption=f"Prompt: {item['prompt']}", use_container_width=True)
            st.write(f"Settings: Steps={item['steps']}, Guidance={item['guidance']}, Seed={item['seed']}")
            
            # Re-download option
            st.download_button(
                label="Download This Image",
                data=item["image"],
                file_name=f"generated_image_{len(st.session_state.history) - i}.png",
                mime="image/png"
            )

# Add helpful tips in the sidebar
st.sidebar.subheader("Tips for Better Results")
st.sidebar.markdown("""
- Be specific and detailed in your prompts
- Include art styles (e.g., "oil painting", "digital art")
- Mention lighting, mood, and perspective
- Use the negative prompt to remove unwanted elements
- Experiment with different guidance values
""")

# Footer
st.markdown("---")
st.caption("Built with Streamlit and Stable Diffusion v1.5")