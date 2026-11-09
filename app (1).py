import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import streamlit.components.v1 as components

# Set mobile-friendly viewport configuration
st.set_page_config(page_title="CDFP Plant Identifier", page_icon="🌿", layout="centered")

# 1. PWA Installation Support Layer
pwa_install_script = """
<script>
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', function() {
            navigator.serviceWorker.register('/sw.js').catch(function(err) {
                console.log('ServiceWorker registration failed: ', err);
            });
        });
    }
</script>
"""
components.html(pwa_install_script, height=0)

# Official taxonomy map based on Co's Digital Flora of the Philippines categories
CDFP_TAXONOMY = {
    0: {"species": "Bulbophyllum (Orchidaceae)", "status": "Native / Endemic", "info": "One of the largest orchid genera in the Philippines."},
    1: {"species": "Hoya (Apocynaceae)", "status": "Native / High Endemism", "info": "Wax plants widely distributed across Philippine tropical forests."},
    2: {"species": "Begonia (Begoniaceae)", "status": "Endemic Species Dominant", "info": "Terrestrial herbs highly specialized to local microclimates."},
    3: {"species": "Syzygium (Myrtaceae)", "status": "Native Flora", "info": "Commonly known as lipote or duhat relatives in local forest systems."}
}

@st.cache_resource
def load_pytorch_model():
    """ Loads and caches the PyTorch model state dict on CPU for stability. """
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CDFP_TAXONOMY))
    
    try:
        checkpoint_path = "model.pth"
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.warning("⚠️ 'model.pth' file not found in directory root. Running in Simulation mode.")
    
    model.eval()
    return model

model = load_pytorch_model()

def preprocess_and_predict(img, model_instance):
    """ Transforms input PIL image to PyTorch tensor format. """
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform_pipeline(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model_instance(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=0)
        conf, class_idx = torch.max(probabilities, dim=0)
    return class_idx.item(), conf.item()

# UI Layout configuration
st.title("🌿 CDFP Plant Identifier")
st.markdown("Instantly identify native, naturalized, and endemic Philippine vascular flora.")

uploaded_file = st.camera_input("Capture plant image") or st.file_uploader("Or upload from gallery", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Target Specimen", use_container_width=True)
    
    with st.spinner("Analyzing botanical features against CDFP digital registry..."):
        try:
            class_idx, confidence = preprocess_and_predict(image, model)
            result = CDFP_TAXONOMY.get(class_idx, {"species": "Unknown Flora", "status": "Unverified", "info": "Data unavailable."})
            
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Identified Genus/Species", value=result["species"])
            with col2:
                st.metric(label="Confidence Score", value=f"{confidence * 100:.1f}%")
                
            st.info(f"**Status:** {result['status']}\n\n{result['info']}")
            
        except Exception as e:
            st.error(f"Inference pipeline execution error: {str(e)}")

# Integrated Mobile Installation Instructions Menu
st.write("---")
with st.expander("📲 How to Install this App on Your Phone"):
    st.markdown("""
    You can install the **CDFP Plant Identifier** directly onto your phone's home screen without using an app store. Follow these quick steps:

    ### 🤖 For Android Users (Google Chrome)
    1. Open your custom live app link in **Chrome**.
    2. Look for an automatic pop-up at the bottom saying **'Add to Home Screen'** or **'Install App'** and tap it.
    3. If you don't see it, tap the **Three Dots (Menu)** in the top-right corner.
    4. Select **'Install app'** or **'Add to Home Screen'**.

    ### 🍏 For iPhone Users (Safari)
    1. Open your custom live app link specifically in the **Safari** browser.
    2. Tap the **Share** button (the square icon with an arrow pointing up at the bottom of your screen).
    3. Scroll down the share menu and tap **'Add to Home Screen'**.
    4. Tap **'Add'** in the top right corner to confirm.
    """)
