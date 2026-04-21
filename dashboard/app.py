import streamlit as st
import requests
import json
from PIL import Image

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("Multimodal Fake News Detection")

st.markdown("""
Input article details below. The system utilizes text, associated images, and metadata through a Multimodal Transformer network.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Inputs")
    article_text = st.text_area("Article Text", height=200, value="Breaking: Local hero saves cats from tree.")
    
    author_cred = st.slider("Author Credibility Score", 0.0, 1.0, 0.8)
    domain_rep = st.slider("Domain Reputation Score", 0.0, 1.0, 0.9)
    
    uploaded_file = st.file_uploader("Upload article image (Optional)", type=["jpg", "jpeg", "png"])
    
    submit = st.button("Check Authenticity")

with col2:
    if submit:
        st.subheader("Prediction Result")
        with st.spinner("Analyzing with Multimodal Architecture..."):
            metadata = json.dumps({
                "author_credibility": author_cred,
                "domain_reputation": domain_rep
            })
            
            data = {"text": article_text, "metadata": metadata}
            files = None
            
            if uploaded_file is not None:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                st.image(uploaded_file, caption="Input Image", use_container_width=True)
            
            try:
                response = requests.post(API_URL, data=data, files=files)
                if response.status_code == 200:
                    result = response.json()
                    prob = result['fake_probability']
                    
                    st.metric("Fake Probability", f"{prob*100:.2f}%")
                    
                    if result['is_fake']:
                        st.error("🚨 This article is classified as FAKE NEWS.")
                    else:
                        st.success("✅ This article is classified as REAL.")
                        
                    if result.get('sources_found', 0) > 0:
                        st.info(f"🔍 Found {result['sources_found']} corroborating sources online.")
                        with st.expander("View Sources"):
                            st.markdown(result.get('search_context', ''))
                    else:
                        st.warning("⚠️ No corroborating sources found online for this claim.")
                else:
                    st.error(f"Failed to get a successful response: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Failed to connect to the backend API. Please make sure the FastAPI server is running on port 8000.")
