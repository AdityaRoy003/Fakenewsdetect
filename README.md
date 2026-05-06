# Enterprise Multimodal Fake News Dashboard 🚨

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009485)
![Vanilla JS](https://img.shields.io/badge/Vanilla%20JS-Frontend-F7DF1E)

Welcome to the **Enterprise Fake News Intelligence Dashboard**. This application leverages a state-of-the-art Deep Learning fusion model wrapped in a stunning, high-performance glassmorphic UI. It analyzes the authenticity of a news article by looking at multiple data vectors simultaneously: text context, visual clues from images, source credibility, and live web corroboration.

## ✨ Next-Gen Features

We have completely overhauled the system from a basic backend into a fully-fledged journalistic intelligence platform.

*   **🌐 Global Intelligence Hub:** A dedicated modal featuring a **Source Lineage Timeline** (tracking how a story evolved from origin to amplification) and a CSS-rendered **Trust Network Graph** mapping out bots vs. credible sources.
*   **📈 Bias & Sentiment Analysis:** A dynamic "Bias Meter" gauge that analyzes text for political bias and emotional manipulation.
*   **🎭 Deepfake & Media Authentication:** Robust support for analyzing `.mp4`, `.webm`, `.jpg`, and `.png` files, with UI elements dedicated to flagging synthetic metadata and lighting anomalies.
*   **🌍 Community Verification Layer:** A crowdsourced "Community Fact-Check" module (similar to X/Twitter Community Notes) that dynamically renders known debunks and context for flagged articles.
*   **🪄 3D Glassmorphic UI:** A buttery smooth, responsive, dark-mode frontend featuring 3D flip-card animations, drag-and-drop media support, and automated local history caching.
*   **⚙️ Advanced User Settings:** Fully persistent frontend configuration allowing you to adjust minimum credibility thresholds, shift the AI weight focus (Author vs. Domain), enable real **Push Notifications**, and change the UI language on the fly.

## 🎯 Neural Architecture

The core relies on a **Multimodal Fusion Model** containing four separate analytic pillars:
1.  **Text Modality (`DistilBERT`):** Reads the article to identify clickbait, aggressive syntax, and sensationalist patterns.
2.  **Visual Modality (`ResNet18`):** Analyzes the cover media for contextual anomalies and misleading graphical elements.
3.  **Metadata Modality (`MLP`):** Considers the reputation of the specific domain and author.
4.  **Live Web Search Corroboration (`DDGS`):** The system actively scrapes the web to cross-reference the claim with legitimate news sources in real-time. 

## 📁 Repository Structure
*   `api/` : The FastAPI high-performance backend serving the AI models and the static UI.
*   `static/` : The completely bespoke HTML/CSS/Vanilla JS frontend dashboard (No clunky frameworks!).
*   `src/` : The core neural network architecture (`extractors.py`, `fusion.py`).
*   `scripts/` : Advanced training scripts utilizing **PyTorch Lightning** and dataset preparation scripts.

---

## 🚀 How to Run the Application Locally

Because we upgraded to a streamlined, statically-served architecture, you only need to run a **single command**!

### 1. Installation 
Ensure you are in the project root directory and activate your virtual environment:

**Windows:**
```bash
.\venv\Scripts\activate
```
**Mac/Linux:**
```bash
source venv/bin/activate
```

Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Start the Server
Run the following command to start the PyTorch inferencing engine and serve the stunning frontend:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
> **Note:** The very first time you start this API, it may take a few minutes to download the `distilbert-base-uncased` and `ResNet18` models from Hugging Face.

### 3. Open the Dashboard
Simply open your web browser and navigate to:
**http://localhost:8000**

---

## 🧠 Training Your Own AI

The backend is fully equipped with `pytorch-lightning` to train on massive datasets (like the Kaggle ISOT Fake News Dataset). 

1. Download the Kaggle dataset.
2. Run `python scripts/prepare_dataset.py` to intelligently format the text data into multimodal tensors.
3. Run `python scripts/train.py`.
4. The API is hardcoded to automatically scan the `lightning_logs/` folder and dynamically load the absolute newest `.ckpt` model weight file upon startup!

## ⚙️ Tech Stack
*   **Machine Learning:** PyTorch, PyTorch Lightning, Transformers, ResNet
*   **Backend Pipeline:** FastAPI, Uvicorn, DuckDuckGo Search (`ddgs`)
*   **Frontend Design:** HTML5, CSS3 (Glassmorphism & 3D Transforms), Vanilla JavaScript
