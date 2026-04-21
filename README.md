# Multimodal Fake News Detection System 🚨

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009485)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)

Welcome to the **Multimodal Fake News Detection** system! This project leverages state-of-the-art Deep Learning models to determine the authenticity of a news article by looking at multiple data sources simultaneously: text context, visual clues from images, source credibility, and live web corroboration.

## 🎯 Architecture
Fake news is becoming highly sophisticated. Simple text-checks are no longer enough. This system utilizes a **Multimodal Fusion Model** containing four separate analytic pillars:
1. **Text Modality (`DistilBERT`):** Reads the article to identify clickbait, aggressive syntax, and sensationalist patterns.
2. **Visual Modality (`ResNet18`):** Analyzes the cover image for contextual anomalies and misleading graphical elements.
3. **Metadata Modality (`MLP`):** Considers the reputation of the specific domain and author.
4. **Live Web Search Corroboration (`DDGS`):** The system actively scrapes the web to cross-reference the claim with legitimate news sources in real-time. If nobody is reporting the news, it drastically raises the fake probability score.

## 📁 Repository Structure
* `api/` : The FastAPI high-performance backend serving the AI models.
* `dashboard/` : The interactive Streamlit frontend for users to upload text and images.
* `src/` : The core neural network architecture.
    * `models/` : Feature extractors (DistilBERT, ResNet18, MLPs).
    * `data/` : Data preprocessing, tokenization, and scaling.
    * `training/` : The PyTorch Lightning training loops.
* `scripts/` : Training scripts for generating the model weights.

---

## 🚀 How to Run the Application Locally

You will need to open **two separate terminal windows** to get the full application running (one for the backend API, and one for the frontend UI).

### 1. Installation 
First, ensure you are in the project root directory and activate your virtual environment:

**Windows:**
```bash
.\venv\Scripts\activate
```
**Mac/Linux:**
```bash
source venv/bin/activate
```

Then, install all required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend
In your activated terminal, run the following command to start the PyTorch model inferencing engine:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
> **Note:** The very first time you start this API, it may take a few minutes to download the `distilbert-base-uncased` and `ResNet18` deep learning models from the Hugging Face Hub. Wait until you see `Application startup complete.`

### 3. Start the Streamlit Dashboard
Open a **new, second terminal window**, activate your virtual environment again, and run:
```bash
streamlit run dashboard/app.py
```
This will automatically open up `http://localhost:8501` in your web browser. 

---

## 🧪 Testing the Detection
1. Paste a fake or highly sensational headline into the Streamlit dashboard text box (e.g. *"Aliens have invaded the Eiffel tower!*").
2. Adjust your theoretical domain reputation.
3. Click "Check Authenticity". 
4. The backend will analyze the text, search the web via DuckDuckGo News natively, realize the story isn't corroborated anywhere, and return a **FAKE NEWS** alert with a percentage score!

## ⚙️ Tech Stack
* **Machine Learning:** PyTorch, PyTorch Lightning, Torchvision, Transformers
* **Backend Pipeline:** FastAPI, Uvicorn, Python-Multipart
* **Frontend:** Streamlit
* **Web Searching:** DuckDuckGo Search (`ddgs`)
