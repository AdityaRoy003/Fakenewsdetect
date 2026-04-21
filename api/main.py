from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import uvicorn
import torch
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocess import TextPreprocessor, ImagePreprocessor, MetadataPreprocessor
from src.models.fusion import MultimodalFusionModel

app = FastAPI(title="Fake News Detection API")

# Initialize models globally
# In a real environment, the model weights would be loaded via `model.load_state_dict(...)`
text_prep = TextPreprocessor(max_length=64)
img_prep = ImagePreprocessor()
meta_prep = MetadataPreprocessor()
model = MultimodalFusionModel(metadata_input_dim=2, text_dim=768, image_dim=768, num_classes=1)
model.eval()

class PredictionResponse(BaseModel):
    fake_probability: float
    is_fake: bool
    sources_found: int = 0
    search_context: str = ""

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    text: str = Form(""),
    metadata: str = Form("{}"), 
    file: UploadFile = File(None)
):
    # Process text
    text_feat = text_prep.preprocess(text)
    t_id = text_feat['input_ids'].unsqueeze(0)
    t_mask = text_feat['attention_mask'].unsqueeze(0)
    
    # Process image
    if file is not None:
        from PIL import Image
        import io
        img_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            i_feat = img_prep.preprocess(image).unsqueeze(0)
        except Exception:
            i_feat = torch.zeros((1, 3, 224, 224))
    else:
        i_feat = torch.zeros((1, 3, 224, 224))
        
    # Process metadata
    try:
        meta_dict = json.loads(metadata)
        meta_array = [float(meta_dict.get('author_credibility', 0.5)), float(meta_dict.get('domain_reputation', 0.5))]
    except Exception:
        meta_array = [0.5, 0.5]
        
    m_feat = meta_prep.preprocess(meta_array).unsqueeze(0)
    
    with torch.no_grad():
        out = model(t_id, t_mask, i_feat, m_feat)
        prob = torch.sigmoid(out).item()
        
    # --- Web Corroboration Override ---
    sources_found = 0
    search_context = ""
    try:
        import asyncio
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException
        query = " ".join(text.split()[:10]) # Use a tighter query length
        if len(query) > 5:
            def _fetch_news():
                with DDGS() as ddgs:
                    return list(ddgs.news(query, max_results=3))
            
            try:
                results = await asyncio.to_thread(_fetch_news)
                if results:
                    sources_found = len(results)
                    search_context = "\n".join([f"- {r.get('title', '')}" for r in results])
                    prob = max(0.05, prob - 0.45) # News corroborated it, extremely likely to be real
                else:
                    prob = min(0.95, prob + 0.40) # Highly suspicious
            except DDGSException:
                # DDGS throws this when literally 0 news articles are found
                prob = min(0.95, prob + 0.40) 
                search_context = "No major news outlets are reporting this."
    except Exception as e:
        pass # If some other major crash occurs, just use base model
        
    return PredictionResponse(
        fake_probability=prob,
        is_fake=(prob > 0.5),
        sources_found=sources_found,
        search_context=search_context
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
