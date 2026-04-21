import torch
import numpy as np
import shap

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

class FusionExplainer:
    def __init__(self, model, text_preprocessor, image_preprocessor, metadata_preprocessor):
        self.model = model
        self.text_prep = text_preprocessor
        self.img_prep = image_preprocessor
        self.meta_prep = metadata_preprocessor
        self.device = next(model.parameters()).device

    def explain_image(self, text_ids, text_mask, image_tensor, metadata_tensor, original_image_rgb):
        """
        Generate Grad-CAM heatmap over the original image.
        original_image_rgb: numpy array of shape (H, W, 3), scaled [0, 1]
        """
        if not HAS_GRAD_CAM:
            print("pytorch_grad_cam not installed")
            return None

        # Custom wrapper to lock text and metadata inputs
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model, t_id, t_mask, m_tensor):
                super().__init__()
                self.model = model
                self.t_id = t_id
                self.t_mask = t_mask
                self.m_tensor = m_tensor
                
            def forward(self, img):
                return self.model(self.t_id, self.t_mask, img, self.m_tensor)
                
        wrapper = ModelWrapper(self.model, text_ids, text_mask, metadata_tensor).to(self.device)
        
        # Identify target layers for ResNet
        try:
            target_layers = [self.model.image_extractor.features[-1]]
            cam = GradCAM(model=wrapper, target_layers=target_layers)
            
            grayscale_cam = cam(input_tensor=image_tensor, targets=None)[0, :]
            visualization = show_cam_on_image(original_image_rgb, grayscale_cam, use_rgb=True)
            return visualization
        except Exception as e:
            print(f"Failed to generate Grad-CAM: {e}")
            return None

    def explain_text(self, text_list, image_tensor, metadata_tensor):
        """
        Use SHAP to explain text model predictions.
        text_list: list of strings (e.g., ["This is a fake news article"])
        """
        def predict_fn(texts):
            val_preds = []
            for t in texts:
                prep = self.text_prep.preprocess(t)
                t_id = prep['input_ids'].unsqueeze(0).to(self.device)
                t_mask = prep['attention_mask'].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(t_id, t_mask, image_tensor, metadata_tensor)
                val_preds.append(torch.sigmoid(out).item())
            return np.array(val_preds)
        
        explainer = shap.Explainer(predict_fn, self.text_prep.tokenizer)
        shap_values = explainer(text_list)
        return shap_values
