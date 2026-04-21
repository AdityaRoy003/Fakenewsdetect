import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class FakeNewsLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.val_preds = []
        self.val_labels = []

    def forward(self, text_ids, text_mask, image, metadata):
        return self.model(text_ids, text_mask, image, metadata)

    def _shared_step(self, batch, batch_idx):
        text_ids = batch['text_input_ids']
        text_mask = batch['text_attention_mask']
        image = batch['image']
        metadata = batch['metadata']
        labels = batch['label'].unsqueeze(1)

        logits = self(text_ids, text_mask, image, metadata)
        loss = self.criterion(logits, labels)
        
        preds = torch.sigmoid(logits)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(labels.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return
            
        preds = torch.cat(self.val_preds).numpy()
        labels = torch.cat(self.val_labels).numpy()
        
        preds_binary = (preds > 0.5).astype(int)
        acc = accuracy_score(labels, preds_binary)
        f1 = f1_score(labels, preds_binary)
        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            auc = 0.5  # Default if only one class in batch

        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_auc', auc, prog_bar=True)
        
        self.val_preds.clear()
        self.val_labels.clear()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
