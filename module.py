import torch
from utils import load_filepaths_and_text
from pytorch_lightning.core.lightning import LightningModule
import model

class SER(LightningModule):
    """
        1) loads audio and label
        2) compute mel-spectrograms from audio files
    """
    def __init__(self):
        super().__init__()
        self.model = model.SERModel()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = 0.01
    
    def forward(self,x):
        return self.model.forward(x)
    
    def training_step(self,batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits,y)
        self.log('train_loss', loss, on_step=True)
        return loss
    def validation_step(self,batch,batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits,y)
        self.log('val_loss', loss, on_step=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)