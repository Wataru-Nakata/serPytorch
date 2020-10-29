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
        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def forward(self):
        return model.forward()
    
    def training_step(self,batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits,y)
        self.log('train_loss', loss, on_step=True)
        return loss

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)