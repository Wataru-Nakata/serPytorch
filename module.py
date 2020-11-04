import torch
from utils import load_filepaths_and_text
from pytorch_lightning.core.lightning import LightningModule
import model
import pytorch_lightning as pl

class SER(LightningModule):
    """
        1) loads audio and label
        2) compute mel-spectrograms from audio files
    """
    def __init__(self):
        super().__init__()
        self.model = model.SERModel()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = 1e-4
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    
    def forward(self,x):
        return self.model.forward(x)
    
    def training_step(self,batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits,y)
        self.log('train_loss', loss, on_step=True)
        self.log('train_acc_step', self.train_accuracy(logits, y))
        return loss
    def training_epoch_end(self, outputs) :
        self.log('train_acc_epoch', self.train_accuracy.compute())

    def validation_step(self,batch,batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits,y)
        self.log('val_loss', loss, on_step=True)
        self.log('val_acc_step', self.val_accuracy(logits, y))
        return loss
    
    def validation_epoch_end(self, outputs) :
        self.log('val_acc_epoch', self.val_accuracy.compute())


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)