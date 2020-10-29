import torch
from utils import load_filepaths_and_text
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning
import model

class IEMOCAP(pytorch_lightning.LightningDataModule):
    def __init__(self,batch_size,data_dir: str = "filelists/iemocap"):
        super().__init__
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        


        