import module
import pytorch_lightning as pl
import torch
import data_utils
import hparams
hparams = hparams.create_hparams()

dataset = data_utils.MelEmoLoader("filelists/iemocap_train.csv",hparams)
train_loader = torch.utils.data.DataLoader(dataset)

trainer = pl.Trainer()
model = module.SER()
trainer.fit(model,train_loader)