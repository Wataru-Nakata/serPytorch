import module
import pytorch_lightning as pl
import torch
import data_utils
import hparams
hparams = hparams.create_hparams()

train_dataset = data_utils.MelEmoLoader("filelists/iemocap_train.csv",hparams)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64,num_workers=20,pin_memory=True)
val_dataset = data_utils.MelEmoLoader("filelists/iemocap_val.csv",hparams)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=64,num_workers=20,pin_memory=True)

trainer = pl.Trainer(gpus=1)
model = module.SER()
trainer.fit(model,train_loader,val_loader)