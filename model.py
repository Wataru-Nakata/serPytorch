import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np 


'''
attention implementation from https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66
slightly modified for this application
'''
class AttentionDecoder(nn.Module):
  
  def __init__(self, hidden_size):
    super(AttentionDecoder, self).__init__()
    self.hidden_size = hidden_size
    
    self.attn = nn.Linear(hidden_size, 1,bias=False)
  
  
  def forward(self, encoder_outputs):
    
    weights = (self.attn(encoder_outputs))
    normalized_weights = F.softmax(weights, 1)
    normalized_weights = normalized_weights.repeat(1,1,256)
    
    attn_applied = torch.mul(encoder_outputs,normalized_weights)
    
    output = torch.sum(attn_applied,1)
    
    return output, normalized_weights



class SERModel(nn.Module):
    def __init__(self):
        super(SERModel, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3,128,(7,11),padding=(3,5)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,(7,11),padding=(3,5)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.Conv2d(256,256,(7,11),padding=(3,5)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.Conv2d(256,256,(7,11),padding=(3,5)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.Conv2d(256,256,(7,11),padding=(3,5)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.delta = torchaudio.transforms.ComputeDeltas()
        self.linear = nn.Linear(256*40,200)
        self.lstm = torch.nn.LSTM(200,128,batch_first=False,bidirectional=True)
        self.attention = AttentionDecoder(256)
        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256,64),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(64,4)
        )
    def forward(self,x):
        delta_x = self.delta(x)
        delta_delta_x = self.delta(delta_x)
        x = x.unsqueeze(1)
        delta_x = delta_x.unsqueeze(1)
        delta_delta_x = delta_delta_x.unsqueeze(1)
        cnn_in_feature = torch.cat([x,delta_x,delta_delta_x],dim=1)
        phi_low = self.convs(cnn_in_feature)
        phi_low = phi_low.transpose(3,1).reshape(x.size(0),x.size(3)//2,-1)
        phi_low = self.linear(phi_low)
        phi_low  = phi_low.transpose(0,1)
        phi_middle, _ = self.lstm(phi_low)
        phi_middle = phi_middle.transpose(0,1)

        phi_atteniton, weights = self.attention(phi_middle)
        out = self.classifier(phi_atteniton)
        return out
    def get_intermediate_features(self,x):
        phi_low = self.convs(x)
        phi_middle = self.lstm(phi_low)
        phi_atteniton = self.attention(phi_middle)
        return phi_low, phi_middle, phi_atteniton



