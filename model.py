import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np 


'''
attention implementation from https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66
'''
class AttentionDecoder(nn.Module):
  
  def __init__(self, hidden_size, output_size, vocab_size):
    super(AttentionDecoder, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    
    self.attn = nn.Linear(hidden_size + output_size, 1)
    self.final = nn.Linear(output_size, vocab_size)
  
  def init_hidden(self):
    return (torch.zeros(1, 1, self.output_size).cuda(),
      torch.zeros(1, 1, self.output_size).cuda())
  
  def forward(self, decoder_hidden, encoder_outputs):
    
    weights = []
    for i in range(len(encoder_outputs)):
      weights.append(self.attn(torch.cat((decoder_hidden[0][0], 
                                          encoder_outputs[i]), dim = 1)))
    normalized_weights = F.softmax(torch.cat(weights, 1), 1)
    
    attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                             encoder_outputs.view(1, -1, self.hidden_size))
    
    output = torch.sum(attn_applied,1)
    
    return output, normalized_weights



class SERModel(nn.Module):
    def __init__(self):
        super(SERModel, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3,128,(3,5),padding=(1,2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,(3,5),padding=(1,2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2)
        )
        self.delta = torchaudio.transforms.ComputeDeltas()
        self.linear = nn.Linear(256*20,200)
        self.lstm = torch.nn.LSTM(200,128,batch_first=False,bidirectional=True)
        self.attention = AttentionDecoder(256,256,4)
        self.decoder_hidden = self.attention.init_hidden()
        self.classifier = torch.nn.Linear(256,4)
    def forward(self,x):
        delta_x = self.delta(x)
        delta_delta_x = self.delta(delta_x)
        x = x.unsqueeze(1)
        delta_x = delta_x.unsqueeze(1)
        delta_delta_x = delta_delta_x.unsqueeze(1)
        cnn_in_feature = torch.cat([x,delta_x,delta_delta_x],dim=1)
        phi_low = self.convs(cnn_in_feature)
        phi_low = phi_low.transpose(3,1).reshape(x.size(0),64,-1)
        phi_low = self.linear(phi_low)
        phi_low  = phi_low.transpose(0,1)
        phi_middle, _ = self.lstm(phi_low)
        phi_middle = phi_middle.transpose(0,1)

        phi_atteniton, weights = self.attention(self.decoder_hidden,phi_middle[0].unsqueeze(1))
        for i in range(1,x.size(0)):
            atteniton_out2, weights = self.attention(self.decoder_hidden,phi_middle[i].unsqueeze(1))
            phi_atteniton = torch.cat([phi_atteniton,atteniton_out2],dim=0)
        out = self.classifier(phi_atteniton)
        return out
    def get_intermediate_features(self,x):
        phi_low = self.convs(x)
        phi_middle = self.lstm(phi_low)
        phi_atteniton = self.attention(phi_middle)
        return phi_low, phi_middle, phi_atteniton



