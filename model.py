import torch.nn as nn
import torch
import torch.functional as F
import torchaudio


'''
attention implementation from https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66
'''
class AttentionDecoder(nn.Module):
  
  def __init__(self, hidden_size, output_size, vocab_size):
    super(AttentionDecoder, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    
    self.attn = nn.Linear(hidden_size + output_size, 1)
    self.lstm = nn.LSTM(hidden_size + vocab_size, output_size) #if we are using embedding hidden_size should be added with embedding of vocab size
    self.final = nn.Linear(output_size, vocab_size)
  
  def init_hidden(self):
    return (torch.zeros(1, 1, self.output_size),
      torch.zeros(1, 1, self.output_size))
  
  def forward(self, decoder_hidden, encoder_outputs, input):
    
    weights = []
    for i in range(len(encoder_outputs)):
      print(decoder_hidden[0][0].shape)
      print(encoder_outputs[0].shape)
      weights.append(self.attn(torch.cat((decoder_hidden[0][0], 
                                          encoder_outputs[i]), dim = 1)))
    normalized_weights = F.softmax(torch.cat(weights, 1), 1)
    
    attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                             encoder_outputs.view(1, -1, self.hidden_size))
    
    output = torch.cat((attn_applied[0], input[0]), dim = 1) #if we are using embedding, use embedding of input here instead
    
    
    output = torch.sum(output,1)
    
    return output, normalized_weights



class SERModel(torch.nn.module):
    def __init__(self):
        super(SERModel, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3,128,(5,3),padding=(2,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,(5,3),padding=(2,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2)
        )
        self.delta = torchaudio.transforms.ComputeDeltas()
        self.lstm = torch.nn.LSTM()
        self.attention = AttentionDecoder()
        self.classifier = nn.Linear(128,8)
    def forward(self,x):
        delta_x = self.delta(x)
        delta_delta_x = self.delta(delta_x)
        cnn_in_feature = torch.cat(x,delta_x,delta_delta_x,dim=1)
        phi_low = self.convs(x)
        phi_middle = self.lstm(phi_low)
        phi_atteniton = self.attention(phi_middle)
        output = self.classifier(phi_atteniton)
        return output
    def get_intermediate_features(self,x):
        phi_low = self.convs(x)
        phi_middle = self.lstm(phi_low)
        phi_atteniton = self.attention(phi_middle)
        return phi_low, phi_middle, phi_atteniton



