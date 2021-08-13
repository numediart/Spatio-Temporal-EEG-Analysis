'''
Created by Victor Delvigne
ISIA Lab, Faculty of Engineering University of Mons, Mons (Belgium)
victor.delvigne@umons.ac.be
Copyright (C) 2021 - UMons
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
'''

import torch
import math

import numpy as np
import torch.nn as nn

from torchvision import models
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Encoder( nn.Module ):
    def __init__(self, e_dim, n_chan, f_dim = 20, dropout = 0.25):
        super( Encoder, self ).__init__()
        self.e_dim = e_dim
        self.batch_norm = nn.BatchNorm1d(30)  
        self.n_chan = n_chan
        self.f_dim = f_dim #feature dimension
        self.dropout = dropout
        #self.encoding = nn.ModuleList([nn.Linear(self.f_dim, self.e_dim) for i in range(self.n_chan)])
        self.encoding = nn.Linear(self.f_dim, self.e_dim)
        self.pos_encoding = PositionalEncoding(self.e_dim, self.dropout)
    def forward(self, x):
        x = torch.stack([self.encoding(x[:, i]) for i in range(self.n_chan)], 0) * math.sqrt(self.e_dim)  
        x = self.pos_encoding(x)
        return x


class Decoder( nn.Module ):
    def __init__(self, h_dim, n_chan, out_dim):
        super( Decoder, self ).__init__()
        self.h_dim = h_dim
        self.n_chan = n_chan
        self.decoding = nn.Linear(h_dim, out_dim)
    def forward(self, x):
        x = torch.stack([self.decoding(x[i]) for i in range(self.n_chan)], 0)
        x = x.transpose(0,1)
        return x

class Attention(nn.Module):
    def __init__(self, e_dim, h_dim, nhead, nlayer, n_chan, f_dim, out_dim, dropout=0.25):
        super( Attention, self ).__init__()
        self.encoder = Encoder(e_dim=e_dim, n_chan=n_chan, f_dim=f_dim).cuda()
        self.nhead = nhead
        self.nlayer = nlayer
        self.e_dim = e_dim
        self.h_dim = h_dim
        self.dropout = dropout
        
        encoder_layer = TransformerEncoderLayer(d_model=self.e_dim, nhead=self.nhead, dim_feedforward=self.e_dim, dropout=self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, self.nlayer)
        self.decoder = Decoder(h_dim=e_dim, n_chan=n_chan, out_dim=out_dim).cuda()
        self.Classifier = nn.Sequential(
            nn.Linear(out_dim*n_chan, 64),
            nn.Softmax(dim=1),
        )
        
        self.s_max = nn.Softmax(dim=1)
        
        self.bn = nn.BatchNorm1d(e_dim)
        
    def forward(self, x):
        b_size = x.shape[0]
        x = self.encoder(x)
        x = self.transformer_encoder(x)
        #x = self.bn(x[-1])
        x = self.decoder(x)
        x = x.reshape(b_size, -1)
        return x

class MultiAttention(nn.Module):
    def __init__(self):
        super(MultiAttention, self ).__init__()
        self.spatial_attention = Attention(e_dim=256,h_dim=256, nhead=1, 
            nlayer=2, n_chan=30, f_dim=20*5, out_dim=10)
        self.temporal_attention = Attention(e_dim=256,h_dim=64, nhead=1, 
            nlayer=2, n_chan=20, f_dim=30*5, out_dim=15)
        self.frequential_attention = Attention(e_dim=512,h_dim=64, nhead=1, 
            nlayer=2, n_chan=5, f_dim=30*20, out_dim=60)
        
        self.Classifier = nn.Sequential(
            nn.Linear(900, 64),
            nn.Dropout(0.25),
            nn.ReLU(True),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
            )
        
    def forward(self, x):
        b_size = x.shape[0]
        f_size = x.shape[1]
        c_size = x.shape[2]
        t_size = x.shape[3]
        
        s_feat = self.spatial_attention(x.transpose(1,2).reshape(b_size, c_size, -1))
        t_feat = self.temporal_attention(x.transpose(1, 3).reshape(b_size, t_size, -1))
        f_feat = self.frequential_attention(x.reshape(b_size, f_size, -1))
        
        feat = torch.cat([s_feat, t_feat, f_feat], axis=1)
        out = self.Classifier(feat)
        
        return out

class BiHDM( nn.Module ):
    def __init__(self, hemi_id):
        super( BiHDM, self ).__init__()
        self.id = hemi_id
        
        ### Horizontal Stream
        self.RNN_lh = nn.RNN(100, 32, 1, batch_first=False, bidirectional=True)
        self.RNN_rh = nn.RNN(100, 32, 1, batch_first=False, bidirectional=True)
        self.RNN_h = nn.RNN(64, 128, 1, batch_first=False, bidirectional=True)
        
        ### Vertical Stream
        self.RNN_lv = nn.RNN(100, 32, 1, batch_first=False, bidirectional=True)
        self.RNN_rv = nn.RNN(100, 32, 1, batch_first=False, bidirectional=True)
        self.RNN_v = nn.RNN(64, 128, 1, batch_first=False, bidirectional=True)
        
        ### Combination
        self.comb_h = nn.Linear(384, 128)
        self.comb_v = nn.Linear(384, 128)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 4),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(4,2),
            nn.Softmax(dim=1)
            )
        
        self.batch_norm = nn.BatchNorm1d(30)        
        self.batch_norm_end = nn.BatchNorm1d(12)
        
    def forward(self, x):
        
        
        b_size = x.shape[0]
        
        
        x = self.batch_norm(x)
        ### Horizontal Stream
        s_lh, _ = self.RNN_lh(x[:, self.id[0]].transpose(0,1))
        s_rh, _ = self.RNN_rh(x[:, self.id[1]].transpose(0,1))
        
        ### Vertical Stream
        s_lv, _ = self.RNN_lv(x[:, self.id[0]].transpose(0,1))
        s_rv, _ = self.RNN_rv(x[:, self.id[1]].transpose(0,1))
                
        ### Pairwise Operation
        s_h = s_lh - s_rh
        s_v = s_lv - s_rv
        
        s_h = self.batch_norm_end(s_h.transpose(0,1)).transpose(0,1)
        s_v = self.batch_norm_end(s_v.transpose(0,1)).transpose(0,1)
        
        ### Combination
        s_h, _ = self.RNN_h(s_h)
        #s_h = self.comb_h(s_h.reshape((b_size, -1)))
        s_v, _ = self.RNN_v(s_v)
        #s_v = self.comb_v(s_v.reshape((b_size, -1)))
        
        s = s_v[-1] + s_h[-1] 
        
        return s

l_h = np.asarray([3,  2, 4, 5, 6, 8, 7, 9, 10, 11, 13, 12])
r_h = np.asarray([29, 28, 25, 26, 27, 24, 23, 19, 20, 21, 18, 17])
l_v = np.asarray([3, 4, 9, 13, 14, 2, 5, 7, 10, 6, 11, 12])
r_v = np.asarray([29, 25, 19, 18, 16, 28, 26, 23, 20, 27, 21, 17])

hem = np.stack((l_h, r_h, l_v, r_v))

class tot_RNN( nn.Module ):
    def __init__(self, hemi_id=hem):
        super( tot_RNN, self ).__init__()
        
        ### Spatial RNN
        self.s_RNN = BiHDM(hemi_id=hemi_id).cuda()
        
        ### Frequential RNN
        self.f_RNN = nn.RNN(600, 128, 1, batch_first=False, bidirectional=True)
        
        ### Temporal RNN
        self.t_RNN = nn.RNN(150, 128, 1, batch_first=False, bidirectional=True)
        
        self.Classifier = nn.Sequential(
            nn.Linear(3*256, 64),
            nn.Dropout(0.25),
            nn.ReLU(True),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
            )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        b_size = x.shape[0]
        c_dim = x.shape[1]
        f_dim = x.shape[2]
        t_dim = x.shape[3]
        
        spatial_x = x.reshape(b_size, c_dim, -1)
        freq_x = x.transpose(1,2).reshape(b_size, f_dim, -1)
        temp_x = x.transpose(1,3).reshape(b_size, t_dim, -1)
        
        h_spat = self.s_RNN(spatial_x)
        h_freq, _ = self.f_RNN(freq_x.transpose(0,1))
        h_freq = h_freq[-1]
        h_temp, _ = self.t_RNN(temp_x.transpose(0,1))
        h_temp = h_temp[-1]
        
        
        hidden = torch.cat([h_spat, h_freq, h_temp], axis=1)
        
        x = self.Classifier(hidden)
        
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class CNN( nn.Module ):
    def __init__(self, in_dim=100):
        super( CNN, self ).__init__()
        
        self.ClassifierCNN = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            models.resnet18().layer1,
            
            models.resnet18().layer2,
            
            models.resnet18().layer3,
            
            models.resnet18().layer4,
            
            nn.AdaptiveAvgPool2d(output_size=(1, 1))

        )        
        
        self.ClassifierFC = nn.Sequential(
                Flatten(),
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2),
                nn.Softmax(dim=1),
            )
        
        
    def forward(self, x):
        x = self.ClassifierCNN(x)
        x = self.ClassifierFC(x.view(x.shape[0], -1))
        return x