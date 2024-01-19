# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: CarpeDiem
@Date: 2023/5/17
@Description: 使用 Pytorch 框架，重构代码
@Improvement:  
"""

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from sklearn.preprocessing import OneHotEncoder

class engine(nn.Module):
    def __init__(self, N_in, N_o, device):
        """
        初始化网络参数
        """
        super(engine, self).__init__()
        self.N_in = N_in
        self.N_o = N_o
        self.device = device

        """SNP Representation Module"""
        # Encoder network, Q
        self.encoder = nn.Sequential(nn.Linear(N_in, 500),          # F_SNP = 2000
                                     nn.ELU(),
                                     nn.Linear(500, 100),)           # 2 * dim(z_SNP) = 100

        # Decoder network, P
        self.decoder = nn.Sequential(nn.Linear(50, 500),            # dim(z_SNP) = 50
                                     nn.ELU(),
                                     nn.Linear(500, 2000))          # F_SNP = 2000

        """Attentive Vector Generation Module"""
        # Generator network, G
        self.generator = nn.Sequential(nn.Linear(56, 100),          # # dim(z) = dim(z_SNP) + dim(c) = 54 Now 56
                                       nn.ELU(),
                                       nn.Linear(100, 180), 
                                       nn.Sigmoid())                # 2 * F_MRI = 90*2 = 180
        
        # Discriminator network, D
        self.discriminator = nn.Sequential(nn.Linear(90, 1),        # F_MRI = 90
                                           nn.Sigmoid())            # real or fake

        """Diagnostician Module"""
        # Diagnostician network, C
        self.diagnostician_share = nn.Sequential(nn.Linear(90, 25), # dim(Concat(a, x_MRI)) = 90
                                                 nn.ELU())
        
        self.diagnostician_clf = nn.Sequential(nn.Linear(25, self.N_o)) 
        self.diagnostician_reg = nn.Sequential(nn.Linear(25, 1))
    
    # Reconstructed SNPs sampling
    def sample(self, eps=None):
        if eps is None:
            eps = torch.randn(10, 50).to(self.device)
        return self.decode(eps, apply_sigmoid=True)
    
    # Represent mu and sigma from the input SNP
    def encode(self, x_SNP):
        mean, logvar =  torch.chunk(self.encoder(x_SNP), 2, dim=1)
        return mean, logvar
    
    # Construct latent distribution
    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean).to(self.device)
        return eps * torch.exp(logvar * .5) + mean

    # Reconstruct the input SNP
    def decode(self, z_SNP, apply_sigmoid=False):    
        logits = self.decoder(z_SNP)
        if apply_sigmoid:
            probs = F.sigmoid(logits).to(self.device)
            return probs
        return logits

    # Attentive vector and fake neuroimaging generation
    def generate(self, z_SNP, c_demo):
        z = torch.cat((c_demo, z_SNP), dim=-1)
        a, x_MRI_fake = torch.chunk(self.generator(z), 2, dim=-1)
        return x_MRI_fake, a

    # Classify the real and the fake neuroimaging
    def discriminate(self, x_MRI_real_or_fake):
        return self.discriminator(x_MRI_real_or_fake)

    # Downstream tasks; brain disease diagnosis and cognitive score prediction
    def diagnose(self, x_MRI, a, apply_logistic_activation=False):                                                
        feature = self.diagnostician_share(x_MRI*a)         # Hadamard production of the attentive vector                                                                                                
        logit_clf = self.diagnostician_clf(feature)
        # print(logit_clf)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            # y_hat = logit_clf.argmax(dim=1)               # 0 1
            y_hat = torch.softmax(logit_clf, -1)
            s_hat = F.sigmoid(logit_reg)
            return y_hat, s_hat
        return logit_clf, logit_reg
    
    def predict(self, x_MRI, a, apply_logistic_activation=False):
        feature = self.diagnostician_share(torch.mul(x_MRI, a))         # Hadamard production of the attentive vector                                                  
        logit_clf = self.diagnostician_clf(feature)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            y_hat = logit_clf.argmax(dim=1)            
            # encoder = OneHotEncoder(sparse=False)
            # y_hat = encoder.fit_transform(y_hat)          # 0 0 1
            s_hat = F.sigmoid(logit_reg)
            return y_hat, s_hat
        return logit_clf, logit_reg