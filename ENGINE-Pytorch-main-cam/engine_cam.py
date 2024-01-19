# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: CarpeDiem
@Date: 2023/12/20
@Description: 使用 Pytorch 框架，类激活图显示权重大小
@Improvement:  
"""

from pytorch_grad_cam import GradCAM
import torch.nn.functional as F 
import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import engine_model
import loaddata

class SigmoidFocalCrossEntropyLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalCrossEntropyLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        sigmoid_p = torch.sigmoid(y_pred)
        ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        p_t = y_true * sigmoid_p + (1 - y_true) * (1 - sigmoid_p)
        focal_loss = ce_loss * ((1 - p_t)**self.gamma)
        alpha_weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        loss = alpha_weight * focal_loss
        return loss.mean()

class EngineWithGradHook(engine_model.engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_gradients = None
        self.reparameterize_gradients = None
        self.generator_gradients = None
        self.diagnostician_share_gradients = None


    def activations_hook(self, name, grad):
        if name == 'encoder':
            self.encoder_gradients = grad
        elif name == 'reparameterize':
            self.reparameterize_gradients = grad
        elif name == 'generator':
            self.generator_gradients = grad
        elif name == 'diagnostician_share':
            self.diagnostician_share_gradients = grad

    def forward(self, x_SNP, c_demo, x_MRI):
        # Encoder
        encoder_output = self.encoder(x_SNP)
        encoder_output.register_hook(lambda grad: self.activations_hook('encoder', grad))
        mean, logvar = torch.chunk(encoder_output, 2, dim=1)

        # Reparameterize
        z_SNP = self.reparameterize(mean, logvar)
        z_SNP.register_hook(lambda grad: self.activations_hook('reparameterize', grad))

        # Generator
        x_MRI_fake, a = self.generate(z_SNP, c_demo)
        x_MRI_fake.register_hook(lambda grad: self.activations_hook('generator', grad))
        a.register_hook(lambda grad: self.activations_hook('generator', grad))

        # Diagnostician share
        feature = self.diagnostician_share(x_MRI_fake * a)
        feature.register_hook(lambda grad: self.activations_hook('diagnostician_share', grad))
        
        # Diagnostician clf
        logit_clf = self.diagnostician_clf(feature)
        return logit_clf


def get_device(memory_rate, my_seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(my_seed)
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.empty(int(total_memory * memory_rate), dtype=torch.int8, device='cuda')
        return 'cuda'
    else:
        return 'cpu'
        

# device configuration
my_seed = 42069
np.random.seed(my_seed)
torch.manual_seed(my_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = get_device(0.1, my_seed)

X_MRI_train, E_SNP_train, C_demo_train, Y_train, S_train, \
X_MRI_test, E_SNP_test, C_demo_test, Y_test, S_test = loaddata.load_dataset(1, ['AD', 'CN'])

N_in, N_o = 2000, 2                           # Look at the number of columns in Y_train
model_state_dict = torch.load('engine.pt')
model = EngineWithGradHook(N_in, N_o, device).to(device)
model.load_state_dict(model_state_dict)

class_idx = 0

x_SNP = torch.FloatTensor(E_SNP_test).to(device)
c_demo = torch.FloatTensor(C_demo_test).to(device)    
x_MRI = torch.FloatTensor(X_MRI_test).to(device)     
yb_clf = torch.FloatTensor(Y_test).to(device)
x_SNP.requires_grad = True

# integrated_gradients = integrated_gradients(x_SNP, c_demo, x_MRI, model)
# print(integrated_gradients.shape)

model.eval()

logit_clf = model(x_SNP, c_demo, x_MRI)
print(logit_clf.shape)
target = logit_clf[0][class_idx]
# print(target.shape)

loss_fn = SigmoidFocalCrossEntropyLoss()
L_clf = loss_fn(yb_clf, logit_clf )


model.zero_grad()

target.backward()       # Calculate gradient

# Get the gradient of x_SNP, which represents the partial derivative of each feature of 
# x_SNP with respect to the encoder output
x_SNP_grad = x_SNP.grad
print(x_SNP_grad.shape)

feature_importance = torch.mean(torch.abs(x_SNP_grad), dim=0)       # Calculate feature importance
print(feature_importance.shape)

top_values, top_indices = torch.topk(feature_importance, 10)        # Get the values and indexes of the top ten largest features

print("Top 10 feature indices:", top_indices.cpu().numpy())         # Print out the index of these features

plt.switch_backend('agg')                                           # # Visualize feature importance
plt.bar(range(len(feature_importance)), feature_importance.cpu().numpy())
plt.title("Feature Importance in x_SNP")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.savefig('feature_path_name.png')
plt.show()