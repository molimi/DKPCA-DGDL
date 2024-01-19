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
import engine
import loaddata
import os
import hydra
from typing import List
import definitions
from definitions import OUT_DIR
from definitions import device
from utils import get_params, Lin_View
import geoopt
from train_lin import terminatingcondition_factory, optimizer_factory, initialize, params_factory, load_model
import loaddata
import pandas
from datetime import datetime
from omegaconf import DictConfig
from encoder_decoder import Net1, Net3
import utils
import logging, sys

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

class Level():
    def __init__(self, phi, psi, s):
        self.phi = phi
        self.psi = psi
        self.s = s


def train_deepkpca(xtrain, levels: List[Level], args_optimizer, ae_weight=None, model_to_load=None):
    if ae_weight is None:
        ae_weight = 10.0
    from kernels import LinearKernel

    N = xtrain.shape[0]
    s = [level.s for level in levels]

    H2_tilde = torch.randn((N, s[1]), device=definitions.device)
    H1_tilde = torch.randn((N, s[0]), device=definitions.device)
    if args_optimizer.name == "geoopt":
        with torch.no_grad():
            H1_tilde = geoopt.ManifoldParameter(H1_tilde, manifold=geoopt.Stiefel(), requires_grad=True).proj_()
            H2_tilde = geoopt.ManifoldParameter(H2_tilde, manifold=geoopt.Stiefel(), requires_grad=True).proj_()
    L1_tilde = torch.randn((s[0],), device=definitions.device, requires_grad=True)
    L2_tilde = torch.randn((s[1],), device=definitions.device, requires_grad=True)
    params = params_factory([H1_tilde, H2_tilde], [L1_tilde, L2_tilde], [], args_optimizer)
    optimizer = optimizer_factory(params, args_optimizer)
    if 'lr_scheduler' in args_optimizer and args_optimizer.lr_scheduler.factor < 1:
        from utils import ReduceLROnPlateau
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, **args_optimizer.lr_scheduler)
    else:
        lr_scheduler = None

    # Explicit feature map
    phi1, psi1 = levels[0].phi, levels[0].psi
    optimizer2 = torch.optim.Adam(list(phi1.parameters()) + list(psi1.parameters()), lr=args_optimizer.lr_nn, weight_decay=0)

    # Load saved model
    if model_to_load is not None:
        H1_tilde.data.copy_(model_to_load["H1"])
        H2_tilde.data.copy_(model_to_load["H2"])
        L1_tilde.data.copy_(model_to_load["L1"])
        L2_tilde.data.copy_(model_to_load["L2"])
        phi1.load_state_dict(model_to_load["phi1"])
        phi1.eval()
        psi1.load_state_dict(model_to_load["psi1"])
        psi1.eval()

    lin_kernel = LinearKernel()
    def rkm2(x):
        op1 = phi1(x)
        op1 = op1 - torch.mean(op1, dim=0)
        Kx = op1 @ op1.t()

        f1 = 0.5 * torch.norm(lin_kernel(H1_tilde.t()) - H2_tilde @ torch.diag(L2_tilde) @ H2_tilde.t(), 'fro') ** 2 + \
             0.5 * torch.norm(Kx + lin_kernel(H2_tilde.t()) - H1_tilde @ torch.diag(L1_tilde) @ H1_tilde.t(), 'fro') ** 2

        x_tilde = psi1(torch.mm(torch.mm(H1_tilde, H1_tilde.t()), op1))
        loss = nn.MSELoss(reduction='sum')
        f2 = 0.5 * loss(x_tilde.view(-1, np.prod(x.shape[1:])), x.view(-1, np.prod(x.shape[1:]))) / x.shape[0]  # Recons_loss

        return 1 * f1 + ae_weight * f2, float(f1.detach().cpu()), float(f2.detach().cpu())


    def log_epoch(train_table, log_dict):
        train_table = pandas.concat([train_table, pandas.DataFrame(log_dict, index=[0])])
        logging.info((train_table.iloc[len(train_table) - 1:len(train_table)]).to_string(header=(t == 0), index=False, justify='right', col_space=15, float_format=utils.float_format, formatters={'mu': lambda x: "%.2f" % x}))
        return train_table

    # Optimization loop
    cost, grad_q, t, train_table, ortos, best_cost = np.inf, np.nan, 0, pandas.DataFrame(), {'orto1': np.inf, 'orto2': np.inf}, np.inf  # Initialize
    train_table = log_epoch(train_table, {'i': t, 'j': float(cost), 'kpca': float(np.inf), 'ae': float(np.inf), 'orto1': float(utils.orto(H1_tilde.t() / torch.linalg.norm(H1_tilde.t(), 2, dim=0))), 'orto2': float(utils.orto(H2_tilde/ torch.linalg.norm(H2_tilde, 2, dim=0))), 'lr': optimizer.param_groups[0]['lr']})
    terminating_condition = terminatingcondition_factory(args_optimizer)
    start = datetime.now()
    while cost > 1e-10 and t < args_optimizer.maxepochs and terminating_condition(cost, rkm2, optimizer):  # run epochs until convergence or cut-off
        loss, f1, f2 = rkm2(xtrain)
        optimizer.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        optimizer.step(lambda: rkm2(xtrain)[0])
        if lr_scheduler is not None:
            lr_scheduler.step(f1)
        t += 1
        cost = float(loss.detach().cpu())
        # Logging
        ortos = {f'orto1': float(utils.orto(H1_tilde / torch.linalg.norm(H1_tilde, 2, dim=0))), f'orto2': float(utils.orto(H2_tilde / torch.linalg.norm(H2_tilde, 2, dim=0)))}
        log_dict = {'i': t, 'j': float(loss.detach().cpu()), 'kpca': f1, 'ae': f2,  'lr': optimizer.param_groups[0]['lr']}
        log_dict = utils.merge_dicts([log_dict, ortos])
        train_table = log_epoch(train_table, log_dict)
    elapsed_time = datetime.now() - start
    logging.info("Training complete in: " + str(elapsed_time))

    return {"train_time": elapsed_time.total_seconds(), 'H2_tilde': H2_tilde.detach().cpu(), 'H1_tilde': H1_tilde.detach().cpu(),
            'L2_tilde': L2_tilde.detach().cpu(), 'L1_tilde': L1_tilde.detach().cpu(), "eigs": [L1_tilde.detach().cpu().numpy(), L2_tilde.detach().cpu().numpy()],
            "phi1": phi1, "psi1": psi1}

def reconstruct_h2(W1, W2, H2, psi1):
    H1 = (W2 @ H2.t()).t()
    x_hat = reconstruct_h1(W1, H1, psi1)
    return x_hat

def reconstruct_h1(W1, H1, psi1):
    x_hat = psi1((W1 @ H1.t()).t())
    return x_hat

def encode_oos(x_test, L1, L2, W1, W2, phi1, ot_train_mean):
    op1 = phi1(x_test)
    op1 = op1 - ot_train_mean
    H1 = torch.inverse(torch.diag(L1) - W2 @ torch.diag(1./L2) @ W2.t()) @ W1.t() @ op1.t()
    h1 = H1.t()
    H2 = torch.diag(1./L2) @ W2.t() @ h1.t()
    h2 = H2.t()
    return h1, h2



class EngineWithGradHook(engine.engine):
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

    def forward(self, x_SNP, c_demo, x_MRI, training_dict, phi1):
        # Encoder
        op1 = phi1(x_SNP)
        H2, L2 = training_dict["H2_tilde"].to(self.device), training_dict["L2_tilde"].to(self.device)
        H1, L1 = training_dict["H1_tilde"].to(self.device), training_dict["L1_tilde"].to(self.device)
        W1, W2 = op1.view(-1, np.prod(op1.shape[1:])).t() @ H1, H1.t() @ H2
        ot_train_mean = torch.mean(op1, dim=0)
        H1_tilde, H2_tilde = encode_oos(x_SNP, L1, L2, W1, W2, phi1, ot_train_mean)
        h_SNP = torch.cat((H1_tilde, H2_tilde), dim=-1)
        h_SNP.register_hook(lambda grad: self.activations_hook('reparameterize', grad))

        # Generator
        x_MRI_fake, a = self.generate(h_SNP, c_demo)
        x_MRI_fake.register_hook(lambda grad: self.activations_hook('generator', grad))
        a.register_hook(lambda grad: self.activations_hook('generator', grad))

        # Diagnostician share
        feature = self.diagnostician_share(x_MRI * a)
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
        
def load_model(label):
    if label is None:
        return None
    model_dir = OUT_DIR.joinpath(label)
    sd_mdl = torch.load(str(model_dir.joinpath("model.pt")), map_location=torch.device('cpu'))
    return {"H1": sd_mdl["H1"], "H2": sd_mdl["H2"], "L1": sd_mdl["L1"], "L2": sd_mdl["L2"],
            "optimizer": sd_mdl["optimizer"]}

@hydra.main(config_path='configs', config_name='config_rkm', version_base=None)
def main(args: DictConfig):
    # device configuration
    my_seed = 42069
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = get_device(0.3, my_seed)

    X_MRI_vis, X_SNP_vis, C_dmg_vis = loaddata.load_data()
    x_SNP = torch.FloatTensor(X_SNP_vis).to(device)
    c_demo = torch.FloatTensor(C_dmg_vis).to(device)   
    x_MRI = torch.FloatTensor(X_MRI_vis).to(device)      
    x_SNP.requires_grad = True

    N_in, N_o = 2000, 2                                 # Look at the number of columns in Y_train
    # Assuming the model has been loaded and set to the correct device
    DKPCA_state_dict = torch.load('./VBM/Result_2024-01-04_18-40-10/DKPCA_01.pt')
    # DKPCA_state_dict = torch.load('./FS/Result_2024-01-04_19-34-06/DKPCA_01.pt')
    nChannels = 1
    cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
    cnn_kwargs = (cnn_kwargs, dict(kernel_size=3, stride=1), 498)
    phi1 = Net1(nChannels, capacity=args.levels.j1.phi.capacity, x_fdim1=args.levels.j1.phi.x_fdim1, x_fdim2=args.levels.j1.phi.x_fdim2, cnn_kwargs=cnn_kwargs).to(device)
    psi1 = Net3(nChannels, capacity=args.levels.j1.phi.capacity, x_fdim1=args.levels.j1.phi.x_fdim1, x_fdim2=args.levels.j1.phi.x_fdim2, cnn_kwargs=cnn_kwargs).to(device)
    levels = [Level(phi1, psi1, args.levels.j1.s), Level(lambda x: x, lambda x: x, args.levels.j2.s)]
    # Train
    training_dict = train_deepkpca(x_SNP, levels, args.optimizer, ae_weight=args.ae_weight, model_to_load=DKPCA_state_dict)

    # engine_state_dict = torch.load('./FS/Result_2024-01-04_19-34-06/3_2024-01-04_19-34-06_engine.pt')
    engine_state_dict = torch.load('./VBM/Result_2024-01-04_18-40-10/2_2024-01-04_18-40-10_engine.pt')
    model = EngineWithGradHook(N_o, device).to(device)
    model.load_state_dict(engine_state_dict)

    
    class_idx = 0           # Select the category index you are interested in, for example 0 AD

    # 确保模型处于评估模式
    model.eval()

    logit_clf = model(x_SNP, c_demo, x_MRI, training_dict, phi1)        # Use GradientTape to calculate the gradient
    print(logit_clf.shape)
    target = torch.mean(logit_clf[:][class_idx])
    # print(target.shape)
    
    model.zero_grad()       # Clear previous gradient

    target.backward()       # Calculate gradient

    # Get the gradient of x_SNP, which represents the partial derivative of 
    # each feature of x_SNP with respect to the encoder output
    x_SNP_grad = x_SNP.grad
    print(x_SNP_grad.shape)

    feature_importance = torch.mean(torch.abs(x_SNP_grad), dim=0)       # Calculate the average feature gradient
    print(feature_importance.shape)

    # Get the values and indexes of the top ten largest features
    top_values, top_indices = torch.topk(feature_importance, 10)

    # Print out the index of these features
    print("Top 10 feature indices:", top_indices.cpu().numpy())

    # Visualize feature importance
    plt.switch_backend('agg')
    plt.bar(range(len(feature_importance)), feature_importance.cpu().numpy())
    plt.title("Feature Importance in x_SNP")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.savefig('feature_path_name.png')
    plt.show()

    feature_importance = x_SNP_grad.detach().cpu().numpy()
    feature_importance = np.abs(feature_importance)
    df_vis = pd.DataFrame({'weights1': feature_importance[0].ravel(), 'weights2': feature_importance[1].ravel(),
                           'weights3': feature_importance[2].ravel(), 'weights4': feature_importance[3].ravel(),
                           'weights5': feature_importance[4].ravel()})
    file_name2 = str(10) + "_weights_vbm_240104.xlsx"
    path_name = os.path.join("./deepkpca-main/", file_name2)
    with pd.ExcelWriter(path_name, engine='openpyxl') as writer:
        df_vis.to_excel(writer, sheet_name='weights_roi', index=False)

if __name__ == '__main__':
    main()
