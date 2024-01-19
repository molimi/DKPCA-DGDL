# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: CarpeDiem
@Date: 2023/5/17
@Description: 使用 Pytorch 框架，重构代码
@Improvement:  
"""

import logging
import loaddata
import engine_model
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, confusion_matrix


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


class experiment():
    def __init__(self, fold_idx, task):
        self.fold_idx = fold_idx
        self.task = task
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # device configuration
        my_seed = 42069
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = self.get_device(0.3, my_seed)
        
        # Learning schedules
        self.num_epochs = 200       # 100
        self.num_batches = 5
        self.initial_learning_rate = 1e-4
        self.decay_steps = 1000
        self.decay_rate = 0.96

        # Loss control hyperparameter
        self.alpha_rec = .7     # reconstruction
        self.alpha_gen = .5     # generation
        self.alpha_dis = 1      # discrimination
        self.alpha_clf = 1      # classification
        self.alpha_reg = .7     # regression

    def lr_decay_func(self, epoch):
        return self.initial_learning_rate * (self.decay_rate ** (epoch // self.decay_steps))
    

    def training(self):
        print(f'Start Training, Fold {self.fold_idx}')

        # Load dataset
        X_MRI_train, E_SNP_train, C_demo_train, Y_train, S_train, \
        X_MRI_test, E_SNP_test, C_demo_test, Y_test, S_test = loaddata.load_dataset(self.fold_idx, self.task)
        N_in, N_out = 2000, Y_train.shape[-1]      
        model = engine_model.engine(N_in=N_in, N_o=N_out, device=self.device).to(self.device)

        # Apply gradients
        # var = model.parameters()
        theta_Q = [model.encoder[0].weight, model.encoder[0].bias, model.encoder[2].weight, model.encoder[2].bias]
        theta_P = [model.decoder[0].weight, model.decoder[0].bias, model.decoder[2].weight, model.decoder[2].bias]
        theta_G = [model.generator[0].weight, model.generator[0].bias, model.generator[2].weight, model.generator[2].bias]
        theta_D = [model.discriminator[0].weight, model.discriminator[0].bias]
        theta_C_share = [model.diagnostician_share[0].weight, model.diagnostician_share[0].bias]
        theta_C_clf = [model.diagnostician_clf[0].weight, model.diagnostician_clf[0].bias]
        theta_C_reg = [model.diagnostician_reg[0].weight, model.diagnostician_reg[0].bias]
        
        # Call optimizers
        opt_rec = torch.optim.Adam(theta_Q + theta_P, lr=self.initial_learning_rate)
        opt_gen = torch.optim.Adam(theta_Q + theta_G, lr=self.initial_learning_rate)
        opt_dis = torch.optim.Adam(theta_D, lr=self.initial_learning_rate)
        opt_clf = torch.optim.Adam(theta_G + theta_C_share + theta_C_clf, lr=self.initial_learning_rate)
        opt_reg = torch.optim.Adam(theta_G + theta_C_share + theta_C_reg, lr=self.initial_learning_rate)

        num_iters = int(Y_train.shape[0]/self.num_batches)      # how many batch
        reporter = []

        for epoch in range(self.num_epochs):
            model.train()
            L_rec_per_epoch = 0
            L_gen_per_epoch = 0
            L_dis_per_epoch = 0
            L_clf_per_epoch = 0
            L_reg_per_epoch = 0

            # Randomize the training dataset
            rand_idx = np.random.permutation(Y_train.shape[0])      
            X_MRI_train = X_MRI_train[rand_idx, ...]
            E_SNP_train = E_SNP_train[rand_idx, ...]
            C_demo_train = C_demo_train[rand_idx, ...]
            Y_train = Y_train[rand_idx, ...]
            S_train = S_train[rand_idx, ...]

            for batch in range(num_iters):
                # Sample a minibatch
                xb_MRI = X_MRI_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]
                eb_SNP = E_SNP_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]
                cb_demo = C_demo_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]
                yb_clf = Y_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]
                sb_reg = S_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]        
                
                # convert Tensor
                xb_MRI = torch.FloatTensor(xb_MRI).to(self.device)
                eb_SNP = torch.FloatTensor(eb_SNP).to(self.device)
                cb_demo = torch.FloatTensor(cb_demo).to(self.device)
                # yb_clf = torch.FloatTensor(yb_clf).to(self.device)
                yb_clf = torch.LongTensor(yb_clf).to(self.device)
                sb_reg = torch.FloatTensor(sb_reg).to(self.device)

                # SNP representation module
                mu, log_sigma_square = model.encode(x_SNP=eb_SNP)
                zb_SNP = model.reparameterize(mean=mu, logvar=log_sigma_square)
                eb_SNP_hat_logit = model.decode(z_SNP=zb_SNP)  
                y_pred_si = 1.0/(1+torch.exp(-eb_SNP_hat_logit))
                cross_ent = -eb_SNP*torch.log(y_pred_si) - (1-eb_SNP)*torch.log(1-y_pred_si)
                log_prob_eb_SNP_given_zb_SNP = -torch.sum(cross_ent, dim=1)
                log_prob_zb_SNP = loaddata.log_normal_pdf_1(sample=zb_SNP, mean=0., logvar=0., device=self.device)
                log_q_zb_given_eb_SNP = loaddata.log_normal_pdf_2(sample=zb_SNP, mean=mu, logvar=log_sigma_square, device=self.device)                              
                # Reconstruction loss
                L_rec = -torch.mean(log_prob_eb_SNP_given_zb_SNP + log_prob_zb_SNP - log_q_zb_given_eb_SNP)
                L_rec *= self.alpha_rec
                
                # MRI-SNP association module
                xb_MRI_fake, ab = model.generate(z_SNP=zb_SNP, c_demo=cb_demo)
                real_output = model.discriminate(x_MRI_real_or_fake=xb_MRI)
                fake_output = model.discriminate(x_MRI_real_or_fake=xb_MRI_fake)
                # Least-Square GAN loss
                L_gen = F.mse_loss(fake_output, torch.ones_like(fake_output))
                L_gen *= self.alpha_gen

                L_dis = F.mse_loss(torch.ones_like(real_output), real_output) \
                            + F.mse_loss(torch.zeros_like(fake_output), fake_output)
                L_dis *= self.alpha_dis

                # Diagnostician module
                yb_clf_hat, sb_reg_hat = model.diagnose(x_MRI=xb_MRI, a=ab, apply_logistic_activation=True)

                # Classification loss
                yb_clf_hat = F.log_softmax(yb_clf_hat)
                L_clf = F.nll_loss(yb_clf_hat, yb_clf)
                L_clf *= self.alpha_clf

                # Regression loss
                L_reg = F.mse_loss(sb_reg_hat, sb_reg)      
                L_reg *= self.alpha_reg

                loss = L_rec + L_gen + L_dis + L_clf + L_reg

                # zero gradients
                opt_rec.zero_grad()
                opt_gen.zero_grad()
                opt_dis.zero_grad()
                opt_clf.zero_grad()
                opt_reg.zero_grad()


                # loss.backward()
                L_rec.backward(retain_graph=True)
                L_gen.backward(retain_graph=True)
                L_dis.backward(retain_graph=True)
                L_clf.backward(retain_graph=True)
                L_reg.backward()

                # step
                opt_rec.step()
                opt_gen.step()
                opt_dis.step()
                opt_clf.step()
                opt_reg.step()

                L_rec_per_epoch += L_rec  
                L_gen_per_epoch += L_gen
                L_dis_per_epoch += L_dis
                L_clf_per_epoch += L_clf
                L_reg_per_epoch += L_reg

            L_rec_per_epoch /= num_iters
            L_gen_per_epoch /= num_iters
            L_dis_per_epoch /= num_iters
            L_clf_per_epoch /= num_iters
            L_reg_per_epoch /= num_iters

            # Loss report
            print(f'Epoch: {epoch + 1}, Lrec: {L_rec_per_epoch:>.4f}, Lgen: {L_gen_per_epoch:>.4f}, '
                   f'Ldis: {L_dis_per_epoch:>.4f}, Lclf: {L_clf_per_epoch:>.4f}, Lreg: {L_reg_per_epoch:>.4f}')  


        # convert Tensor
        X_MRI_test = torch.FloatTensor(X_MRI_test).to(self.device)
        E_SNP_test = torch.FloatTensor(E_SNP_test).to(self.device)
        C_demo_test = torch.FloatTensor(C_demo_test).to(self.device)


        # Results
        model.eval()
        with torch.no_grad():
            mu, log_sigma_square = model.encode(E_SNP_test)
            Z_SNP_test = model.reparameterize(mu, log_sigma_square)
            _, A_test = model.generate(Z_SNP_test, C_demo_test)
            Y_test_hat, S_test_hat = model.predict(X_MRI_test, A_test, True)
            Y_test_hat = Y_test_hat.detach().cpu().numpy()
            S_test_hat = S_test_hat.detach().cpu().numpy()

            print(f'Test Accuracy Score: {accuracy_score(Y_test, Y_test_hat):>.4f}')
            print(f'Test AUC: {roc_auc_score(Y_test, Y_test_hat):>.4f}')
            rmse = np.sqrt(mean_squared_error(S_test * 30., S_test_hat * 30.))
            print(f'Test Regression RMSE: {rmse:>.4f}')
            with open("CN_AD_result_231225.txt", "a+", encoding='utf-8') as f:
                f.write("\n**********************************\n")
                f.write(f'Start Training, Fold {self.fold_idx}\n')
                f.write(f'Test AUC: {roc_auc_score(Y_test, Y_test_hat):>.4f}\n')
                f.write(f'Test Accuracy Score: {accuracy_score(Y_test, Y_test_hat):>.4f}\n')
                f.write(f'Test Regression RMSE: {rmse:>.4f}\n')
            return
    

    def get_device(self, memory_rate, my_seed):
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(my_seed)
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.empty(int(total_memory * memory_rate), dtype=torch.int8, device='cuda')
            return 'cuda'
        else:
            return 'cpu'
        

task = ['CN', 'MCI', 'AD']          # ['HC', 'AD'], ['AD', 'MCI'], ['MCI', 'HC'], ['HC', 'MCI', 'AD']
for fold in range(5):               # five-fold cross-validation
    exp = experiment(fold + 1, ['CN', 'AD'])
    exp.training()