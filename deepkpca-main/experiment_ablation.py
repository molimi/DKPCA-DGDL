# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: CarpeDiem
@Date: 2023/11/01
@Description: 使用 Pytorch 框架，重构代码
@Improvement:  
"""
import logging
import loaddata
import engine
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, confusion_matrix, mean_absolute_error
from datetime import datetime
from omegaconf import DictConfig
from encoder_decoder import Net1, Net3
import utils

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
        self.num_epochs = 200       # 200
        self.num_batches = 5
        self.initial_learning_rate = 1e-4
  
        # Loss control hyperparameter
        self.alpha_rec = .5    # reconstruction 0.7
        self.alpha_gen = .5     # generation
        self.alpha_dis = 1      # discrimination
        self.alpha_clf = 1      # classification
        self.alpha_reg = .7     # regression


    def training(self, levels, args_optimizer, ae_weight=None):
        print(f'Start Training, Fold {self.fold_idx}')
        in_start = 111500       # 挑选基因的位置
        in_end = 113500
        # Load dataset
        X_MRI_train, E_SNP_train, C_demo_train, Y_train, S_train, \
        X_MRI_test, E_SNP_test, C_demo_test, Y_test, S_test = loaddata.load_dataset(self.fold_idx, self.task, in_start, in_end)
        # N_in, N_out = 2000, Y_test.max() + 1           # 看Y_train的列数 [0, 1, 0, 1]
        N_in, N_out = 2000, Y_train.shape[-1]        # 看Y_train的列数 [[0, 1], [1, 0]]
        
        model = engine.engine(N_o=N_out, device=self.device).to(self.device)

        if ae_weight is None:
            ae_weight = 10.0
        from kernels import LinearKernel

        N = self.num_batches
        s = [level.s for level in levels]

        H2_tilde = torch.randn((N, s[1]), device=self.device)
        H1_tilde = torch.randn((N, s[0]), device=self.device)
        if args_optimizer.name == "geoopt":
            with torch.no_grad():
                H1_tilde = geoopt.ManifoldParameter(H1_tilde, manifold=geoopt.Stiefel(), requires_grad=True).proj_()
                H2_tilde = geoopt.ManifoldParameter(H2_tilde, manifold=geoopt.Stiefel(), requires_grad=True).proj_()
        L1_tilde = torch.randn((s[0],), device=self.device, requires_grad=True)
        L2_tilde = torch.randn((s[1],), device=self.device, requires_grad=True)
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

        # Apply gradients
        theta_G = [model.generator[0].weight, model.generator[0].bias, model.generator[2].weight, model.generator[2].bias]
        theta_D = [model.discriminator[0].weight, model.discriminator[0].bias]
        theta_C_share = [model.diagnostician_share[0].weight, model.diagnostician_share[0].bias]
        theta_C_clf = [model.diagnostician_clf[0].weight, model.diagnostician_clf[0].bias]
        theta_C_reg = [model.diagnostician_reg[0].weight, model.diagnostician_reg[0].bias]
        # Call optimizers
        opt_gen = torch.optim.Adam(theta_G, lr=self.initial_learning_rate)
        opt_dis = torch.optim.Adam(theta_D, lr=self.initial_learning_rate)
        opt_clf = torch.optim.Adam(theta_G + theta_C_share + theta_C_clf, lr=self.initial_learning_rate)
        opt_reg = torch.optim.Adam(theta_G + theta_C_share + theta_C_reg, lr=self.initial_learning_rate)
        num_iters = int(Y_train.shape[0]/self.num_batches)      # how many batch
        reporter = []
        epoch = 0

        # for epoch in range(self.num_epochs):
        while cost > 1e-10 and t < args_optimizer.maxepochs and terminating_condition(cost, rkm2, optimizer) and epoch < self.num_epochs:  # run epochs until convergence or cut-off
            model.train()
            L_rec_per_epoch = 0
            L_gen_per_epoch = 0
            L_dis_per_epoch = 0
            L_clf_per_epoch = 0
            L_reg_per_epoch = 0

            # Randomize the training dataset
            rand_idx = np.random.permutation(Y_train.shape[0])      # Randomly sort the columns
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
                yb_clf = torch.FloatTensor(yb_clf).to(self.device)
                # yb_clf = torch.LongTensor(yb_clf).to(self.device)
                sb_reg = torch.FloatTensor(sb_reg).to(self.device)

                # SNP representation module
                L_rec, f1, f2 = rkm2(eb_SNP)
                L_rec *= self.alpha_rec
                
                # MRI-SNP association module
                # xb_MRI_fake, ab = model.generate(z_SNP=torch.cat((H1_tilde, H2_tilde), dim=-1), c_demo=cb_demo)
                xb_MRI_fake, ab = model.generate(z_SNP=torch.cat((H1_tilde, H2_tilde), dim=-1))     # Do not embed demographic information
                real_output = model.discriminate(x_MRI_real_or_fake=xb_MRI)
                fake_output = model.discriminate(x_MRI_real_or_fake=xb_MRI_fake)
                # Least-Square GAN loss
                L_gen = F.mse_loss(torch.ones_like(fake_output), fake_output)
                L_gen *= self.alpha_gen

                L_dis = 0
                # L_dis = F.mse_loss(torch.ones_like(real_output), real_output) \
                #             + F.mse_loss(torch.zeros_like(fake_output), fake_output)
                # L_dis *= self.alpha_dis

                # Diagnostician module
                yb_clf_hat, sb_reg_hat = model.diagnose(x_MRI=xb_MRI, a=ab, apply_logistic_activation=True)

                # Classification loss
                loss_fn = SigmoidFocalCrossEntropyLoss()
                L_clf = loss_fn(yb_clf, yb_clf_hat)
                L_clf *= self.alpha_clf

                # Regression loss
                L_reg = F.mse_loss(sb_reg, torch.squeeze(sb_reg_hat))      
                L_reg *= self.alpha_reg

                loss = L_rec + L_gen + L_dis + L_clf + L_reg

                # zero gradient
                optimizer.zero_grad()
                optimizer2.zero_grad()
                opt_gen.zero_grad()
                opt_dis.zero_grad()
                opt_clf.zero_grad()
                opt_reg.zero_grad()

                # calculate gradient
                L_rec.backward(retain_graph=True)
                L_gen.backward(retain_graph=True)
                L_dis.backward(retain_graph=True)
                L_clf.backward(retain_graph=True)
                L_reg.backward()

                # update parameters
                optimizer2.step()
                optimizer.step(lambda: rkm2(eb_SNP)[0])
                if lr_scheduler is not None:
                    lr_scheduler.step(f1)

                opt_gen.step()
                opt_dis.step()
                opt_clf.step()
                opt_reg.step()
                # calculate the average of the loss function
                L_rec_per_epoch += L_rec
                L_gen_per_epoch += L_gen
                L_dis_per_epoch += L_dis
                L_clf_per_epoch += L_clf
                L_reg_per_epoch += L_reg

            # calculate the average loss after each epoch
            L_rec_per_epoch /= num_iters
            L_gen_per_epoch /= num_iters
            L_dis_per_epoch /= num_iters
            L_clf_per_epoch /= num_iters
            L_reg_per_epoch /= num_iters
            # Loss report
            print(f'Epoch: {epoch + 1}, Lrec: {L_rec_per_epoch:>.4f}, Lgen: {L_gen_per_epoch:>.4f}, '
                    f'Ldis: {L_dis_per_epoch:>.4f}, Lclf: {L_clf_per_epoch:>.4f}, Lreg: {L_reg_per_epoch:>.4f}')
            epoch += 1
            t += 1
            cost = float(L_rec.detach().cpu())
            # Logging
            ortos = {f'orto1': float(utils.orto(H1_tilde / torch.linalg.norm(H1_tilde, 2, dim=0))), f'orto2': float(utils.orto(H2_tilde / torch.linalg.norm(H2_tilde, 2, dim=0)))}
            log_dict = {'i': t, 'j': float(loss.detach().cpu()), 'kpca': f1, 'ae': f2,  'lr': optimizer.param_groups[0]['lr']}
            log_dict = utils.merge_dicts([log_dict, ortos])
            train_table = log_epoch(train_table, log_dict)

            with open("MCI_HC_loss_231228.txt", "a+", encoding='utf-8') as f:
                f.write(f'Epoch: {epoch + 1}, Lrec: {L_rec_per_epoch:>.4f}, Lgen: {L_gen_per_epoch:>.4f}, Ldis: {L_dis_per_epoch:>.4f}, Lclf: {L_clf_per_epoch:>.4f}, Lreg: {L_reg_per_epoch:>.4f}\n')  

            
        elapsed_time = datetime.now() - start
        logging.info("Training complete in: " + str(elapsed_time))

        training_dict = {"train_time": elapsed_time.total_seconds(), 'H2_tilde': H2_tilde.detach().cpu(), 'H1_tilde': H1_tilde.detach().cpu(),
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


        # save model
        # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # folder_name = f"Result_{current_time}"
        # folder_path = os.path.join(os.getcwd(), folder_name)
        # os.makedirs(folder_path, exist_ok=True)
        # file_name = str(self.fold_idx) + '_' + current_time + '_engine.pt'
        # file_path = os.path.join(folder_path, file_name)
        # torch.save(model, file_path)        

        # convert Tensor
        X_MRI_test = torch.FloatTensor(X_MRI_test).to(self.device)
        E_SNP_test = torch.FloatTensor(E_SNP_test).to(self.device)
        C_demo_test = torch.FloatTensor(C_demo_test).to(self.device)
        N = X_MRI_test.shape[0]
        op1 = phi1(eb_SNP)
        H2, L2 = training_dict["H2_tilde"].to(self.device), training_dict["L2_tilde"].to(self.device)
        H1, L1 = training_dict["H1_tilde"].to(self.device), training_dict["L1_tilde"].to(self.device)
        W1, W2 = op1.view(-1, np.prod(op1.shape[1:])).t() @ H1, H1.t() @ H2
        ot_train_mean = torch.mean(op1, dim=0)
        # Results
        model.eval()
        with torch.no_grad():
            H1_tilde, H2_tilde = encode_oos(E_SNP_test, L1, L2, W1, W2, phi1, ot_train_mean)
            Z_SNP_test = torch.cat((H1_tilde, H2_tilde), dim=-1)
            _, A_test = model.generate(Z_SNP_test, C_demo_test)
            Y_test_hat, S_test_hat = model.diagnose(X_MRI_test, A_test, True)
            # Y_test_hat, S_test_hat = model.predict(X_MRI_test, A_test, True)
            Y_test_hat = Y_test_hat.detach().cpu().numpy()
            S_test_hat = S_test_hat.detach().cpu().numpy()
            # print('预测值：', Y_test_hat)
            # print('真实值：', Y_test)
            print(f'Test AUC: {roc_auc_score(Y_test, Y_test_hat):>.4f}')
            # convert predicted probabilities to class labels
            y_pred_labels = np.argmax(Y_test_hat, axis=1)       # 将 0， 1， 转换为 1
            y_true_labels = np.argmax(Y_test, axis=1)
            # print('预测值：', y_pred_labels)
            # print('真实值：', y_true_labels)
            print(f'Test Accuracy Score: {accuracy_score(y_true_labels, y_pred_labels):>.4f}')
            # print(f'Test Accuracy Score: {accuracy_score(Y_test, Y_test_hat):>.4f}')
            cm = confusion_matrix(y_true_labels, y_pred_labels)
            bca = np.mean([cm[i][i]/sum(cm[i]) for i in range(len(cm))])
            print(f'Test bca: {bca:>.4f}')
            rmse = np.sqrt(mean_squared_error(S_test * 30., S_test_hat * 30.))
            mae = mean_absolute_error(S_test * 30., S_test_hat * 30.)
            print(f'Test Regression RMSE: {rmse:>.4f}')
            file_name = "./Case6/HC&MCI/" + 'VBM_HC_MCI_result_231228.txt'
            # file_path = os.path.join(folder_path, file_name)
            with open(file_name, "a+", encoding='utf-8') as f:
                f.write("\n**********************************\n")
                f.write(f'Start Training, Fold {self.fold_idx}\n')
                f.write(f'Test AUC: {roc_auc_score(Y_test, Y_test_hat):>.4f}\n')
                f.write(f'Test Accuracy Score: {accuracy_score(y_true_labels, y_pred_labels):>.4f}\n')
                # f.write(f'Test Accuracy Score: {accuracy_score(Y_test, Y_test_hat):>.4f}\n')
                f.write(f'Test bca: {bca:>.4f}\n')
                f.write(f'Test Regression RMSE: {rmse:>.4f}\n')
                f.write(f'Test Regression MAE: {mae:>.4f}\n')
            return
    

    def get_device(self, memory_rate, my_seed):
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(my_seed)
            torch.cuda.set_device(2)
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.empty(int(total_memory * memory_rate), dtype=torch.int8, device='cuda')
            return 'cuda'
        else:
            return 'cpu'
        

@hydra.main(config_path='configs', config_name='config_rkm', version_base=None)
def main(args: DictConfig):
    # task = ['AD', 'MCI', 'HC']        # ['HC', 'AD'], ['AD', 'MCI'], ['HC', 'MCI'], ['AD', 'MCI', 'HC']
    for fold in range(5):               # five-fold cross-validation
        exp = experiment(fold + 1, ['MCI', 'HC'])
        nChannels = 1
        cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
        cnn_kwargs = (cnn_kwargs, dict(kernel_size=3, stride=1), 498)
        phi1 = Net1(nChannels, capacity=args.levels.j1.phi.capacity, x_fdim1=args.levels.j1.phi.x_fdim1, x_fdim2=args.levels.j1.phi.x_fdim2, cnn_kwargs=cnn_kwargs).to(device)
        psi1 = Net3(nChannels, capacity=args.levels.j1.phi.capacity, x_fdim1=args.levels.j1.phi.x_fdim1, x_fdim2=args.levels.j1.phi.x_fdim2, cnn_kwargs=cnn_kwargs).to(device)
        levels = [Level(phi1, psi1, args.levels.j1.s), Level(lambda x: x, lambda x: x, args.levels.j2.s)]
        # Train
        exp.training(levels, args.optimizer, ae_weight=args.ae_weight)

if __name__ == '__main__':
    main()