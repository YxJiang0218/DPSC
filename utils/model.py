import torch
import torch.nn as nn
from utils.metric import thrC, post_proC, err_rate
from sklearn.metrics.cluster import normalized_mutual_info_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np



class SelfExpression(nn.Module):
    def __init__(self, n_samples):
        super(SelfExpression, self).__init__()
        self.cof = nn.Parameter(1.0e-8 * torch.ones(n_samples, n_samples, dtype=torch.float32), requires_grad=True)
        self.n_samples = n_samples

    def forward(self, latent):
        z = latent.reshape(self.n_samples, -1)
        cof = self.cof - torch.diag(torch.diag(self.cof))
        z_c = torch.matmul(cof, z)
        latent_re = z_c.reshape(latent.shape)
        return cof, latent_re


class DPSC():
    def __init__(self,
                 n_all,
                 label,
                 alpha,
                 dim,
                 ro,
                 gamma1,
                 gamma2,
                 device=torch.device('cuda'),
                 weight_decay=0.00,
                 random_seed=41,
                 model_path=None,
                 show_res=10):
        self.device = device
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.alpha = alpha
        self.dim = dim
        self.ro = ro
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.n_all = n_all
        self.model_path = model_path
        self.show_res = show_res
        self.label = label
        self.c_common = torch.nn.Parameter(1.0e-8 * torch.ones(n_all, n_all, dtype=torch.float32).to(self.device),
                                           requires_grad=True)
        self.self_express = SelfExpression(n_all).to(device)

    def train(self, data, epochs, lr=1e-3, ft=False, show_res=20):
        views_data = data
        views_data = views_data.to(self.device)
        optimizer = torch.optim.Adam(self.self_express.parameters(), lr=lr, weight_decay=self.weight_decay)
        loss_values = []
        accs = []
        nmis = []
        epoch_iter = tqdm(range(epochs))
        for epoch in epoch_iter:
            c, re = self.self_express(views_data)
            expr_loss = torch.sum(torch.pow(re - views_data, 2.0))

            reg_loss = torch.sum(torch.pow(c, 2.0))

            loss = self.gamma1 * expr_loss + self.gamma2 * reg_loss

            reg_loss = torch.sum(torch.pow(c, 2.0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (ft & (epoch%show_res==0)):
                Coef = thrC(c.detach().cpu().numpy(), self.alpha)
                y_hat, L = post_proC(Coef, self.label.max(), self.dim, self.ro)
                missrate_x = err_rate(self.label, y_hat)
                acc_x = 1 - missrate_x
                nmi = normalized_mutual_info_score(self.label, y_hat)
                accs.append(acc_x)
                nmis.append(nmi)
                print("epoch: %d" % epoch, "nmi: %.4f" % nmi, "accuracy: %.4f" % acc_x)

            epoch_iter.set_description(f"# Epoch {epoch}, train_loss: {loss.item():.4f}, "
                                       f"self_exp_loss: {expr_loss.item():.4f}, "
                                       f"reg_loss: {reg_loss.item():.4f}")

            loss_values.append(loss.item())


        Coef = thrC(c.detach().cpu().numpy(), self.alpha)
        y_hat, L = post_proC(Coef, self.label.max(), self.dim, self.ro)
        missrate_x = err_rate(self.label, y_hat)
        acc_x = 1 - missrate_x
        nmi = normalized_mutual_info_score(self.label, y_hat)

        print("epoch: %d" % epoch, "nmi: %.4f" % nmi, "accuracy: %.4f" % acc_x)
        return loss_values, nmis, accs