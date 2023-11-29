import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from subgraphs import *

class GMVAE(nn.Module):
    def __init__(self, model_path, k=10, n_x=784, n_z=64, qy_dims=[16, 16], qz_dims=[16, 16], pz_dims=[16, 16], px_dims=[16, 16],
                 r_nent=1.0, lr=0.00001):
        super(GMVAE, self).__init__()
        self.model_path = model_path
        self.k = k
        self.n_x = n_x
        self.n_z = n_z
        self.qy_dims = qy_dims
        self.qz_dims = qz_dims
        self.pz_dims = pz_dims
        self.px_dims = px_dims
        self.r_nent = r_nent
        #self.batch_size = batch_size
        self.lr = lr
        #self.last_epoch = 0
        self.store = torch.empty(1, 4)
        self.build()

    def build(self):
        self.ytransform = YTranform(self.k)
        self.qy_graph = QYGraph(self.n_x, self.k, self.qy_dims)
        self.qz_graph = QZGraph(self.n_x, self.n_z, self.k, self.qz_dims)
        self.pz_graph = PZGraph(self.n_z, self.k, self.pz_dims)
        self.px_graph = PXGraph(self.n_z, self.n_x, self.px_dims)
        self.gaussian_sample = GaussianSample()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        '''
        qy_logits, qy = self.qy_graph(x)
        z = self.gaussian_sample(*self.qz_graph(x, qy))
        x_recon = self.px_graph(z)
        return qy_logits, qy, z, x_recon
        '''
        qy_logit, qy = self.qy_graph(x)

        z, zm, zv, zm_prior, zv_prior, xm, xv = [[None] * self.k for i in range(7)]
        for i in range(self.k):
            y_tmp = torch.zeros(x.size(0), self.k) + torch.eye(self.k)[i]
            yt = self.ytransform(y_tmp)
            
            zm[i], zv[i] = self.qz_graph(x, yt)
            zm_prior[i], zv_prior[i] = self.pz_graph(y_tmp)
            z[i] = self.gaussian_sample(zm[i], zv[i])
            xm[i], xv[i] = self.px_graph(z[i])
            #y[i] = y_tmp
        return qy_logit, qy, z, zm, zv, zm_prior, zv_prior, xm, xv

    '''def labeled_loss(self, x, qy_logits, z):
        qy = torch.softmax(qy_logits, dim=1)
        zm_prior, zv_prior = self.pz_graph(qy)
        xm, xv = self.px_graph(z)
        return -self.log_normal(x, xm, xv) + self.log_normal(z, zm_prior, zv_prior) - torch.log(1 / self.k)
    '''
    def loss(self, x, qy_logits, qy, z, zm, zv, zm_prior, zv_prior, xm, xv):
        loss = 0

        nent = F.cross_entropy(qy_logits, qy, reduction='none')
        loss -= nent*self.r_nent

        loss_yi = torch.zeros(x.size(0), self.k)
        for i in range(self.k):
            loss_yi[:, i] = self.labeled_loss(x, xm[i], xv[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])

        loss += (qy*loss_yi).sum(dim=1)
            #loss += torch.mean(qy[:, i]*(self.labeled_loss(x, xm[i], xv[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i]) + self.r_nent*torch.log(qy[:, i])))

        return loss.mean(), nent.mean()

    def labeled_loss(self, x, xm, xv, z, zm, zv, zm_prior, zv_prior):
        return -self.log_normal(x, xm, xv) + self.log_normal(z, zm, zv) - self.log_normal(z, zm_prior, zv_prior) - np.log(1 / self.k)
    
    def log_normal(self, x, mean, var):
        return -0.5 * torch.sum(np.log(2 * np.pi) + torch.log(var) + (x - mean) ** 2 / var, dim=-1)

    def train_model(self, train_loader, val_loader, epochs):        
        every_n_epochs = 50
        for epoch in range(epochs):
            if epoch % every_n_epochs != 0:
                self.train()
                for x in train_loader:
                    self.optimizer.zero_grad()
                    qy_logit, qy, z, zm, zv, zm_prior, zv_prior, xm, xv= self(x)
                    loss, _ = self.loss(x, qy_logit, qy, z, zm, zv, zm_prior, zv_prior, xm, xv)
                    loss.backward()
                    self.optimizer.step()
            else:
                running_train_loss = 0
                running_train_nent = 0

                self.train()
                for x in train_loader:
                    self.optimizer.zero_grad()
                    qy_logit, qy, z, zm, zv, zm_prior, zv_prior, xm, xv= self(x)
                    loss, nent = self.loss(x, qy_logit, qy, z, zm, zv, zm_prior, zv_prior, xm, xv)
                    loss.backward()
                    self.optimizer.step()

                    running_train_loss += loss
                    running_train_nent += nent
                train_loss = running_train_loss / len(train_loader)
                train_nent = running_train_nent / len(train_loader)

            # Optionally print or store training loss
            if epoch % every_n_epochs == 0:
                running_val_loss = 0
                running_val_nent = 0

                self.eval()
                with torch.no_grad():
                    for x in val_loader:
                        qy_logit, qy, z, zm, zv, zm_prior, zv_prior, xm, xv= self(x)
                        batch_val_loss, batch_val_nent = self.loss(x, qy_logit, qy, z, zm, zv, zm_prior, zv_prior, xm, xv)
                        running_val_loss += batch_val_loss
                        running_val_nent += batch_val_nent

                    val_loss = running_val_loss / len(val_loader)
                    val_nent = running_val_nent / len(val_loader)

                msg = f'{"tr_ent":>10s},{"tr_loss":>10s},{"val_ent":>10s},{"val_loss":>10s},{"epoch":>10s}'
                print(msg)
                msg = f'{train_nent:10.2e},{train_loss:10.2e},{val_nent:10.2e},{val_loss:10.2e},{epoch:10d}'
                print(msg)
                self.store = torch.cat((self.store, torch.tensor([[train_nent, train_loss, val_nent, val_loss]])), dim=0)

        # Store the info
        #self.last_epoch = epoch
        self.store = self.store[1:]
        torch.save(self.store, self.model_path + '/loss.pt')
        torch.save(self.state_dict(), self.model_path + '/model.pt')


    def encode_y(self, data):
        self.eval()
        with torch.no_grad():
            _, qy, _, _, _, _, _, _, _ = self(data)
            ys = qy
        return ys

    def encode_z(self, data):
        _, ys, zs, _, _, _, _, _, _ = self(data)
        z = torch.zeros(zs[0].size())
        for z_i in range(zs[0].size(1)):
            for y_i in range(ys.size(1)):
                z[:, z_i] += zs[y_i][:,z_i]*ys[:,y_i]
        return z
            

    def reconstruct(self, data):
        _, ys, _, _, _, _, _, xs, _ = self(data)
        x = np.zeros(xs[0].size())
        for x_i in range(xs[0].size(1)):
            for y_i in range(ys.size(1)):
                x[:, x_i] += xs[y_i][:,x_i]*ys[:,y_i]
        return x
