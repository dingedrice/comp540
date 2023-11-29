import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class GaussianSample(nn.Module):
    def forward(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
'''
    
class GaussianSample(nn.Module):
    def forward(self, mean, var):
        sample = torch.normal(mean, torch.sqrt(var))
        return sample
    
class QYGraph(nn.Module):
    def __init__(self, n_x, k, hidden_dims):
        super(QYGraph, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(n_x, hidden_dims[0]),
            *[
                nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ],
            nn.Linear(hidden_dims[-1], k),
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers[:-1]:
            x = F.relu(layer(x))
        qy_logits = self.fc_layers[-1](x) # remove a ReLU activation layer here compared with the original code
        #qy_logits = F.relu(self.fc_layers[-1](x))
        qy = F.softmax(qy_logits, dim=1)
        return qy_logits, qy

class YTranform(nn.Module):
    def __init__(self, k):
        super(YTranform, self).__init__()
        self.fc_layers = nn.Linear(k, k)

    def forward(self, y):
        yt = self.fc_layers(y)
        return yt

class QZGraph(nn.Module):
    def __init__(self, n_x, n_z, k, hidden_dims):
        super(QZGraph, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(n_x + k, hidden_dims[0]),
            *[
                nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ],
            nn.Linear(hidden_dims[-1], n_z * 2),
        ])
        for i in range(len(self.fc_layers)):
            nn.init.normal_(self.fc_layers[i].weight, 0.1, 0.1)
            nn.init.zeros_(self.fc_layers[i].bias)

    def forward(self, x, yt):
        #y = torch.matmul(qy, torch.eye(qy.size(1)))
        x = x.view(x.size(0), -1)
        yt = yt.view(yt.size(0), -1)
        xy = torch.cat((x, yt), dim=1)
        for layer in self.fc_layers[:-1]:
            xy = F.relu(layer(xy))
        z_params = self.fc_layers[-1](xy)
        zm, zv = torch.split(z_params, z_params.size(1) // 2, dim=1)
        zv = F.softplus(zv) + 1e-5 # positive garantueed
        return zm, zv

class PZGraph(nn.Module):
    def __init__(self, n_z, k, hidden_dims):
        super(PZGraph, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(k, hidden_dims[0]),
            *[
                nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ],
            nn.Linear(hidden_dims[-1], n_z * 2),
        ])

    def forward(self, y):
        #y = torch.matmul(qy, torch.eye(qy.size(1)))
        y = y.view(y.size(0), -1)
        for layer in self.fc_layers[:-1]:
            y = F.relu(layer(y))
        z_params = self.fc_layers[-1](y)
        zm, zv = torch.split(z_params, z_params.size(1) // 2, dim=1)
        zv = F.softplus(zv) + 1e-5 # positive garantueed
        return zm, zv

class PXGraph(nn.Module):
    def __init__(self, n_z, n_x, hidden_dims):
        super(PXGraph, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(n_z, hidden_dims[0]),
            *[
                nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ],
            nn.Linear(hidden_dims[-1], n_x * 2),
        ])

    def forward(self, z):
        for layer in self.fc_layers[:-1]:
            z = F.relu(layer(z))
        x_params = self.fc_layers[-1](z)
        xm, xv = torch.split(x_params, x_params.size(1) // 2, dim=1)
        xv = F.softplus(xv) + 1e-5 # positive garantueed
        return xm, xv
