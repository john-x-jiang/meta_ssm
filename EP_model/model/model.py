import torch
import torch.nn as nn
import numpy as np
from model.modules import *
from abc import abstractmethod
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class MetaDynamics(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_dim,
                 obs_dim,
                 init_dim,
                 rnn_type):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.init_dim = init_dim
        self.rnn_type = rnn_type

        # Domain model
        self.domain = DomainEncoder(obs_dim, num_channel, latent_dim)
        self.mu_c = nn.Linear(latent_dim, latent_dim)
        self.var_c = nn.Linear(latent_dim, latent_dim)

        # initialization
        self.initial = InitialEncoder(init_dim, num_channel, latent_dim)
        self.mu_0 = nn.Linear(latent_dim, latent_dim)
        self.var_0 = nn.Linear(latent_dim, latent_dim)

        # time modeling
        self.propagation = Transition(latent_dim, latent_dim, stochastic=False)

        # decoder
        self.decoder = Decoder(num_channel, latent_dim)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]
        
        self.domain.setup(heart_name, params)
        self.initial.setup(heart_name, params)
        self.decoder.setup(heart_name, params)
    
    def latent_domain(self, D, K, heart_name):
        N, _, V, T = D.shape
        D_z_c = []
        for i in range(K):
            Di = D[:, i, :, :].view(N, V, T)
            z_c_i = self.domain(Di, heart_name)
            D_z_c.append(z_c_i)

        z_c = sum(D_z_c) / len(D_z_c)
        mu_c = self.mu_c(z_c)
        logvar_c = self.var_c(z_c)
        mu_c = torch.clamp(mu_c, min=-100, max=85)
        logvar_c = torch.clamp(logvar_c, min=-100, max=85)
        z = self.reparameterization(mu_c, logvar_c)

        return z, mu_c, logvar_c
    
    def latent_initial(self, x, heart_name):
        x = x[:, :, :self.init_dim]
        z_0 = self.initial(x, heart_name)
        mu_0 = self.mu_0(z_0)
        logvar_0 = self.var_0(z_0)
        mu_0 = torch.clamp(mu_0, min=-100, max=85)
        logvar_0 = torch.clamp(logvar_0, min=-100, max=85)
        z = self.reparameterization(mu_0, logvar_0)
        
        return z, mu_0, logvar_0
    
    def time_modeling(self, T, z_0, z_c):
        N, V, C = z_0.shape

        z_prev = z_0
        z = []
        for i in range(1, T):
            z_t = self.propagation(z_prev, z_c)
            z_prev = z_t
            z_t = z_t.view(1, N, V, C)
            z.append(z_t)
        z = torch.cat(z, dim=0)
        z_0 = z_0.view(1, N, V, C)
        z = torch.cat([z_0, z], dim=0)
        z = z.permute(1, 2, 3, 0).contiguous()

        return z
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y, D_x, D_y, heart_name):
        N, V, T = x.shape

        # q(c | D)
        K = D_x.shape[1]
        z_c, mu_c, logvar_c = self.latent_domain(D_x, K, heart_name)

        # q(z)
        z_0, mu_0, logvar_0 = self.latent_initial(x, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0, z_c)
        x_ = self.decoder(z, heart_name)

        # KL on all data
        x = x.view(N, 1, -1, T)
        D_x_cat = torch.cat([D_x, x], dim=1)
        _, mu_t, logvar_t = self.latent_domain(D_x_cat, K, heart_name)

        return (x_, ), (mu_c, logvar_c, mu_t, logvar_t, mu_0, logvar_0)

    def prediction(self, qry_x, spt_x, D_x, D_y, heart_name):
        N, V, T = qry_x.shape

        # q(c | D)
        K = D_x.shape[1]
        z_c, mu_c, logvar_c = self.latent_domain(D_x, K, heart_name)

        # q(z)
        z_0, mu_0, logvar_0 = self.latent_initial(qry_x, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0, z_c)
        x_ = self.decoder(z, heart_name)

        return (x_, ), (None, None, None, None, None, None)


class MetaDynamics_Instance(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_dim,
                 obs_dim,
                 init_dim,
                 rnn_type):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.init_dim = init_dim
        self.rnn_type = rnn_type

        # Domain model
        self.domain = DomainEncoder(obs_dim, num_channel, latent_dim)

        # initialization
        self.initial = InitialEncoder(init_dim, num_channel, latent_dim)
        self.mu_0 = nn.Linear(latent_dim, latent_dim)
        self.var_0 = nn.Linear(latent_dim, latent_dim)

        # time modeling
        self.propagation = Transition(latent_dim, latent_dim, stochastic=False)

        # decoder
        self.decoder = Decoder(num_channel, latent_dim)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]
        
        self.domain.setup(heart_name, params)
        self.initial.setup(heart_name, params)
        self.decoder.setup(heart_name, params)
    
    def latent_domain(self, x, heart_name):
        N, V, T = x.shape
        z_c = self.domain(x, heart_name)

        return z_c
    
    def latent_initial(self, x, heart_name):
        x = x[:, :, :self.init_dim]
        z_0 = self.initial(x, heart_name)
        mu_0 = self.mu_0(z_0)
        logvar_0 = self.var_0(z_0)
        mu_0 = torch.clamp(mu_0, min=-100, max=85)
        logvar_0 = torch.clamp(logvar_0, min=-100, max=85)
        z = self.reparameterization(mu_0, logvar_0)
        
        return z, mu_0, logvar_0
    
    def time_modeling(self, T, z_0, z_c):
        N, V, C = z_0.shape

        z_prev = z_0
        z = []
        for i in range(1, T):
            z_t = self.propagation(z_prev, z_c)
            z_prev = z_t
            z_t = z_t.view(1, N, V, C)
            z.append(z_t)
        z = torch.cat(z, dim=0)
        z_0 = z_0.view(1, N, V, C)
        z = torch.cat([z_0, z], dim=0)
        z = z.permute(1, 2, 3, 0).contiguous()

        return z
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y, D_x, D_y, heart_name):
        N, V, T = x.shape

        # q(c | x)
        z_c = self.latent_domain(x, heart_name)

        # q(z)
        z_0, mu_0, logvar_0 = self.latent_initial(x, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0, z_c)
        x_ = self.decoder(z, heart_name)

        return (x_, ), (None, None, None, None, mu_0, logvar_0)

    def prediction(self, qry_x, spt_x, D_x, D_y, heart_name):
        N, V, T = qry_x.shape

        # q(c | x)
        z_c = self.latent_domain(spt_x, heart_name)

        # q(z)
        z_0, mu_0, logvar_0 = self.latent_initial(qry_x, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0, z_c)
        x_ = self.decoder(z, heart_name)

        return (x_, ), (None, None, None, None, None, None)


class BaseDynamics(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_dim,
                 init_dim,
                 rnn_type):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.init_dim = init_dim
        self.rnn_type = rnn_type

        # initialization
        self.initial = InitialEncoder(init_dim, num_channel, latent_dim)
        self.mu_0 = nn.Linear(latent_dim, latent_dim)
        self.var_0 = nn.Linear(latent_dim, latent_dim)

        # time modeling
        self.propagation = Transition_NoDomain(latent_dim, latent_dim, identity_init=False, stochastic=False)

        # decoder
        self.decoder = Decoder(num_channel, latent_dim)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]
        
        self.initial.setup(heart_name, params)
        self.decoder.setup(heart_name, params)
    
    def latent_initial(self, x, heart_name):
        x = x[:, :, :self.init_dim]
        z_0 = self.initial(x, heart_name)
        mu_0 = self.mu_0(z_0)
        logvar_0 = self.var_0(z_0)
        mu_0 = torch.clamp(mu_0, min=-100, max=85)
        logvar_0 = torch.clamp(logvar_0, min=-100, max=85)
        z = self.reparameterization(mu_0, logvar_0)
        
        return z, mu_0, logvar_0
    
    def time_modeling(self, T, z_0):
        N, V, C = z_0.shape

        z_prev = z_0
        z = []
        for i in range(1, T):
            z_t = self.propagation(z_prev)
            z_prev = z_t
            z_t = z_t.view(1, N, V, C)
            z.append(z_t)
        z = torch.cat(z, dim=0)
        z_0 = z_0.view(1, N, V, C)
        z = torch.cat([z_0, z], dim=0)
        z = z.permute(1, 2, 3, 0).contiguous()

        return z
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y, heart_name):
        N, V, T = x.shape

        # q(z)
        z_0, mu_0, logvar_0 = self.latent_initial(x, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0)
        x_ = self.decoder(z, heart_name)

        return (x_, ), (None, None, None, None, mu_0, logvar_0)

    def prediction(self, x, y, heart_name):
        N, V, T = x.shape

        # q(z)
        z_0, mu_0, logvar_0 = self.latent_initial(x, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0)
        x_ = self.decoder(z, heart_name)

        return (x_, ), (None, None, None, None, None, None)
