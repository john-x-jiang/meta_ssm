import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
import numpy as np
from model.modules import *
from data_loader.seq_util import seq_collate_fn, pack_padded_seq
from abc import abstractmethod

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


class RecurrentDynamics(BaseModel):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 obs_filters,
                 obs_dim,
                 init_filters,
                 init_dim,
                 ems_filters,
                 rnn_type,
                 rnn_layers,
                 rnn_bidirection,
                 train_init,
                 reverse_rnn_input=True,
                 sample=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.obs_filters = obs_filters
        self.obs_dim = obs_dim
        self.init_filters = init_filters
        self.init_dim = init_dim
        self.ems_filters = ems_filters
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.rnn_bidirection = rnn_bidirection
        self.train_init = train_init
        self.reverse_rnn_input = reverse_rnn_input
        self.sample = sample

        # domain
        # self.seq_encoder = RnnEncoder(input_dim**2, input_dim**2 // 2,
        #                               n_layer=rnn_layers, drop_rate=0.0,
        #                               bd=rnn_bidirection, nonlin='tanh',
        #                               rnn_type=rnn_type,
        #                               reverse_input=reverse_rnn_input)
        # self.aggregator = Aggregator(input_dim**2 // 2, latent_dim, obs_dim, stochastic=False)
        self.gaussian = Gaussian(latent_dim, latent_dim)
        self.domain_function = LatentStateEncoder(obs_dim, obs_filters, 1, latent_dim, stochastic=False)

        # initialization
        self.initial_function = LatentStateEncoder(init_dim, init_filters, 1, latent_dim)

        # generative model
        self.transition = Transition(latent_dim, latent_dim * 2,
                                     identity_init=True,
                                     domain=True,
                                     stochastic=False)
        self.emission = EmissionDecoder(input_dim, ems_filters, 1, latent_dim)
    
    def latent_initialization(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.init_dim, self.input_dim, self.input_dim)
        z_0, mu_0, var_0 = self.initial_function(x)
        return z_0, mu_0, var_0
    
    def latent_domain(self, x):
        batch_size = x.shape[0]
        # x = x.view(batch_size, self.obs_dim, self.input_dim**2)
        # s_rnn = self.seq_encoder(x)
        # z_c = self.aggregator(s_rnn)
        x = x.view(batch_size, self.obs_dim, self.input_dim, self.input_dim)
        z_c = self.domain_function(x)
        mu, var, z = self.gaussian(z_c)
        return z, mu, var

    def latent_dynamics(self, T, z_0, z_c):
        batch_size = z_0.shape[0]
        z_ = torch.zeros([batch_size, T, self.latent_dim]).to(device)
        z_prev = z_0
        z_[:, 0, :] = z_prev

        for t in range(1, T):
            zt = self.transition(z_prev, z_c)
            z_prev = zt
            z_[:, t, :] = zt
        return z_

    def forward(self, x):
        T = x.size(1)
        batch_size = x.size(0)

        z_c, mu_c, var_c = self.latent_domain(x[:, :self.obs_dim, :])
        z_0, mu_0, var_0 = self.latent_initialization(x[:, :self.init_dim, :])
        z_ = self.latent_dynamics(T, z_0, z_c)
        z_ = z_.view(batch_size * T, -1)
        x_ = self.emission(z_, batch_size, T)

        return x_, mu_0, var_0, mu_c, var_c


class LatentODE(BaseModel):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 num_filters,
                 init_dim,
                 obs_dim,
                 ode_layer,
                 rnn_layers,
                 rnn_bidirection,
                 train_init,
                 sample=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.init_dim = init_dim
        self.obs_dim = obs_dim
        self.ode_layer = ode_layer
        self.rnn_layers = rnn_layers
        self.rnn_bidirection = rnn_bidirection
        self.train_init = train_init
        self.sample = sample

        # domain
        self.aggregator = ODE_RNN(input_dim**2, latent_dim, ode_layer=ode_layer)
        self.gaussian = Gaussian(latent_dim, latent_dim)

        # initialization
        self.initial_function = LatentStateEncoder(init_dim, num_filters, 1, latent_dim)

        # generative model
        self.transition = Transition_ODE(latent_dim, ode_layer=ode_layer,
                                         domain=True,
                                         stochastic=False)
        self.emission = EmissionDecoder(input_dim, num_filters, 1, latent_dim)
    
    def latent_initialization(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.init_dim, self.input_dim, self.input_dim)
        z_0, mu_0, var_0 = self.initial_function(x)
        return z_0, mu_0, var_0

    def latent_domain(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.obs_dim, self.input_dim * self.input_dim)
        z_c = self.aggregator(x[:, :self.obs_dim, :])
        mu, var, z = self.gaussian(z_c)
        return z, mu, var

    def latent_dynamics(self, T, z_0, z_c):
        batch_size = z_0.shape[0]
        z_ = self.transition(T, z_0, z_c)
        return z_

    def forward(self, x):
        T = x.size(1)
        batch_size = x.size(0)

        # condition
        z_0, mu_0, var_0 = self.latent_initialization(x[:, :self.init_dim, :])
        z_c, mu_c, var_c = self.latent_domain(x[:, :self.obs_dim, :])

        # dynamics on target set
        z_ = self.latent_dynamics(T, z_0, z_c)
        z_ = z_.view(batch_size * T, -1)
        x_ = self.emission(z_, batch_size, T)

        return x_, mu_0, var_0, mu_c, var_c


class MetaDynamics(BaseModel):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 num_filters,
                 obs_dim,
                 init_dim,
                 rnn_type,
                 rnn_layers,
                 rnn_bidirection,
                 train_init,
                 reverse_rnn_input=True,
                 sample=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.obs_dim = obs_dim
        self.init_dim = init_dim
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.rnn_bidirection = rnn_bidirection
        self.train_init = train_init
        self.reverse_rnn_input = reverse_rnn_input
        self.sample = sample

        # domain
        # self.seq_encoder = RnnEncoder(input_dim**2, input_dim**2 // 2,
        #                               n_layer=rnn_layers, drop_rate=0.0,
        #                               bd=rnn_bidirection, nonlin='tanh',
        #                               rnn_type=rnn_type,
        #                               reverse_input=reverse_rnn_input)
        # self.aggregator = Aggregator(input_dim**2 // 2, latent_dim, obs_dim, stochastic=False)
        self.gaussian = Gaussian(latent_dim, latent_dim)
        self.domain_function = LatentStateEncoder(obs_dim, num_filters, 1, latent_dim, stochastic=False)

        # initialization
        self.initial_function = LatentStateEncoder(init_dim, num_filters, 1, latent_dim)

        # generative model
        self.transition = Transition(latent_dim, latent_dim * 2,
                                     identity_init=True,
                                     domain=True,
                                     stochastic=False)
        self.emission = EmissionDecoder(input_dim, num_filters, 1, latent_dim)
    
    def latent_initialization(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.init_dim, self.input_dim, self.input_dim)
        z_0, mu_0, var_0 = self.initial_function(x)
        return z_0, mu_0, var_0

    def latent_domain(self, D, K):
        batch_size = D.shape[0]
        D_z_c = []
        for k in range(K):
            x_i = D[:, k, :]
            # x_i = x_i.view(batch_size, self.obs_dim, self.input_dim * self.input_dim)
            # s_rnn = self.seq_encoder(x_i)
            # z_c_i = self.aggregator(s_rnn)
            z_c_i = self.domain_function(x_i)
            D_z_c.append(z_c_i)
        
        z_c = sum(D_z_c) / len(D_z_c)
        mu, var, z = self.gaussian(z_c)
        return z, mu, var

    def latent_dynamics(self, T, z_c, z_0):
        batch_size = z_0.shape[0]
        z_ = torch.zeros([batch_size, T, self.latent_dim]).to(device)
        z_prev = z_0
        z_[:, 0, :] = z_0
        
        for t in range(1, T):
            zt = self.transition(z_prev, z_c)
            z_prev = zt
            z_[:, t, :] = zt
        return z_

    def forward(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)

        # domain condition
        K = D.shape[1]
        x = x.view(batch_size, 1, T, self.input_dim, self.input_dim)
        D_cat = torch.cat([D, x], dim=1)
        z_c, mu_c, var_c = self.latent_domain(D_cat[:, :, :self.obs_dim, :], K + 1)
        x = x.view(batch_size, T, self.input_dim, self.input_dim)
        # initial condition
        z_0, mu_0, var_0 = self.latent_initialization(x[:, :self.init_dim, :])
        z_ = self.latent_dynamics(T, z_0, z_c)
        z_ = z_.view(batch_size * T, -1)
        x_ = self.emission(z_, batch_size, T)

        # Meta-model based dynamics on context set
        D_, D_mu0, D_var0 = [], [], []
        for i in range(K):
            D_z0_i, D_mu0_i, D_var0_i= self.latent_initialization(D[:, i, :self.init_dim, :])
            D_z = self.latent_dynamics(T, D_z0_i, z_c)
            D_z = D_z.view(batch_size * T, -1)
            D_x_i = self.emission(D_z, batch_size, T)

            D_x_i = D_x_i.view(batch_size, -1, T, self.input_dim, self.input_dim)
            D_mu0_i = D_mu0_i.view(batch_size, -1, self.latent_dim)
            D_var0_i = D_var0_i.view(batch_size, -1, self.latent_dim)
            D_.append(D_x_i)
            D_mu0.append(D_mu0_i)
            D_var0.append(D_var0_i)
        D_ = torch.cat(D_, dim=1)
        D_mu0 = torch.cat(D_mu0, dim=1)
        D_var0 = torch.cat(D_var0, dim=1)

        # Regularization on context and target sets
        # x = x.view(batch_size, 1, T, self.input_dim, self.input_dim)
        # D_cat = torch.cat([D, x], dim=1)
        # _, mu_t, var_t = self.latent_domain(D_cat[:, :, :self.obs_dim, :], K + 1)
        mu_t, var_t = torch.zeros_like(mu_c), torch.zeros_like(var_c)

        return x_, D_, mu_c, var_c, mu_t, var_t, mu_0, var_0, D_mu0, D_var0

    def prediction(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)

        # domain condition
        K = D.shape[1]
        D_ss = []
        for i in range(K):
            D_i = D[:, i, :].view(batch_size, T, self.input_dim**2)
            D_si = self.embedding(D_i)
            D_ss.append(D_si)
        
        mu_c, var_c = self.latent_domain(D_ss)
        z_c = self.reparameterization(mu_c, var_c)

        # initial condition
        x = x.view(batch_size, T, self.input_dim**2)
        s_x = self.embedding(x)
        z_0, mu_0, var_0 = self.latent_initialization(s_x[:, :self.init_dim, :])

        # Meta-model-based dynamics on target set
        z_ = self.latent_dynamics(T, z_c, z_0, s_x)
        x_ = self.emission(z_)
        x_ = x_.view(batch_size, T, self.input_dim, self.input_dim)

        return x_


class MetaDynamics_LatentODE(BaseModel):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 emission_dim,
                 init_dim,
                 obs_dim,
                 ode_layer,
                 rnn_layers,
                 rnn_bidirection,
                 train_init,
                 sample=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.emission_dim = emission_dim
        self.init_dim = init_dim
        self.obs_dim = obs_dim
        self.ode_layer = ode_layer
        self.rnn_layers = rnn_layers
        self.rnn_bidirection = rnn_bidirection
        self.train_init = train_init
        self.sample = sample

        self.embedding = nn.Sequential(
            nn.Linear(input_dim**2, 2 * input_dim**2),
            nn.ELU(),
            nn.Linear(2 * input_dim**2, 2 * input_dim**2),
            nn.ELU(),
            nn.Linear(2 * input_dim**2, latent_dim),
            nn.ELU()
        )

        # domain
        self.aggregator = ODE_RNN(latent_dim, ode_layer=ode_layer, stochastic=False)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.var = nn.Linear(latent_dim, latent_dim)
        # initialization
        self.initalization = ODE_RNN(latent_dim, ode_layer=ode_layer)

        # generative model
        self.transition = Transition_ODE(latent_dim, ode_layer=ode_layer,
                                         domain=True,
                                         stochastic=False)
        self.emission = Emission(latent_dim, emission_dim, input_dim**2)

    def reparameterization(self, mu, var):
        if not self.sample:
            return mu
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def latent_initialization(self, y):
        mu_0, var_0 = self.initalization(y)
        z_0 = self.reparameterization(mu_0, var_0)
        return z_0, mu_0, var_0

    def latent_domain(self, D_ss):
        D_z_c = []
        for s in D_ss:
            z_c_i = self.aggregator(s[:, :self.obs_dim, :])
            D_z_c.append(z_c_i)
        
        z_c = sum(D_z_c) / len(D_z_c)
        mu = self.mu(z_c)
        var = self.var(z_c)
        mu = torch.clamp(mu, min=-100, max=85)
        var = torch.clamp(var, min=-100, max=85)
        return mu, var

    def latent_dynamics(self, T, z_0, z_c):
        batch_size = z_0.shape[0]
        z_ = self.transition(T, z_0, z_c)
        return z_

    def forward(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)

        # domain condition
        K = D.shape[1]
        D_ss = []
        for i in range(K):
            D_i = D[:, i, :].view(batch_size, T, self.input_dim**2)
            D_si = self.embedding(D_i)
            D_ss.append(D_si)
        
        mu_c, var_c = self.latent_domain(D_ss)
        z_c = self.reparameterization(mu_c, var_c)

        # initial condition
        x = x.view(batch_size, T, self.input_dim**2)
        s_x = self.embedding(x)
        z_0, mu_0, var_0 = self.latent_initialization(s_x[:, :self.init_dim, :])

        # Meta-model-based dynamics on target set
        z_ = self.latent_dynamics(T, z_0, z_c)
        x_ = self.emission(z_)
        x_ = x_.view(batch_size, T, self.input_dim, self.input_dim)

        # Meta-model based dynamics on context set
        # D_, D_mu0, D_var0 = [], [], []
        # for i in range(K):
        #     D_z0_i, D_mu0_i, D_var0_i= self.latent_initialization(D_ss[i][:, :self.init_dim, :])
        #     D_z = self.latent_dynamics(T, z_c, D_z0_i)
        #     D_x_i = self.emission(D_z)

        #     D_x_i = D_x_i.view(batch_size, -1, T, self.input_dim, self.input_dim)
        #     D_mu0_i = D_mu0_i.view(batch_size, -1, self.latent_dim)
        #     D_var0_i = D_var0_i.view(batch_size, -1, self.latent_dim)
        #     D_.append(D_x_i)
        #     D_mu0.append(D_mu0_i)
        #     D_var0.append(D_var0_i)
        # D_ = torch.cat(D_, dim=1)
        # D_mu0 = torch.cat(D_mu0, dim=1)
        # D_var0 = torch.cat(D_var0, dim=1)
        D_ = torch.zeros_like(D)
        D_mu0 = torch.zeros_like(D)
        D_var0 = torch.zeros_like(D)

        # Regularization on context and target sets
        D_ss.append(s_x)
        mu_t, var_t = self.latent_domain(D_ss)

        return x_, D_, mu_c, var_c, mu_t, var_t, mu_0, var_0, D_mu0, D_var0

    def prediction(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)

        # domain condition
        K = D.shape[1]
        D_ss = []
        for i in range(K):
            D_i = D[:, i, :].view(batch_size, T, self.input_dim**2)
            D_si = self.embedding(D_i)
            D_ss.append(D_si)
        
        mu_c, var_c = self.latent_domain(D_ss)
        z_c = self.reparameterization(mu_c, var_c)

        # initial condition
        x = x.view(batch_size, T, self.input_dim**2)
        s_x = self.embedding(x)
        z_0, mu_0, var_0 = self.latent_initialization(s_x[:, :self.init_dim, :])

        # Meta-model-based dynamics on target set
        z_ = self.latent_dynamics(T, z_c, z_0)
        x_ = self.emission(z_)
        x_ = x_.view(batch_size, T, self.input_dim, self.input_dim)

        return x_
