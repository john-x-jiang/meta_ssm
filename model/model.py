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


class BaseDynamics(BaseModel):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 obs_filters,
                 obs_dim,
                 init_filters,
                 init_dim,
                 ems_filters,
                 trans_model,
                 trans_args):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.obs_filters = obs_filters
        self.obs_dim = obs_dim
        self.init_filters = init_filters
        self.init_dim = init_dim
        self.ems_filters = ems_filters
        self.trans_model = trans_model
        self.trans_args = trans_args

        # domain
        self.domain_function = LatentStateEncoder(obs_dim, obs_filters, 1, latent_dim, stochastic=False)
        self.gaussian = Gaussian(latent_dim, latent_dim)

        # initialization
        self.initial_function = LatentStateEncoder(init_dim, init_filters, 1, latent_dim)

        # generative model
        if trans_model == 'recurrent':
            self.transition = Transition_Recurrent(**trans_args)
        elif trans_model == 'RGN':
            self.transition = Transition_RGN(**trans_args)
        elif trans_model == 'RGN_residual':
            self.transition = Transition_RGN_res(**trans_args)
        elif trans_model == 'LSTM':
            self.transition = Transition_LSTM(**trans_args)
        elif trans_model == 'ODE':
            self.transition = Transition_ODE(**trans_args)
        self.emission = EmissionDecoder(input_dim, ems_filters, 1, latent_dim)
    
    def latent_initialization(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.init_dim, self.input_dim, self.input_dim)
        z_0, mu_0, var_0 = self.initial_function(x)
        return z_0, mu_0, var_0
    
    def latent_domain(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.obs_dim, self.input_dim, self.input_dim)
        z_c = self.domain_function(x)
        mu, var, z = self.gaussian(z_c)
        return z, mu, var

    def latent_dynamics(self, T, z_0, z_c):
        batch_size = z_0.shape[0]
        if self.trans_model in ['recurrent', 'RGN', 'RGN_residual']:
            z_ = torch.zeros([batch_size, T, self.latent_dim]).to(device)
            z_prev = z_0
            z_[:, 0, :] = z_prev

            for t in range(1, T):
                zt = self.transition(z_prev, z_c)
                z_prev = zt
                z_[:, t, :] = zt
        elif self.trans_model in ['ODE', 'LSTM']:
            z_ = self.transition(T, z_0, z_c)
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


class MetaDynamics(BaseModel):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 obs_filters,
                 obs_dim,
                 init_filters,
                 init_dim,
                 ems_filters,
                 trans_model,
                 trans_args):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.obs_filters = obs_filters
        self.obs_dim = obs_dim
        self.init_filters = init_filters
        self.init_dim = init_dim
        self.ems_filters = ems_filters
        self.trans_model = trans_model
        self.trans_args = trans_args

        # domain
        self.domain_function = LatentStateEncoder(obs_dim, obs_filters, 1, latent_dim, stochastic=False)
        self.gaussian = Gaussian(latent_dim, latent_dim)

        # initialization
        self.initial_function = LatentStateEncoder(init_dim, init_filters, 1, latent_dim)

        # generative model
        if trans_model == 'recurrent':
            self.transition = Transition_Recurrent(**trans_args)
        elif trans_model == 'RGN':
            self.transition = Transition_RGN(**trans_args)
        elif trans_model == 'RGN_residual':
            self.transition = Transition_RGN_res(**trans_args)
        elif trans_model == 'LSTM':
            self.transition = Transition_LSTM(**trans_args)
        elif trans_model == 'ODE':
            self.transition = Transition_ODE(**trans_args)
        self.emission = EmissionDecoder(input_dim, ems_filters, 1, latent_dim)
    
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
            z_c_i = self.domain_function(x_i)
            D_z_c.append(z_c_i)
        
        z_c = sum(D_z_c) / len(D_z_c)
        mu, var, z = self.gaussian(z_c)
        return z, mu, var

    def latent_dynamics(self, T, z_0, z_c):
        batch_size = z_0.shape[0]
        if self.trans_model in ['recurrent', 'RGN', 'RGN_residual']:
            z_ = torch.zeros([batch_size, T, self.latent_dim]).to(device)
            z_prev = z_0
            z_[:, 0, :] = z_prev

            for t in range(1, T):
                zt = self.transition(z_prev, z_c)
                z_prev = zt
                z_[:, t, :] = zt
        elif self.trans_model in ['ODE', 'LSTM']:
            z_ = self.transition(T, z_0, z_c)
        return z_

    def forward(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)

        # domain condition
        K = D.shape[1]
        z_c, mu_c, var_c = self.latent_domain(D[:, :, :self.obs_dim, :], K)
        
        # initial condition
        z_0, mu_0, var_0 = self.latent_initialization(x[:, :self.init_dim, :])
        z_ = self.latent_dynamics(T, z_0, z_c)
        z_ = z_.view(batch_size * T, -1)
        x_ = self.emission(z_, batch_size, T)

        # Regularization on context and target sets
        x = x.view(batch_size, 1, T, self.input_dim, self.input_dim)
        D_cat = torch.cat([D, x], dim=1)
        _, mu_t, var_t = self.latent_domain(D_cat[:, :, :self.obs_dim, :], K + 1)

        return x_, mu_c, var_c, mu_t, var_t, mu_0, var_0

    def prediction(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)

        # domain condition
        K = D.shape[1]
        z_c, mu_c, var_c = self.latent_domain(D[:, :, :self.obs_dim, :], K)
        
        # initial condition
        z_0, mu_0, var_0 = self.latent_initialization(x[:, :self.init_dim, :])
        z_ = self.latent_dynamics(T, z_0, z_c)
        z_ = z_.view(batch_size * T, -1)
        x_ = self.emission(z_, batch_size, T)

        return x_


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
