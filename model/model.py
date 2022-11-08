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
                 domain,
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
        self.domain = domain
        self.trans_model = trans_model
        self.trans_args = trans_args

        # domain
        self.domain_function = LatentDomainEncoder(obs_dim, obs_filters, 1, latent_dim, stochastic=False)
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

    def latent_dynamics(self, T, z_0, z_c=None):
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

        if self.domain:
            z_c, mu_c, var_c = self.latent_domain(x[:, :self.obs_dim, :])
        else:
            z_c, mu_c, var_c = None, None, None
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
        self.domain_function = LatentDomainEncoder(obs_dim, obs_filters, 1, latent_dim, stochastic=False)
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
        
        z_c = sum(D_z_c) / len(D_z_c) # TODO save for embedding
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
    
    def prediction_embedding(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)

        # domain condition
        K = D.shape[1]
        z_c, mu_c, var_c = self.latent_domain(D[:, :, :self.obs_dim, :], K)

        return z_c, mu_c, var_c
    
    def prediction_sampling(self, x, z_0, z_center):
        """
        Function that handles predicting from the same initial state z0 when given differing z_c as a result
        of a K-Means algorithm applied to the train set embeddings
        """
        T = x.size(1)
        batch_size = x.size(0)

        z_ = self.latent_dynamics(T, z_0, z_center)
        z_ = z_.view(batch_size * T, -1)
        x_ = self.emission(z_, batch_size, T)
        x_ = x_.view(x.shape)
        return x_

    def prediction_debug(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)

        # domain condition
        K = D.shape[1]
        D_z_c = []
        for k in range(K):
            x_i = D[:, k, :]
            z_c_i = self.domain_function(x_i)
            D_z_c.append(z_c_i)
        
        z_c_ = sum(D_z_c) / len(D_z_c)
        mu, var, z_c = self.gaussian.debug(z_c_)

        z_0, mu_0, var_0 = self.latent_initialization(x[:, :self.init_dim, :])
        xs_ = []
        N, d = z_c.shape
        for i in range(N):
            z_c_i = z_c[i].view(1, d)
            z_ = self.latent_dynamics(T, z_0, z_c_i)
            z_ = z_.view(batch_size * T, -1)
            x_ = self.emission(z_, batch_size, T)
            x_ = x_.view(x.shape)
            xs_.append(x_)
        xs_ = torch.cat(xs_)
        return z_c, xs_


class DetMetaDynamics(BaseModel):
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
        self.domain_function = LatentDomainEncoder(obs_dim, obs_filters, 1, latent_dim, stochastic=False)
        # self.gaussian = Gaussian(latent_dim, latent_dim)

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
        
        z_c = sum(D_z_c) / len(D_z_c) # TODO: plot out D_z_c in TSNE
        return z_c

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
        z_c = self.latent_domain(D[:, :, :self.obs_dim, :], K)
        
        # initial condition
        z_0, mu_0, var_0 = self.latent_initialization(x[:, :self.init_dim, :])
        z_ = self.latent_dynamics(T, z_0, z_c)
        z_ = z_.view(batch_size * T, -1)
        x_ = self.emission(z_, batch_size, T)

        return x_, None, None, None, None, mu_0, var_0

    def prediction(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)

        # domain condition
        K = D.shape[1]
        z_c = self.latent_domain(D[:, :, :self.obs_dim, :], K)
        
        # initial condition
        z_0, mu_0, var_0 = self.latent_initialization(x[:, :self.init_dim, :])
        z_ = self.latent_dynamics(T, z_0, z_c)
        z_ = z_.view(batch_size * T, -1)
        x_ = self.emission(z_, batch_size, T)

        return x_
    
    def prediction_embedding(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)

        # domain condition
        K = D.shape[1]
        z_c = self.latent_domain(D[:, :, :self.obs_dim, :], K)

        return z_c, torch.zeros_like(z_c), torch.zeros_like(z_c)

    def prediction_sampling(self, x, z_0, z_center):
        T = x.size(1)
        batch_size = x.size(0)

        z_ = self.latent_dynamics(T, z_0, z_center)
        z_ = z_.view(batch_size * T, -1)
        x_ = self.emission(z_, batch_size, T)
        x_ = x_.view(x.shape)
        return x_
    
    def repara_debug(self, mu):
        d = mu.shape[1]
        zs = []
        for i in range(d):
            std = torch.zeros_like(mu)
            std[:, i] = 0.1
            # z = mu - torch.exp(std)
            z = mu - std
            zs.append(z)
        zs.append(mu + torch.ones_like(mu))
        for i in range(d):
            std = torch.zeros_like(mu)
            std[:, i] = 0.1
            # z = mu + torch.exp(std)
            z = mu + std
            zs.append(z)
        zs = torch.cat(zs)
        return zs

    def prediction_debug(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)

        # domain condition
        K = D.shape[1]
        D_z_c = []
        for k in range(K):
            x_i = D[:, k, :]
            z_c_i = self.domain_function(x_i)
            D_z_c.append(z_c_i)
        
        z_c_ = sum(D_z_c) / len(D_z_c)
        z_c = self.repara_debug(z_c_)

        z_0, mu_0, var_0 = self.latent_initialization(x[:, :self.init_dim, :])
        xs_ = []
        N, d = z_c.shape
        for i in range(N):
            z_c_i = z_c[i].view(1, d)
            z_ = self.latent_dynamics(T, z_0, z_c_i)
            z_ = z_.view(batch_size * T, -1)
            x_ = self.emission(z_, batch_size, T)
            x_ = x_.view(x.shape)
            xs_.append(x_)
        xs_ = torch.cat(xs_)
        return z_c, xs_


class DKF(BaseModel):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 rnn_dim,
                 obs_dim,
                 rnn_bidirection,
                 rnn_type,
                 rnn_layers,
                 ems_filters,
                 trans_args,
                 estim_args):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.rnn_dim = rnn_dim
        self.obs_dim = obs_dim
        self.rnn_bidirection = rnn_bidirection
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.ems_filters = ems_filters
        self.trans_args = trans_args
        self.estim_args = estim_args

        # observation
        self.embedding = nn.Sequential(
            nn.Linear(input_dim**2, 2 * input_dim**2),
            nn.ReLU(),
            nn.Linear(2 * input_dim**2, 2 * input_dim**2),
            nn.ReLU(),
            nn.Linear(2 * input_dim**2, rnn_dim),
            nn.ReLU()
        )
        self.encoder = RnnEncoder(rnn_dim, rnn_dim,
                                  n_layer=rnn_layers, drop_rate=0.0,
                                  bd=rnn_bidirection, nonlin='relu',
                                  rnn_type=rnn_type,
                                  reverse_input=False)

        # generative model
        self.transition = Transition_Recurrent(**trans_args)
        self.estimation = Correction(**estim_args)

        self.emission = EmissionDecoder(input_dim, ems_filters, 1, latent_dim)

        # initialize hidden states
        self.mu_p_0, self.var_p_0 = self.transition.init_z_0(trainable=False)
        self.z_q_0 = self.estimation.init_z_q_0(trainable=False)
    
    def reparameterization(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def latent_dynamics(self, T, x_rnn):
        batch_size = x_rnn.shape[0]

        if T > self.obs_dim:
            T_final = T
        else:
            T_final = self.obs_dim

        z_ = torch.zeros([batch_size, T_final, self.latent_dim]).to(device)
        mu_ps = torch.zeros([batch_size, self.obs_dim, self.latent_dim]).to(device)
        var_ps = torch.zeros([batch_size, self.obs_dim, self.latent_dim]).to(device)
        mu_qs = torch.zeros([batch_size, self.obs_dim, self.latent_dim]).to(device)
        var_qs = torch.zeros([batch_size, self.obs_dim, self.latent_dim]).to(device)

        z_q_0 = self.z_q_0.expand(batch_size, self.latent_dim)  # q(z_0)
        mu_p_0 = self.mu_p_0.expand(batch_size, 1, self.latent_dim)
        var_p_0 = self.var_p_0.expand(batch_size, 1, self.latent_dim)
        z_prev = z_q_0
        z_[:, 0, :] = z_prev

        for t in range(self.obs_dim):
            # zt = self.transition(z_prev)
            mu_q, var_q = self.estimation(x_rnn[:, t, :], z_prev,
                                          rnn_bidirection=self.rnn_bidirection)
            zt_q = self.reparameterization(mu_q, var_q)
            z_prev = zt_q

            # p(z_{t+1} | z_t)
            mu_p, var_p = self.transition(z_prev)
            zt_p = self.reparameterization(mu_p, var_p)

            z_[:, t, :] = zt_q
            mu_qs[:, t, :] = mu_q
            var_qs[:, t, :] = var_q
            mu_ps[:, t, :] = mu_p
            var_ps[:, t, :] = var_p
        
        if T > self.obs_dim:
            for t in range(self.obs_dim, T):
                # p(z_{t+1} | z_t)
                mu_p, var_p = self.transition(z_prev)
                zt_p = self.reparameterization(mu_p, var_p)
                z_[:, t, :] = zt_p
                z_prev = zt_p
        
        mu_ps = torch.cat([mu_p_0, mu_ps[:, :-1, :]], dim=1)
        var_ps = torch.cat([var_p_0, var_ps[:, :-1, :]], dim=1)

        return z_, mu_qs, var_qs, mu_ps, var_ps

    def forward(self, x):
        T = x.size(1)
        batch_size = x.size(0)

        x = x.view(batch_size, T, -1)
        x = self.embedding(x)
        x_rnn = self.encoder(x)

        z_, mu_qs, var_qs, mu_ps, var_ps = self.latent_dynamics(T, x_rnn)
        
        z_ = z_.view(batch_size * T, -1)
        x_ = self.emission(z_, batch_size, T)

        return x_, mu_qs, var_qs, mu_ps, var_ps


class MetaDKF(BaseModel):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 rnn_dim,
                 obs_dim,
                 obs_filters,
                 rnn_bidirection,
                 rnn_type,
                 rnn_layers,
                 ems_filters,
                 trans_args,
                 estim_args):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.rnn_dim = rnn_dim
        self.obs_dim = obs_dim
        self.obs_filters = obs_filters
        self.rnn_bidirection = rnn_bidirection
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.ems_filters = ems_filters
        self.trans_args = trans_args
        self.estim_args = estim_args

        # domain
        self.domain_function = LatentDomainEncoderDKF(obs_filters, 1, latent_dim, stochastic=False)

        # observation
        self.embedding = nn.Sequential(
            nn.Linear(input_dim**2, 2 * input_dim**2),
            nn.ReLU(),
            nn.Linear(2 * input_dim**2, 2 * input_dim**2),
            nn.ReLU(),
            nn.Linear(2 * input_dim**2, rnn_dim),
            nn.ReLU()
        )
        self.encoder = RnnEncoder(rnn_dim, rnn_dim,
                                  n_layer=rnn_layers, drop_rate=0.0,
                                  bd=rnn_bidirection, nonlin='relu',
                                  rnn_type=rnn_type,
                                  reverse_input=False)

        # generative model
        self.transition = Transition_Recurrent(**trans_args)
        self.estimation = Correction(**estim_args)

        self.emission = EmissionDecoder(input_dim, ems_filters, 1, latent_dim)

        # initialize hidden states
        self.mu_p_0, self.var_p_0 = self.transition.init_z_0(trainable=False)
        self.z_q_0 = self.estimation.init_z_q_0(trainable=False)
    
    def reparameterization(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def latent_domain(self, D, K):
        D_z_c = []
        for k in range(K):
            x_i = D[:, k, :]
            z_c_i = self.domain_function(x_i)
            D_z_c.append(z_c_i)
        
        z_c = sum(D_z_c) / len(D_z_c)
        return z_c

    def latent_dynamics(self, T, x_rnn, z_c):
        batch_size = x_rnn.shape[0]

        if T > self.obs_dim:
            T_final = T
        else:
            T_final = self.obs_dim

        z_ = torch.zeros([batch_size, T_final, self.latent_dim]).to(device)
        mu_ps = torch.zeros([batch_size, self.obs_dim, self.latent_dim]).to(device)
        var_ps = torch.zeros([batch_size, self.obs_dim, self.latent_dim]).to(device)
        mu_qs = torch.zeros([batch_size, self.obs_dim, self.latent_dim]).to(device)
        var_qs = torch.zeros([batch_size, self.obs_dim, self.latent_dim]).to(device)

        z_q_0 = self.z_q_0.expand(batch_size, self.latent_dim)  # q(z_0)
        mu_p_0 = self.mu_p_0.expand(batch_size, 1, self.latent_dim)
        var_p_0 = self.var_p_0.expand(batch_size, 1, self.latent_dim)
        z_prev = z_q_0
        z_[:, 0, :] = z_prev

        for t in range(self.obs_dim):
            # zt = self.transition(z_prev)
            mu_q, var_q = self.estimation(x_rnn[:, t, :], z_prev,
                                          rnn_bidirection=self.rnn_bidirection,
                                          z_c=z_c[:, t])
            zt_q = self.reparameterization(mu_q, var_q)
            z_prev = zt_q

            # p(z_{t+1} | z_t)
            mu_p, var_p = self.transition(z_prev, z_c[:, t])
            zt_p = self.reparameterization(mu_p, var_p)

            z_[:, t, :] = zt_q
            mu_qs[:, t, :] = mu_q
            var_qs[:, t, :] = var_q
            mu_ps[:, t, :] = mu_p
            var_ps[:, t, :] = var_p
        
        if T > self.obs_dim:
            for t in range(self.obs_dim, T):
                # p(z_{t+1} | z_t)
                mu_p, var_p = self.transition(z_prev, z_c[:, t])
                zt_p = self.reparameterization(mu_p, var_p)
                z_[:, t, :] = zt_p
                z_prev = zt_p
        
        mu_ps = torch.cat([mu_p_0, mu_ps[:, :-1, :]], dim=1)
        var_ps = torch.cat([var_p_0, var_ps[:, :-1, :]], dim=1)

        return z_, mu_qs, var_qs, mu_ps, var_ps

    def forward(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)
        
        K = D.shape[1]
        z_c = self.latent_domain(D, K)

        x = x.view(batch_size, T, -1)
        x = x[:, :self.obs_dim, :]
        x = self.embedding(x)
        x_rnn = self.encoder(x)

        z_, mu_qs, var_qs, mu_ps, var_ps = self.latent_dynamics(T, x_rnn, z_c)
        
        z_ = z_.view(batch_size * T, -1)
        x_ = self.emission(z_, batch_size, T)

        return x_, mu_qs, var_qs, mu_ps, var_ps, None, None
    
    def prediction(self, x, D):
        T = x.size(1)
        batch_size = x.size(0)

        K = D.shape[1]
        z_c = self.latent_domain(D, K)

        x = x.view(batch_size, T, -1)
        x = x[:, :self.obs_dim, :]
        x = self.embedding(x)
        x_rnn = self.encoder(x)

        z_, mu_qs, var_qs, mu_ps, var_ps = self.latent_dynamics(T, x_rnn, z_c)
        
        z_ = z_.view(batch_size * T, -1)
        x_ = self.emission(z_, batch_size, T)

        return x_
