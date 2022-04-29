import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.distributions import MultivariateNormal, Normal
from torchdiffeq import odeint
from data_loader.seq_util import reverse_sequence


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Useful modules
"""
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, input):
        nc = input[0].numel() // (self.w ** 2)
        return input.view(input.size(0), nc, self.w, self.w)


"""
Generative modules
"""


class Emission(nn.Module):
    """
    Parameterize the Bernoulli observation likelihood `p(x_t | z_t)`

    Parameters
    ----------
    z_dim: int
        Dim. of latent variables
    emission_dim: int
        Dim. of emission hidden units
    input_dim: int
        Dim. of inputs

    Returns
    -------
        A valid probability that parameterizes the
        Bernoulli distribution `p(x_t | z_t)`
    """
    def __init__(self, z_dim, emission_dim, input_dim, is_sigmoid=True):
        super().__init__()
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.is_sigmoid = is_sigmoid

        self.lin1 = nn.Linear(z_dim, emission_dim)
        self.lin2 = nn.Linear(emission_dim, emission_dim)
        self.lin3 = nn.Linear(emission_dim, input_dim)
        
        self.act = nn.ELU(inplace=True)
        self.out = nn.Sigmoid()

    def forward(self, z_t):
        h1 = self.act(self.lin1(z_t))
        h2 = self.act(self.lin2(h1))
        if self.is_sigmoid:
            return self.out(self.lin3(h2))
        else:
            return self.lin3(h2)


class Transition(nn.Module):
    """
    Parameterize the diagonal Gaussian latent transition probability
    `p(z_t | z_{t-1})`

    Parameters
    ----------
    z_dim: int
        Dim. of latent variables
    transition_dim: int
        Dim. of transition hidden units
    gated: bool
        Use the gated mechanism to consider both linearity and non-linearity
    identity_init: bool
        Initialize the linearity transform as an identity matrix;
        ignored if `gated == False`
    clip: bool
        clip the value for numerical issues

    Returns
    -------
    mu: tensor (b, z_dim)
        Mean that parameterizes the Gaussian
    logvar: tensor (b, z_dim)
        Log-variance that parameterizes the Gaussian
    """
    def __init__(self, z_dim, transition_dim, identity_init=True, domain=False, stochastic=True):
        super().__init__()
        self.z_dim = z_dim
        self.transition_dim = transition_dim
        self.identity_init = identity_init
        self.domain = domain
        self.stochastic = stochastic

        if domain:
            # compute the gain (gate) of non-linearity
            self.lin1 = nn.Linear(z_dim*2, transition_dim*2)
            self.lin2 = nn.Linear(transition_dim*2, z_dim)
            # compute the proposed mean
            self.lin3 = nn.Linear(z_dim*2, transition_dim*2)
            self.lin4 = nn.Linear(transition_dim*2, z_dim)
            # linearity
            self.lin0 = nn.Linear(z_dim*2, z_dim)
        else:
            # compute the gain (gate) of non-linearity
            self.lin1 = nn.Linear(z_dim, transition_dim)
            self.lin2 = nn.Linear(transition_dim, z_dim)
            # compute the proposed mean
            self.lin3 = nn.Linear(z_dim, transition_dim)
            self.lin4 = nn.Linear(transition_dim, z_dim)
        
        # compute the linearity part
        self.lin_n = nn.Linear(z_dim, z_dim)

        if identity_init:
            self.lin_n.weight.data = torch.eye(z_dim)
            self.lin_n.bias.data = torch.zeros(z_dim)

        # compute the variation
        self.lin_v = nn.Linear(z_dim, z_dim)
        # var activation
        # self.act_var = nn.Softplus()
        self.act_var = nn.Tanh()

        self.act_weight = nn.Sigmoid()
        self.act = nn.ELU(inplace=True)

    def init_z_0(self, trainable=True):
        return nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable), \
            nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable)

    def forward(self, z_t_1, z_domain=None):
        if self.domain:
            z_combine = torch.cat((z_t_1, z_domain), dim=1)
            _g_t = self.act(self.lin1(z_combine))
            g_t = self.act_weight(self.lin2(_g_t))
            _h_t = self.act(self.lin3(z_combine))
            h_t = self.act(self.lin4(_h_t))
            _mu = self.lin0(z_combine)
            mu = (1 - g_t) * self.lin_n(_mu) + g_t * h_t
            mu = mu + _mu
        else:
            _g_t = self.act(self.lin1(z_t_1))
            g_t = self.act_weight(self.lin2(_g_t))
            _h_t = self.act(self.lin3(z_t_1))
            h_t = self.act(self.lin4(_h_t))
            mu = (1 - g_t) * self.lin_n(z_t_1) + g_t * h_t
            mu = mu + z_t_1
        
        if self.stochastic:
            _var = self.lin_v(h_t)
            _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_var(_var)
            return mu, var
        else:
            return mu


"""
Inference modules
"""


class Correction(nn.Module):
    """
    Parameterize variational distribution `q(z_t | z_{t-1}, x_{t:T})`
    a diagonal Gaussian distribution

    Parameters
    ----------
    z_dim: int
        Dim. of latent variables
    rnn_dim: int
        Dim. of RNN hidden states
    clip: bool
        clip the value for numerical issues

    Returns
    -------
    mu: tensor (b, z_dim)
        Mean that parameterizes the variational Gaussian distribution
    logvar: tensor (b, z_dim)
        Log-var that parameterizes the variational Gaussian distribution
    """
    def __init__(self, z_dim, rnn_dim, stochastic=True):
        super().__init__()
        self.z_dim = z_dim
        self.rnn_dim = rnn_dim
        self.stochastic = stochastic

        self.lin1 = nn.Linear(z_dim, rnn_dim)
        self.act = nn.Tanh()

        self.lin2 = nn.Linear(rnn_dim, z_dim)
        self.lin_v = nn.Linear(rnn_dim, z_dim)

        # self.act_var = nn.Softplus()
        self.act_var = nn.Tanh()

    def init_z_q_0(self, trainable=True):
        return nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable)

    def forward(self, h_rnn, z_t_1=None, rnn_bidirection=False):
        """
        z_t_1: tensor (b, z_dim)
        h_rnn: tensor (b, rnn_dim)
        """
        assert z_t_1 is not None
        h_comb_ = self.act(self.lin1(z_t_1))
        if rnn_bidirection:
            h_comb = (1.0 / 3) * (h_comb_ + h_rnn[:, :self.rnn_dim] + h_rnn[:, self.rnn_dim:])
        else:
            h_comb = 0.5 * (h_comb_ + h_rnn)
        mu = self.lin2(h_comb)

        if self.stochastic:
            _var = self.lin_v(h_comb)
            _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_var(_var)
            return mu, var
        else:
            return mu


class RnnEncoder(nn.Module):
    """
    RNN encoder that outputs hidden states h_t using x_{t:T}

    Parameters
    ----------
    input_dim: int
        Dim. of inputs
    rnn_dim: int
        Dim. of RNN hidden states
    n_layer: int
        Number of layers of RNN
    drop_rate: float [0.0, 1.0]
        RNN dropout rate between layers
    bd: bool
        Use bi-directional RNN or not

    Returns
    -------
    h_rnn: tensor (b, T_max, rnn_dim * n_direction)
        RNN hidden states at every time-step
    """
    def __init__(self, input_dim, rnn_dim, n_layer=1, drop_rate=0.0, bd=False,
                 nonlin='relu', rnn_type='rnn', orthogonal_init=False,
                 reverse_input=True):
        super().__init__()
        self.n_direction = 1 if not bd else 2
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.n_layer = n_layer
        self.drop_rate = drop_rate
        self.bd = bd
        self.nonlin = nonlin
        self.reverse_input = reverse_input

        if not isinstance(rnn_type, str):
            raise ValueError("`rnn_type` should be type str.")
        self.rnn_type = rnn_type
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim,
                              nonlinearity=nonlin, batch_first=True,
                              bidirectional=bd, num_layers=n_layer,
                              dropout=drop_rate)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=rnn_dim,
                              batch_first=True,
                              bidirectional=bd, num_layers=n_layer,
                              dropout=drop_rate)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=rnn_dim,
                               batch_first=True,
                               bidirectional=bd, num_layers=n_layer,
                               dropout=drop_rate)
        else:
            raise ValueError("`rnn_type` must instead be ['rnn', 'gru', 'lstm'] %s"
                             % rnn_type)

        if orthogonal_init:
            self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def calculate_effect_dim(self):
        return self.rnn_dim * self.n_direction

    def init_hidden(self, trainable=True):
        if self.rnn_type == 'lstm':
            h0 = nn.Parameter(torch.zeros(self.n_layer * self.n_direction, 1, self.rnn_dim), requires_grad=trainable)
            c0 = nn.Parameter(torch.zeros(self.n_layer * self.n_direction, 1, self.rnn_dim), requires_grad=trainable)
            return h0, c0
        else:
            h0 = nn.Parameter(torch.zeros(self.n_layer * self.n_direction, 1, self.rnn_dim), requires_grad=trainable)
            return h0

    def forward(self, x):
        """
        x: pytorch packed object
            input packed data; this can be obtained from
            `util.get_mini_batch()`
        h0: tensor (n_layer * n_direction, b, rnn_dim)
        """
        B, T, _ = x.shape
        seq_lengths = T * torch.ones(B).int().to(device)
        h_rnn, _ = self.rnn(x)
        if self.reverse_input:
            h_rnn = reverse_sequence(h_rnn, seq_lengths)
        return h_rnn


class Aggregator(nn.Module):
    def __init__(self, rnn_dim, z_dim, time_dim, bd=True, init=False, stochastic=True):
        super().__init__()
        self.rnn_dim = rnn_dim
        self.z_dim = z_dim
        self.time_dim = time_dim
        self.bd = bd
        self.init = init
        self.stochastic = stochastic
        
        self.lin1 = nn.Linear(time_dim, 1)
        self.act = nn.ELU(inplace=True)

        if bd:
            self.lin2 = nn.Linear(2 * rnn_dim, z_dim)
        else:
            self.lin2 = nn.Linear(rnn_dim, z_dim)
        
        self.lin_m = nn.Linear(z_dim, z_dim)
        self.lin_v = nn.Linear(z_dim, z_dim)

        if init:
            self.lin_m.weight.data = torch.eye(z_dim)
            self.lin_m.bias.data = torch.zeros(z_dim)
        
        self.act_v = nn.Tanh()

    def forward(self, x):
        B, T, _ = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = self.lin1(x)
        x = torch.squeeze(x)
        
        x = self.lin2(x)
        mu = self.lin_m(x)
        if self.stochastic:
            _var = self.lin_v(x)
            _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_v(_var)
            return mu, var
        else:
            return mu


class ODE_RNN(nn.Module):
    def __init__(self, latent_dim, ode_layer=3, stochastic=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.ode_layer = ode_layer
        self.stochastic = stochastic
        
        self.rnn = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        
        self.layers_dim = [latent_dim] + (ode_layer - 2) * [latent_dim * 2] + [latent_dim]
        self.layers = []
        self.acts = []
        self.layer_norms = []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act('leaky_relu') if i < ode_layer else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out, device=device))
            self.layer_norms.append(nn.LayerNorm(n_out, device=device) if True and i < ode_layer else nn.Identity())

        self.lin_m = nn.Linear(latent_dim, latent_dim)
        if stochastic:
            self.lin_v = nn.Linear(latent_dim, latent_dim)
            # self.lin_m.weight.data = torch.eye(latent_dim)
            # self.lin_m.bias.data = torch.zeros(latent_dim)
            self.act_v = nn.Tanh()
    
    def ode_solver(self, t, x):
        for norm, a, layer in zip(self.layer_norms, self.acts, self.layers):
            x = a(norm(layer(x)))
        return x
    
    def forward(self, x):
        B, T = x.shape[0], x.shape[1]

        t = torch.Tensor([0, 1]).float().to(device)
        solver = lambda t, x: self.ode_solver(t, x)

        seq_length = T * torch.ones(B).int().to(device)
        x_reverse = reverse_sequence(x, seq_length)

        z_0 = torch.zeros_like(x_reverse[:, 0, :])
        z_0 = self.rnn(z_0, x_reverse[:, 0, :])
        for i in range(T):
            zt_ = odeint(solver, z_0, t, method='rk4', rtol=1e-5, atol=1e-7, options={'step_size': 0.5})
            zt_ = zt_[-1, :]
            xt = x_reverse[:, i, :]
            zt = self.rnn(zt_, xt)
            z_0 = zt
        
        z_0 = z_0.view(B, -1)
        mu = self.lin_m(z_0)
        if self.stochastic:
            _var = self.lin_v(z_0)
            _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_v(_var)
            return mu, var
        else:
            return mu


class Transition_ODE(nn.Module):
    def __init__(self, latent_dim, ode_layer=3, domain=False, stochastic=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.ode_layer = 3
        self.domain = domain
        self.stochastic = stochastic

        self.layers_dim = [latent_dim] + (ode_layer - 2) * [latent_dim * 2] + [latent_dim]
        self.layers = []
        self.acts = []
        self.layer_norms = []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act('leaky_relu') if i < ode_layer else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out, device=device))
            self.layer_norms.append(nn.LayerNorm(n_out, device=device) if True and i < ode_layer else nn.Identity())

        if domain:
            self.combine = nn.Linear(2 * latent_dim, latent_dim)
        
        self.lin_m = nn.Linear(latent_dim, latent_dim)
        if stochastic:
            self.lin_v = nn.Linear(latent_dim, latent_dim)
            self.act_v = nn.Tanh()
    
    def ode_solver(self, t, x):
        for norm, a, layer in zip(self.layer_norms, self.acts, self.layers):
            x = a(norm(layer(x)))
        return x
    
    def forward(self, T, z_0, z_c=None):
        B = z_0.shape[0]
        t = torch.linspace(0, T - 1, T).to(device)
        solver = lambda t, x: self.ode_solver(t, x)

        if self.domain:
            z_in = torch.cat([z_0, z_c], dim=1)
            z_in = self.combine(z_in)
        else:
            z_in = z_0

        zt = odeint(solver, z_in, t, method='rk4', rtol=1e-5, atol=1e-7, options={'step_size': 0.5})
        zt = zt.permute(1, 0, 2).contiguous()
        
        mu = self.lin_m(zt)
        if self.stochastic:
            _var = self.lin_v(zt)
            _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_v(_var)
            return mu, var
        else:
            return mu


def get_act(act="relu"):
    """
    Return torch function of a given activation function
    :param act: activation function
    :return: torch object
    """
    if act == "relu":
        return nn.ReLU()
    elif act == "leaky_relu":
        return nn.LeakyReLU()
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":
        return nn.Identity()
    elif act == 'softplus':
        return nn.modules.activation.Softplus()
    elif act == 'softmax':
        return nn.Softmax()
    elif act == "swish":
        return nn.SiLU()
    else:
        return None
