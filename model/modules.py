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


class Gaussian(nn.Module):
    """ Gaussian sample layer with 2 simple linear layers """
    def __init__(self, in_dim, z_dim):
        super().__init__()

        self.mu = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(in_dim // 2, z_dim)
        )

        self.var = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(in_dim // 2, z_dim)
        )
        self.act_var = nn.Tanh()

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        z = mu + (noise * std)
        return z

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)

        # Clamp the variance
        if (mu < -100).any() or (mu > 85).any() or (logvar < -100).any() or (logvar > 85).any():
            mu = torch.clamp(mu, min=-100, max=85)
            logvar = torch.clamp(logvar, min=-100, max=85)
            # print("Explosion in mu/logvar of component")
        logvar = self.act_var(logvar)

        # Reparameterize and sample
        z = self.reparameterization(mu, logvar)
        return mu, logvar, z


"""
Generative modules
"""
class LatentStateEncoder(nn.Module):
    def __init__(self, time_steps, num_filters, num_channels, latent_dim, stochastic=True):
        """
        Holds the convolutional encoder that takes in a sequence of images and outputs the
        initial state of the latent dynamics
        :param time_steps: how many GT steps are used in initialization
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        :param num_channels: how many image color channels there are
        :param latent_dim: dimension of the latent dynamics
        """
        super().__init__()
        self.time_steps = time_steps
        self.num_channels = num_channels
        self.stochastic = stochastic

        # Encoder, q(z_0 | x_{0:time_steps})
        self.initial_encoder = nn.Sequential(
            nn.Conv2d(time_steps * num_channels, num_filters, kernel_size=5, stride=2, padding=(2, 2)),  # 14,14
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=5, stride=2, padding=(2, 2)),  # 7,7
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_filters * 2, num_filters, kernel_size=5, stride=2, padding=(2, 2)),
            nn.Tanh(),
            Flatten(),
        )
        if stochastic:
            self.output = Gaussian(num_filters * 4 ** 2, latent_dim)
        else:
            self.output = nn.Linear(num_filters * 4 ** 2, latent_dim)

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        if self.stochastic:
            z = self.initial_encoder(x[:, :self.time_steps * self.num_channels])
            mu, var, z = self.output(z)
            return mu, var, z
        else:
            z = self.initial_encoder(x[:, :self.time_steps * self.num_channels])
            z = self.output(z)
            return z


class SpatialTemporalBlock(nn.Module):
    def __init__(self, t_in, t_out, n_in, n_out, num_channels, last):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.n_in = n_in
        self.n_out = n_out
        self.num_channels = num_channels
        self.last = last

        self.conv = nn.Conv2d(n_in, n_out, kernel_size=5, stride=2, padding=(2, 2))
        self.bn = nn.BatchNorm2d(n_out)
        self.act = nn.LeakyReLU(0.1)
        self.lin_t = nn.Linear(t_in, t_out)

        if last:
            self.act_last = nn.Tanh()

    def forward(self, x):
        B, _, _, H_in, W_in = x.shape
        x = x.contiguous()
        x = x.view(B * self.t_in, self.n_in, H_in, W_in)
        x = self.act(self.bn(self.conv(x)))
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.view(B, self.t_in, self.n_out, H_out, W_out)
        
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.lin_t(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        if self.last:
            x = self.act_last(x)
            x = x.view(B, -1, H_out, W_out)
        else:
            x = self.act(x)
        return x


class LatentDomainEncoder(nn.Module):
    def __init__(self, time_steps, num_filters, num_channels, latent_dim, stochastic=True):
        """
        Holds the convolutional encoder that takes in a sequence of images and outputs the
        domain of the latent dynamics
        :param time_steps: how many GT steps are used in domain
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        :param num_channels: how many image color channels there are
        :param latent_dim: dimension of the latent dynamics
        """
        super().__init__()
        self.time_steps = time_steps
        self.num_channels = num_channels
        self.stochastic = stochastic

        # Encoder, q(z_0 | x_{0:time_steps})
        self.encoder = nn.Sequential(
            SpatialTemporalBlock(time_steps, time_steps // 2, 1, num_filters, num_channels, False),
            SpatialTemporalBlock(time_steps // 2, time_steps // 4, num_filters, num_filters * 2, num_channels, False),
            SpatialTemporalBlock(time_steps // 4, 1, num_filters * 2, num_filters, num_channels, True),
            Flatten()
        )
        if stochastic:
            self.output = Gaussian(num_filters * 4 ** 2, latent_dim)
        else:
            self.output = nn.Linear(num_filters * 4 ** 2, latent_dim)

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        B, _, H, W = x.shape
        x = x.view(B, self.time_steps, self.num_channels, H, W)
        z = self.encoder(x)
        if self.stochastic:
            mu, var, z = self.output(z)
            return mu, var, z
        else:
            z = self.output(z)
            return z


class SpatialBlock(nn.Module):
    def __init__(self, n_in, n_out, last):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.last = last

        self.conv = nn.Conv2d(n_in, n_out, kernel_size=5, stride=2, padding=(2, 2))
        self.bn = nn.BatchNorm2d(n_out)
        self.act = nn.LeakyReLU(0.1)

        if last:
            self.act_last = nn.Tanh()

    def forward(self, x):
        B, T, _, H_in, W_in = x.shape
        x = x.contiguous()
        x = x.view(B * T, self.n_in, H_in, W_in)
        x = self.act(self.bn(self.conv(x)))
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.view(B, T, self.n_out, H_out, W_out)

        if self.last:
            x = self.act_last(x)
            x = x.view(B, T, -1)
        else:
            x = self.act(x)
        return x


class LatentDomainEncoderDKF(nn.Module):
    def __init__(self, num_filters, num_channels, latent_dim, stochastic=True):
        """
        Holds the convolutional encoder that takes in a sequence of images and outputs the
        domain of the latent dynamics
        :param time_steps: how many GT steps are used in domain
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        :param num_channels: how many image color channels there are
        :param latent_dim: dimension of the latent dynamics
        """
        super().__init__()
        self.num_channels = num_channels
        self.stochastic = stochastic

        # Encoder, q(z_0 | x_{0:time_steps})
        self.encoder = nn.Sequential(
            SpatialBlock(num_channels, num_filters, False),
            SpatialBlock(num_filters, num_filters * 2, False),
            SpatialBlock(num_filters * 2, num_filters, True)
        )
        if stochastic:
            self.output = Gaussian(num_filters * 4 ** 2, latent_dim)
        else:
            self.output = nn.Linear(num_filters * 4 ** 2, latent_dim)

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        B, T, H, W = x.shape
        x = x.view(B, T, self.num_channels, H, W)
        z = self.encoder(x)
        if self.stochastic:
            mu, var, z = self.output(z)
            return mu, var, z
        else:
            z = self.output(z)
            return z


class EmissionDecoder(nn.Module):
    def __init__(self, dim, num_filters, num_channels, latent_dim):
        """
        Holds the convolutional decoder that takes in a batch of individual latent states and
        transforms them into their corresponding data space reconstructions
        """
        super().__init__()
        self.dim = dim
        self.num_channels = num_channels

        # Variable that holds the estimated output for the flattened convolution vector
        self.conv_dim = num_filters * 4 ** 2

        # Emission model handling z_i -> x_i
        self.decoder = nn.Sequential(
            # Transform latent vector into 4D tensor for deconvolution
            nn.Linear(latent_dim, self.conv_dim),
            nn.LeakyReLU(0.1),
            UnFlatten(4),

            # Perform de-conv to output space
            nn.ConvTranspose2d(self.conv_dim // 16, num_filters * 8, kernel_size=4, stride=1, padding=(0, 0)),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=5, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=5, stride=2, padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(num_filters * 2, num_channels, kernel_size=5, stride=1, padding=(2, 2)),
            nn.Sigmoid(),
        )

    def forward(self, zts, batch_size, T):
        """
        Handles decoding a batch of individual latent states into their corresponding data space reconstructions
        :param zts: latent states [BatchSize * GenerationLen, LatentDim]
        :return: data output [BatchSize, GenerationLen, NumChannels, H, W]
        """
        x_rec = self.decoder(zts)

        # Reshape to image output
        x_rec = x_rec.view([batch_size, T, self.num_channels, self.dim, self.dim])
        x_rec = torch.squeeze(x_rec)
        return x_rec


class Transition_Recurrent(nn.Module):
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
            self.lin0 = nn.Linear(z_dim, z_dim)
        
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

    def forward(self, z_t_1, z_c=None):
        if self.domain:
            z_combine = torch.cat((z_t_1, z_c), dim=1)
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
            _mu = self.lin0(z_t_1)
            mu = mu + _mu
        
        if self.stochastic:
            _var = self.lin_v(h_t)
            _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_var(_var)
            return mu, var
        else:
            return mu


class Transition_RGN(nn.Module):
    def __init__(self, latent_dim, transition_dim, num_layers, domain=False):
        """ Latent dynamics function where the state is given and the next state is output """
        super().__init__()
        self.latent_dim = latent_dim
        self.transition_dim = transition_dim
        self.num_layers = num_layers
        self.domain = domain

        # Array that holds dimensions over hidden layers
        if domain:
            self.layers_dim = [latent_dim * 2] + num_layers * [transition_dim] + [latent_dim]
        else:
            self.layers_dim = [latent_dim] + num_layers * [transition_dim] + [latent_dim]

        # Build network layers
        self.acts = []
        self.layers = []
        self.layer_norms = []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act('leaky_relu') if i < num_layers else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out, device=device))
            self.layer_norms.append(nn.LayerNorm(n_out, device=device) if True and i < num_layers else nn.Identity())

    def forward(self, z_t_1, z_c=None):
        """ Given a latent state z, output z+1 """
        if self.domain:
            z = torch.cat((z_t_1, z_c), dim=1)
        else:
            z = z_t_1
        for norm, a, layer in zip(self.layer_norms, self.acts, self.layers):
            z = a(norm(layer(z)))
        return z


class Transition_RGN_res(nn.Module):
    def __init__(self, latent_dim, transition_dim, num_layers, domain=False):
        """ Latent dynamics function where the state is given and the next state is output """
        super().__init__()
        self.latent_dim = latent_dim
        self.transition_dim = transition_dim
        self.num_layers = num_layers
        self.domain = domain

        # Array that holds dimensions over hidden layers
        if domain:
            self.layers_dim = [latent_dim * 2] + num_layers * [transition_dim] + [latent_dim]
            # Combine
            self.combine = nn.Linear(latent_dim * 2, latent_dim)
        else:
            self.layers_dim = [latent_dim] + num_layers * [transition_dim] + [latent_dim]

        # Build network layers
        self.acts = []
        self.layers = []
        self.layer_norms = []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act('leaky_relu') if i < num_layers else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out, device=device))
            self.layer_norms.append(nn.LayerNorm(n_out, device=device) if True and i < num_layers else nn.Identity())

    def forward(self, z_t_1, z_c=None):
        """ Given a latent state z, output z+1 """
        if self.domain:
            z_init = torch.cat((z_t_1, z_c), dim=1)
            z_combine = self.combine(z_init)
        else:
            z_init = z_t_1
            z_combine = z_init
        for lidx, (norm, a, layer) in enumerate(zip(self.layer_norms, self.acts, self.layers)):
            if lidx == 0:
                z = a(norm(layer(z_init)))
            else:
                z = a(norm(layer(z)))

        # Perform residual block
        z = z + z_combine
        return z


class Transition_LSTM(nn.Module):
    def __init__(self, latent_dim, transition_dim, domain=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.transition_dim = transition_dim
        self.domain = domain

        if domain:
            # Combine
            self.combine = nn.Linear(latent_dim * 2, latent_dim)

        # Recurrent dynamics function
        self.dynamics_func = nn.LSTMCell(input_size=latent_dim, hidden_size=transition_dim)
        self.dynamics_out = nn.Linear(transition_dim, latent_dim)  
    
    def forward(self, T, z_0, z_c=None):
        # Evaluate model forward over T to get L latent reconstructions
        # TODO
        t = torch.linspace(1, T, T).to(device)

        if self.domain:
            z = torch.cat((z_0, z_c), dim=1)
            z_init = self.combine(z)
        else:
            z_init = z_0

        # Evaluate forward over timesteps by recurrently passing in output states
        zt = []
        for tidx in t:
            if tidx == 1:
                z_hid, c_hid = self.dynamics_func(z_init)
            else:
                z_hid, c_hid = self.dynamics_func(z, (z_hid, c_hid))

            z = self.dynamics_out(z_hid)
            zt.append(z)

        # Stack zt and decode zts
        zt = torch.stack(zt)
        # TODO: dimension
        zt = zt.contiguous().view([zt.shape[0] * zt.shape[1], -1])
        return zt


class Transition_ODE(nn.Module):
    def __init__(self, latent_dim, transition_dim, ode_layer=2, domain=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.transition_dim = transition_dim
        self.ode_layer = ode_layer
        self.domain = domain

        if domain:
            self.combine = nn.Linear(2 * latent_dim, latent_dim)
            self.layers_dim = [2 * latent_dim] + ode_layer * [transition_dim] + [2 * latent_dim]
        else:
            self.layers_dim = [latent_dim] + ode_layer * [transition_dim] + [latent_dim]
        self.layers = []
        self.acts = []
        self.layer_norms = []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act('leaky_relu') if i < ode_layer else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out, device=device))
            self.layer_norms.append(nn.LayerNorm(n_out, device=device) if True and i < ode_layer else nn.Identity())
    
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
        else:
            z_in = z_0

        zt = odeint(solver, z_in, t, method='rk4', rtol=1e-5, atol=1e-7, options={'step_size': 0.5})
        zt = zt.permute(1, 0, 2).contiguous()
        if self.domain:
            zt = self.combine(zt)
        
        return zt


'''
Turbulence Flow Specific Part
'''

class LatentStateEncoderFlow(nn.Module):
    def __init__(self, time_steps, num_filters, num_channels, latent_dim, stochastic=True):
        """
        Holds the convolutional encoder that takes in a sequence of images and outputs the
        initial state of the latent dynamics
        :param time_steps: how many GT steps are used in initialization
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        :param num_channels: how many image color channels there are
        :param latent_dim: dimension of the latent dynamics
        """
        super().__init__()
        self.time_steps = time_steps
        self.num_channels = num_channels
        self.stochastic = stochastic

        # Encoder, q(z_0 | x_{0:time_steps})
        self.initial_encoder = nn.Sequential(
            nn.Conv2d(time_steps * num_channels, num_filters, kernel_size=5, stride=2, padding=(2, 2)),  # 14,14
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=5, stride=2, padding=(2, 2)),  # 7,7
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=5, stride=2, padding=(2, 2)),  # 7,7
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_filters * 4, num_filters, kernel_size=5, stride=2, padding=(2, 2)),
            nn.Tanh(),
            Flatten(),
        )
        if stochastic:
            self.output = Gaussian(num_filters * 4 ** 2, latent_dim)
        else:
            self.output = nn.Linear(num_filters * 4 ** 2, latent_dim)

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        if self.stochastic:
            z = self.initial_encoder(x[:, :self.time_steps * self.num_channels])
            mu, var, z = self.output(z)
            return mu, var, z
        else:
            z = self.initial_encoder(x[:, :self.time_steps * self.num_channels])
            z = self.output(z)
            return z


class LatentDomainEncoderFlow(nn.Module):
    def __init__(self, time_steps, num_filters, num_channels, latent_dim, stochastic=True):
        """
        Holds the convolutional encoder that takes in a sequence of images and outputs the
        domain of the latent dynamics
        :param time_steps: how many GT steps are used in domain
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        :param num_channels: how many image color channels there are
        :param latent_dim: dimension of the latent dynamics
        """
        super().__init__()
        self.time_steps = time_steps
        self.num_channels = num_channels
        self.stochastic = stochastic

        # Encoder, q(z_0 | x_{0:time_steps})
        self.encoder = nn.Sequential(
            SpatialTemporalBlock(time_steps, time_steps // 2, 1, num_filters, num_channels, False),
            SpatialTemporalBlock(time_steps // 2, time_steps // 4, num_filters, num_filters * 2, num_channels, False),
            SpatialTemporalBlock(time_steps // 4, time_steps // 8, num_filters * 2, num_filters * 4, num_channels, False),
            SpatialTemporalBlock(time_steps // 8, 1, num_filters * 4, num_filters, num_channels, True),
            Flatten()
        )
        if stochastic:
            self.output = Gaussian(num_filters * 4 ** 2, latent_dim)
        else:
            self.output = nn.Linear(num_filters * 4 ** 2, latent_dim)

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        B, _, H, W = x.shape
        x = x.view(B, self.time_steps, self.num_channels, H, W)
        z = self.encoder(x)
        if self.stochastic:
            mu, var, z = self.output(z)
            return mu, var, z
        else:
            z = self.output(z)
            return z


class LatentDomainEncoderDKF_Flow(nn.Module):
    def __init__(self, num_filters, num_channels, latent_dim, stochastic=True):
        """
        Holds the convolutional encoder that takes in a sequence of images and outputs the
        domain of the latent dynamics
        :param time_steps: how many GT steps are used in domain
        :param num_filters: base convolutional filters, upscaled by 2 every layer
        :param num_channels: how many image color channels there are
        :param latent_dim: dimension of the latent dynamics
        """
        super().__init__()
        self.num_channels = num_channels
        self.stochastic = stochastic

        # Encoder, q(z_0 | x_{0:time_steps})
        self.encoder = nn.Sequential(
            SpatialBlock(num_channels, num_filters, False),
            SpatialBlock(num_filters, num_filters * 2, False),
            SpatialBlock(num_filters * 2, num_filters * 4, False),
            SpatialBlock(num_filters * 4, num_filters, True)
        )
        if stochastic:
            self.output = Gaussian(num_filters * 4 ** 2, latent_dim)
        else:
            self.output = nn.Linear(num_filters * 4 ** 2, latent_dim)

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        B, T, H, W = x.shape
        x = x.view(B, T, self.num_channels, H, W)
        z = self.encoder(x)
        if self.stochastic:
            mu, var, z = self.output(z)
            return mu, var, z
        else:
            z = self.output(z)
            return z


class EmissionDecoderFlow(nn.Module):
    def __init__(self, dim, num_filters, num_channels, latent_dim):
        """
        Holds the convolutional decoder that takes in a batch of individual latent states and
        transforms them into their corresponding data space reconstructions
        """
        super().__init__()
        self.dim = dim
        self.num_channels = num_channels

        # Variable that holds the estimated output for the flattened convolution vector
        self.conv_dim = num_filters * 4 ** 2

        # Emission model handling z_i -> x_i
        self.decoder = nn.Sequential(
            # Transform latent vector into 4D tensor for deconvolution
            nn.Linear(latent_dim, self.conv_dim),
            nn.LeakyReLU(0.1),
            UnFlatten(4),

            # Perform de-conv to output space
            nn.ConvTranspose2d(self.conv_dim // 16, num_filters * 8, kernel_size=5, stride=2, padding=(2, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=5, stride=2, padding=(2, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=5, stride=2, padding=(2, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(num_filters * 2, num_channels, kernel_size=5, stride=2, padding=(2, 2), output_padding=(1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, zts, batch_size, T):
        """
        Handles decoding a batch of individual latent states into their corresponding data space reconstructions
        :param zts: latent states [BatchSize * GenerationLen, LatentDim]
        :return: data output [BatchSize, GenerationLen, NumChannels, H, W]
        """
        x_rec = self.decoder(zts)

        # Reshape to image output
        x_rec = x_rec.view([batch_size, T, self.num_channels, self.dim, self.dim])
        x_rec = torch.squeeze(x_rec)
        return x_rec

'''
End Turbulence Flow
'''


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


'''
Not in use
'''

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
    def __init__(self, z_dim, rnn_dim, stochastic=True, domain=False):
        super().__init__()
        self.z_dim = z_dim
        self.rnn_dim = rnn_dim
        self.stochastic = stochastic
        self.domain = domain

        if not self.domain:
            self.lin1 = nn.Linear(z_dim, rnn_dim)
        else:
            self.lin1 = nn.Linear(2 * z_dim, rnn_dim)
        self.act = nn.Tanh()

        self.lin2 = nn.Linear(rnn_dim, z_dim)
        self.lin_v = nn.Linear(rnn_dim, z_dim)

        # self.act_var = nn.Softplus()
        self.act_var = nn.Tanh()

    def init_z_q_0(self, trainable=True):
        return nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable)

    def forward(self, h_rnn, z_t_1=None, rnn_bidirection=False, z_c=None):
        """
        z_t_1: tensor (b, z_dim)
        h_rnn: tensor (b, rnn_dim)
        """
        assert z_t_1 is not None
        if not self.domain:
            h_comb_ = self.act(self.lin1(z_t_1))
        else:
            z_combine = torch.cat((z_t_1, z_c), dim=1)
            h_comb_ = self.act(self.lin1(z_combine))
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


class GRU_unit(nn.Module):
    def __init__(self, input_dim, latent_dim, num_units=100):
        super().__init__()
        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, num_units),
            nn.Tanh(),
            nn.Linear(num_units, latent_dim),
            nn.Sigmoid()
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, num_units),
            nn.Tanh(),
            nn.Linear(num_units, latent_dim),
            nn.Sigmoid()
        )
        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, num_units),
            nn.Tanh(),
            nn.Linear(num_units, latent_dim)
        )
        self.init_network_weights(self.update_gate)
        self.init_network_weights(self.reset_gate)
        self.init_network_weights(self.new_state_net)

    def init_network_weights(self, net, std = 0.1):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, h, x):
        x_cat = torch.cat([x, h], dim=1)
        update_gate = self.update_gate(x_cat)
        reset_gate = self.reset_gate(x_cat)

        concat = torch.cat([h * reset_gate, x], dim=1)
        new_state = self.new_state_net(concat)
        new_x = (1 - update_gate) * new_state + update_gate * h

        return new_x


class ODE_RNN(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_layer=3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ode_layer = ode_layer
        
        # self.rnn = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        self.rnn = GRU_unit(input_dim, latent_dim, num_units=input_dim // 2)
        
        self.layers_dim = [latent_dim] + (ode_layer - 2) * [latent_dim * 2] + [latent_dim]
        self.layers = []
        self.acts = []
        self.layer_norms = []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act('leaky_relu') if i < ode_layer else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out, device=device))
            self.layer_norms.append(nn.LayerNorm(n_out, device=device) if True and i < ode_layer else nn.Identity())
    
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

        z_0 = torch.zeros([B, self.latent_dim]).to(device)
        z_0 = self.rnn(z_0, x_reverse[:, 0, :])
        for i in range(T):
            zt_ = odeint(solver, z_0, t, method='rk4', rtol=1e-5, atol=1e-7, options={'step_size': 0.5})
            zt_ = zt_[-1, :]
            xt = x_reverse[:, i, :]
            zt = self.rnn(zt_, xt)
            z_0 = zt
        
        z_0 = z_0.view(B, -1)
        return z_0
