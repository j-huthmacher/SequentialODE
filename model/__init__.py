"""
    Implementation of a neural ode.

    @paper: https://arxiv.org/abs/1806.07366
    @author: jhuthmacher
"""

import torch
from torch import nn
from torchdiffeq import odeint


class NeuralODE(nn.Module):
    """ NeuralODE model implementation
    """

    def __init__(self, input_dim: [int], output_dim: [int]):
        """ Initilization of the NeuralODE

            Parameters:
            -----------
                input_dim: [int]
                    Array of dimensions for each layer of the ODE network.
                output_dim: [int]
                    Array of dimensions for each layer of the final MLP.
        """
        super(NeuralODE, self).__init__()

        self.ode_layer = ODELayer(ODEFunc(input_dim))

        self.linear_layers = nn.ModuleList()
        self.elu = nn.ELU(inplace=True)

        # old_dim = input_dim[-1]
        # Output have to be the same length as input
        old_dim = input_dim[0]

        for i, layer_dim in enumerate(output_dim):
            if i == 0:
                self.linear_layers += [nn.Linear(old_dim, layer_dim),
                                       nn.ELU(inplace=True)]
            else:
                self.linear_layers += [nn.Linear(old_dim, layer_dim),
                                       nn.ELU(inplace=True)]
            old_dim = layer_dim

        self.linear_model = nn.Sequential(*self.linear_layers)

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor, ts: [float] = None):
        """ Forward pass of the NeuralODE.

            Parameters:
            -----------
                x: torch.Tensor
                    Input data.
                ts: [float]
                    Time steps for evaluation the ODE integral.
                    This array decides how far we do the predictions in the
                    future.
        """
        x = self.ode_layer(x, ts)
        x = self.linear_model(x)

        return x

    @property
    def is_cuda(self):
        """ Returns if the model using cuda
        """
        return next(self.parameters()).is_cuda


class ODEFunc(nn.Module):
    """ Neural network that is used as ODE function.
    """

    def __init__(self, dim: [int]):
        """ Initilization of the ODE-function net.

            Parameter:
            ----------
                dim: [int]
                    Array of dimensions for each layer.
        """
        super(ODEFunc, self).__init__()

        self.linear_layers = nn.ModuleList()
        self.dimensions = dim
        old_dim = dim[0]

        for i, layer_dim in enumerate(dim[1:]):
            if i == 0:
                self.linear_layers += [nn.Linear(old_dim, layer_dim)]
            else:
                self.linear_layers += [nn.Linear(old_dim, layer_dim)]
            old_dim = layer_dim

        # The output must have the same length as the input date
        # for the integration!
        self.linear_layers += [nn.Linear(old_dim, dim[0])]

        self.elu = nn.ELU(inplace=True)
        # Number of Function evaluation. Just for statistics.
        # self.nfe = 0

    # pylint: disable=arguments-differ, unused-argument
    def forward(self, time: torch.Tensor, x: torch.Tensor):
        """ Forward pass of the neural net.

            Parameters:
            -----------
                t: torch.Tensor
                    Tensor of dimension (1,d) that contains the
                    evaluation points.
                    It is not important here, but it is needed from the
                    torchdiffeq library.
                x: torch.Tensor
                    Input data which we want to process

            Return:
            -------
                torch.Tensor: Output of the neural network for the input x.
        """
        # self.nfe += 1

        for linear_layer in self.linear_layers:
            x = linear_layer(x)
            x = self.elu(x)

        return x


class ODELayer(nn.Module):
    """ ODE layer that aggregate the ode calculation.
    """

    def __init__(self, ode_func: nn.Module, rtol: float = 1e-3,
                 atol: float = 1e-3):
        """ Initialization of the ODE layer

            Parameters:
            -----------
                ode_func: nn.Module
                    Neural net that is used ot approximate the derivative
                    that we want to find.
                rtol: float
                    Relative error tolerance
                atol: float
                    Absolute error tolerance
        """
        super(ODELayer, self).__init__()

        self.rtol = rtol
        self.atol = atol

        self.ode_func = ode_func

        # Represents the integration time steps we uses.
        # Default [0,1]: Evaluation for one step in the future
        # pylint: disable=not-callable
        self.integration_time = torch.tensor([0, 1]).float()

    # pylint: disable=arguments-differ
    def forward(self, x, ts: [float] = None):
        """ Forward pass through the ODE layer

            Parameters:
            ----------
                x: torch.Tensor
                    Represents the input data.
                ts: [float]
                    Time steps we want to use to evaluat the integral.
                    It has to be a strictly increasing array with time steps.

            Return:
            -------
                torch.Tensor: Output of the ODE layer for the input x.
        """

        # If we want to do predictions for more than one step in the future.
        # Default configuration contains one step prediction.
        if ts:
            # pylint: disable=not-callable
            self.integration_time = torch.tensor(ts).float()

        self.integration_time = self.integration_time.type_as(x)

        # Integration of the ODE to find the actual function we need!
        out = odeint(self.ode_func, x, self.integration_time,
                     rtol=self.rtol, atol=self.atol)

        return out

    @property
    def nfe(self):
        """ Number of function evaluations used for statistics of the model.
        """
        return self.ode_func.nfe

    @nfe.setter
    def nfe(self, value):
        """ Sett function for the nfe property.
        """
        self.ode_func.nfe = value
