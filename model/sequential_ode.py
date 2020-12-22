"""
    Implementation of a sequential ode that can process sequential data.

    @author: jhuthmacher
"""

import torch
from torch import nn

from model.neural_ode import NeuralODE


class SequentialODE(nn.Module):
    """ Sequential ODE implementation.

        It contains of a Sequential Processor that is
        represented by an RNN and a NeuralODE component.
    """

    # pylint: disable=too-many-arguments, dangerous-default-value
    def __init__(self, seq_input_dim: int, node_input_dim: [int] = [32],
                 node_output_dim: [int] = [32], hidden_state_dim: int = 32,
                 seq_output_dim: int = 32):
        """ Initilization of the SequentialODE

            Parameters:
                seq_input_dim: int
                    Input dimension for the SequenceProcessor
                    (RNN, corresponds to num features)
                node_input_dim: [int]
                    Input dimension for the NeuralODE (ODEFunc).
                    This dimension is forwarded to the NN that represents
                    the ODE function.
                    Each entry in the array corresponds to the input
                    dimension of the the layer.
                    IMPORTANT: The ODESolver requires the same input length
                    as the data, therefore a linear layer that ensures this
                    property is appended by design.
                node_output_dim: [int]
                    Contains the dimension for the final linear layers at
                    the end of the NeuralODE (after ODEFunc). This layers
                    are needed to get an output of a desired dimension.
                hidden_state_dim: int
                    Corresponds hidden layers of the RNN.
                    The term hidden relates to the global task 2.
                seq_output_dim: int
                    Dimension of the output of the SequenceProcesser.

        """
        super(SequentialODE, self).__init__()

        self.seq_output_dim = seq_output_dim

        self.sequence_processor = SequenceProcessor(seq_input_dim,
                                                    hidden_state_dim,
                                                    seq_output_dim)

        self.neural_ode = NeuralODE([seq_output_dim] + node_input_dim,
                                    node_output_dim)

    # pylint: disable=arguments-differ
    def forward(self, x_input, time: [float] = None):
        """
            Parameters:
                x: torch.Tensor
                    Input date for the forward pass.
                time: [float]
                    Array with timestamps.
            Return:
                torch.Tensor: Output of the forward pass.
        """
        # initial value for the ODE
        y_0 = self.sequence_processor(x_input)
        output = self.neural_ode(y_0, time)

        return output


class SequenceProcessor(nn.Module):
    """ Component of the Sequential ODE that processes the sequential input
        and ouputs the initial value for the NeuralODE.

        In this case we use a recurrent neural network architecture.
    """

    def __init__(self, input_dim: int, hidden_state_dim: int = 32,
                 output_dim: int = 32):
        """ Initilization of the SequentialProcesser (simple RNN)

            Parameters:
                input_dim: int
                    Dimension of the data (num features)
                hidden_state_dim: int
                    Dimension of the hidden sate.
                output_dim: int
                    Output dimension of the RNN
        """
        super(SequenceProcessor, self).__init__()

        self.hidden_state_dim = hidden_state_dim

        self.recurrent_layer = nn.Linear(input_dim + hidden_state_dim,
                                         hidden_state_dim)
        self.output_layer = nn.Linear(hidden_state_dim, output_dim)

    # pylint: disable=arguments-differ
    def forward(self, x_seq):
        """ Forward method

            Parameters:
                x_seq: torch.Tensor
                    Tensor that contains the seuquences which
                    should be processed.
            Return:
                torch.Tensor: Output of the model forward.
        """
        # pylint: disable=no-member, invalid-name
        h = torch.zeros(self.hidden_state_dim)

        for x_input in x_seq:
            # pylint: disable=no-member
            x_input = torch.cat((x_input, h))
            h = torch.tanh(self.recurrent_layer(x_input))

        out = self.output_layer(h)
        return out
