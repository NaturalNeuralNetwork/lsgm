# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import torch
import numpy
from header import nonlinear

class dense_network(torch.nn.Module):
    """
    The dense network is a memory-less feed-forward network
    """
    def __init__(self, layer_configurations):
        super().__init__();

        self.layers      = [];
        self.activations = [];

        for i, configuration in enumerate(layer_configurations):
            self.layers.append(torch.nn.Linear(configuration[0], configuration[1]));
            self.layers[-1].weight.data.normal_(0.0,0.1);

            if len(configuration) > 3:
                self.layers[-1].bias.data = torch.from_numpy(configuration[3]).float();
            else:
                self.layers[-1].bias.data.fill_(0.1);

            setattr(self, 'layer' + str(i), self.layers[-1]);

            self.activations.append(nonlinear[configuration[2]]);

    def print_fn(self):
        return [(l.weight.grad, l.bias.grad) for l in self.layers];

    def forward(self, input_tensor):
        """
        Input tensor has shape [batch_size * sequence_length * embedding length]
        """
        ip_sequence = torch.unbind(input_tensor, 1);

        op_sequence = [];

        for ip in ip_sequence:
            current_tensor = ip;

            for (l, a) in zip(self.layers, self.activations):
                current_tensor = a(l(current_tensor));

            op_sequence.append(current_tensor);

        return torch.stack(op_sequence, 1);
