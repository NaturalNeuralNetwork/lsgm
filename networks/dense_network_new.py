# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import torch
import numpy
from header import nonlinear

# Sept 30 2018
# nonlinear   = {'sigmoid':sigmoid, 'ReLu':relu, 'tanh':tanh, 'softmax':lambda x : softmax(x, dim=1), 'noop':lambda x : x};
class NoOp(torch.nn.Module):
    def __init__(self):
        super().__init__();

    def forward(self, tensor):
        return tensor;

NonLinear = {'sigmoid':torch.nn.Sigmoid, 'ReLu': torch.nn.ReLU, 'tanh':torch.nn.Tanh, 'noop':NoOp};

class dense_network_new(torch.nn.Module):
    """
    The dense network is a memory-less feed-forward network
    """
    def __init__(self, layer_configurations):
        super().__init__();

        layers      = []; # Anand - streamlining dense layer for speed: Sept 30 2018
        # self.activations = []; # Anand - streamlining dense layer for speed: Sept 30 2018

        # Anand: self.layers changed to layers in the for loop below: streamlining dense layer for speed: Sept 30 2018
        for i, configuration in enumerate(layer_configurations):
            layers.append(torch.nn.Linear(configuration[0], configuration[1]));

            layers[-1].weight.data.normal_(0.0,0.1);

            if len(configuration) > 3:
                layers[-1].bias.data = torch.from_numpy(configuration[3]).float();
            else:
                layers[-1].bias.data.fill_(0.1);

            # setattr(self, 'layer' + str(i), self.layers[-1]); # Anand - streamlining dense layer for speed: Sept 30 2018

            ### ANAND: Hack for retrying Cbf1 with dropout: Oct 4th
            if i == len(layer_configurations) - 2:
                print("Adding Dropout");
                layers.append(torch.nn.Dropout(p=0.5));
            #### Hack end

            # self.activations.append(nonlinear[configuration[2]]); # Anand - streamlining dense layer for speed: Sept 30 2018
            layers.append(NonLinear[configuration[2]]());

        self.layers = torch.nn.Sequential(*layers); # Anand - streamlining dense layer for speed: Sept 30 2018

    def print_fn(self):
        return [(l.weight.grad, l.bias.grad) for l in self.layers];

    def forward(self, input_tensor):
        """
        Input tensor has shape [batch_size * sequence_length * embedding length]
        """

        # Following commented out on Sept 30 2018. Statement below introduced on same date. Message - streamlining dense layer for speed
        # ip_sequence = torch.unbind(input_tensor, 1);

        # op_sequence = [];

        # for ip in ip_sequence:
        #     current_tensor = ip;

        #     for (l, a) in zip(self.layers, self.activations):
        #         current_tensor = a(l(current_tensor));

        #     op_sequence.append(current_tensor);

        # return torch.stack(op_sequence, 1);

        return self.layers(input_tensor);
