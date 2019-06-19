# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import torch
import numpy
from dense_network import dense_network
from dense_network_new import dense_network_new

new_model = False;

class lstm_network(torch.nn.Module):
    def __init__(self, layer_configurations, bidirectional=True, device=0):
        super().__init__();

        configuration       = layer_configurations[0];
        """
        Configuration format - the first configuration represents the LSTM layers
        configuration[0] = input channels
        configuration[1] = hidden_channels
        configuration[2] = number of lstm layers
        """
        self.lstm_network = torch.nn.LSTM(configuration[0], configuration[1], num_layers=configuration[2], batch_first=True, bidirectional=bidirectional);
        for param in self.lstm_network.parameters():
            param.data.normal_(0.0,0.1);

        self.lstm_init    = (configuration[2] * (2 if bidirectional is True else 1), None, configuration[1]);

        """ Subsequent configurations should conform to the dense_network requirements """
        if new_model: # Added if on Sept 30 2018 for order 1 LSGM using a new, more efficient, dense network
            print("NOTE: Using new dense network model");
            self.dense_layers = dense_network_new(layer_configurations[1:]);
        else:
            self.dense_layers = dense_network(layer_configurations[1:]); # Anand: conserved for backward compatibility: Sept 30 2018

        self.device = device;

    def forward(self, input_tensor):
        """Initilization of LSTM
        h0/c0 : (num_layers*num_directions, batch_size, hidden_size)
        """
        batch_size = input_tensor.size()[0];

        h0 = torch.autograd.Variable(torch.zeros(self.lstm_init[0], batch_size, self.lstm_init[2]));
        c0 = torch.autograd.Variable(torch.zeros(self.lstm_init[0], batch_size, self.lstm_init[2]));

        if next(self.parameters()).is_cuda:
            h0 = h0.cuda(self.device);
            c0 = c0.cuda(self.device);

        current_tensor, state = self.lstm_network(input_tensor, (h0, c0));
        op_sequence           = self.dense_layers(current_tensor);

        return op_sequence;
