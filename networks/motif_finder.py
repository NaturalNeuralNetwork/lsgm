# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import torch
from bidirectional_lsgm import bidirectional_lsgm as blsgm
from functools import reduce
import numpy as np
from header import small_value
from header import weight_variable

""" Power activations. The steep slope of log is countered by the flatness of power functions """
class Exponential(torch.nn.Module):
    def __init__(self):
        super().__init__();

    def forward(self, tensor):
        return torch.exp(tensor);

class Square(torch.nn.Module):
    def __init__(self):
        super().__init__();

    def forward(self, tensor):
        return tensor ** 2;

class Cube(torch.nn.Module):
    def __init__(self):
        super().__init__();

    def forward(self, tensor):
        return tensor ** 3;

class Quadruple(torch.nn.Module):
    def __init__(self):
        super().__init__();

    def forward(self, tensor):
        return tensor ** 4;

class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__();

    def forward(self, tensor):
        return tensor;

class motif_finder(torch.nn.Module):
    def __init__(self, \
            num_motifs, \
            motif_length, \
            mode, \
            order, \
            lt_recur = False, \
            bidirectional = True, \
            recurrence_type="LSTM", \
            device = 0, \
            preactivation = "square",  \
            regressor = True, \
            num_lt_states = 128, \
            train_pwms = True, \
            remove_final_layer = False, \
            use_likelihood_ratio = False, \
            use_log_forward=False,
        ):
        super().__init__();

        lt_type = "Markov" if lt_recur is False else "LT";

        self.recurrence_type  = recurrence_type;
        self.bidirectional    = bidirectional;
        self.device           = device;
        self.mode             = mode;   # Deprecated
        self.order            = order;

        self.motif_model      = \
            blsgm(\
                motif_length    = motif_length, \
                num_motifs      = num_motifs, \
                order           = order, \
                transition_type = lt_type, \
                recurrence_type = self.recurrence_type, \
                device          = device, \
                num_lt_states   = num_lt_states, \
                train_pwms      = train_pwms, \
                use_log_forward = use_log_forward, \
            );

        self.motif_model.mode = mode;   # Deprecated
        self.preactivation    = preactivation;
        self.regressor        = regressor;
        num_outputs           = 1 if regressor else 2;
        penultimate           = Exponential() if preactivation == "exp" else \
                                    Square() if preactivation == "square" else \
                                        Cube() if preactivation == "cube" else \
                                            Quadruple() if preactivation == "quadruple" else \
                                                torch.nn.ReLU() if preactivation == "linear" else \
                                                    Linear();

        linear1s = torch.nn.Linear(2,256);
        linear1b = torch.nn.Linear(4,256);
        linear2  = torch.nn.Linear(256,256);
        linear3  = torch.nn.Linear(256,num_outputs);

        def init_weights(layer):
            weight_shape = layer.weight.data.size();
            bias_shape   = layer.bias.data.size();

            weight_shape = (weight_shape[0], weight_shape[1]);
            bias_shape   = (bias_shape[0],);

            layer.weight.data = weight_variable(weight_shape).float();
            layer.bias.data   = torch.from_numpy(np.ones(bias_shape)).float() / 100.0;

        init_weights(linear1s);
        init_weights(linear1b);
        init_weights(linear2);
        init_weights(linear3);

        """ Small neural network to combine the predictions together """
        if remove_final_layer:
            self.dense_layer_b = torch.nn.Sequential(
                        linear1b,
                        torch.nn.ReLU(),
                        linear3,
                        penultimate
                );
            self.dense_layer_s = torch.nn.Sequential(
                        linear1s,
                        torch.nn.ReLU(),
                        linear3,
                        penultimate
                );
        else:
            self.dense_layer_b = torch.nn.Sequential(
                        linear1b,
                        torch.nn.ReLU(),
                        linear2,
                        penultimate,
                        linear3
                );
            self.dense_layer_s = torch.nn.Sequential(
                        linear1s,
                        torch.nn.ReLU(),
                        linear2,
                        penultimate,
                        linear3
                );

        self.W = torch.nn.Parameter(torch.Tensor([1.0]).float());
        self.b = torch.nn.Parameter(torch.Tensor([0.1]).float());

        self.scale = torch.nn.Parameter(torch.Tensor([0.01]).float());

        self.use_likelihood_ratio = use_likelihood_ratio;

    def forward(self, items):
        coded_sequences      = items[0];
        embeddings_sequences = items[1];
        mask_tensor          = items[2];
        dense_input          = self.motif_model([coded_sequences, embeddings_sequences, mask_tensor], bidirectional=self.bidirectional);
        
        if not self.use_likelihood_ratio:
            # prediction           = self.dense_layer_b(dense_input.float()).double() if self.bidirectional else \
            #                         self.dense_layer_s(dense_input.float()).double();

            prediction           = self.dense_layer_b(dense_input.float()) if self.bidirectional else \
                                    self.dense_layer_s(dense_input.float());

        else:
            if self.bidirectional:
                raise ValueError("Use of log-likelihood not implemented for bidirectional lsgm");

            difference  = dense_input[:,0] - dense_input[:,1]; 
            scale_diff  = self.scale * difference.float();
            relu_diff   = scale_diff;
            
            # Mimic exponential distribution
            if self.preactivation in ["square", "cube", "quadruple"]:
                relu_diff   = torch.nn.functional.relu(scale_diff);

            output = {  "square"    : lambda x : x * x, \
                        "cube"      : lambda x : x * x * x, \
                        "quadruple" : lambda x : x * x * x * x, \
                        "exp"       : lambda x : torch.exp(x), \
            }

            # prediction  = (output[self.preactivation](relu_diff) * self.W + self.b).double();
            prediction  = (output[self.preactivation](relu_diff) * self.W + self.b);

        return prediction;
