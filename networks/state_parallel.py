# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import torch
# from  lstm_network import lstm_network
# from dense_network import dense_network
import numpy as np
from header import weight_variable

# network_types  = {'lstm':lstm_network, 'dense':dense_network};

class motif_collection(torch.nn.Module):
    def __init__(self, length, num_motifs, device = 0, train_pwms = True):
        super().__init__();

        self.length  = length;
        self.num     = num_motifs;
        self.device  = device;

        self.weights = torch.nn.Parameter(weight_variable(shape=(self.num,4,self.length)), requires_grad = train_pwms);

    def forward(self, items, no_post_process = False):
        [coded_sequences, embeddings_sequences, mask_tensor] = items;

        emissions       = [];
        sequence_length = coded_sequences.size()[1];
        batch_size      = coded_sequences.size()[0];

        pwmemissions    = torch.exp(self.weights);
        pwmemissions    = pwmemissions / torch.sum(pwmemissions, dim=1, keepdim=True);
        pwmemissions    = torch.log(pwmemissions);              # Kernel - [out_channels x 4 x length]
        mask_kernel     = torch.transpose(mask_tensor,1,2);     # Input  - [batch x 4 x seq_length]
        emissions       = torch.nn.functional.conv1d(mask_kernel, pwmemissions); # Output - [batch x out_chanels x seq_length - length + 1]
        emissions       = torch.exp(emissions);

        if not no_post_process:
            zero_pad  = torch.autograd.Variable(torch.zeros(batch_size,self.num,self.length-1)); # Pad - [batch x out_channels x length - 1]
            is_cuda   = next(self.parameters()).is_cuda;
            if is_cuda: zero_pad = zero_pad.cuda(self.device);
            emissions = torch.cat((zero_pad, emissions), dim=2); # batch x num_motifs x length

        return torch.transpose(emissions, 0, 1); # num_motifs x batch x length 

class notif_state(motif_collection):
    def __init__(self):
        super().__init__(length=1, num_motifs=1);

# class motif_state_w(torch.nn.Module):
#     def __init__(self, length, max_batch_size=100, device = 0):
#         super().__init__();
# 
#         self.length       = length;
#         self.weights      = torch.nn.Parameter(weight_variable(shape=(self.length,4)));
#         self.device       = device;
# 
#     # @profile
#     def forward(self, items):
#         [coded_sequences, embeddings_sequences, mask_tensor] = items;
# 
#         emissions       = [];
#         sequence_length = coded_sequences.size()[1];
#         batch_size      = coded_sequences.size()[0];
#         pwmemissions    = torch.nn.functional.softmax(self.weights).view(self.length,4,1);
#         pwmemissions    = torch.log(torch.transpose(pwmemissions, 0, 2).contiguous());
#         mask_kernel     = torch.transpose(mask_tensor, 1, 2);
#         emissions       = torch.exp(torch.squeeze(torch.nn.functional.conv1d(mask_kernel, pwmemissions), dim=1));
#         zero_pad        = torch.autograd.Variable(torch.zeros(batch_size,self.length-1));
# 
#         is_cuda = next(self.parameters()).is_cuda;
# 
#         if is_cuda: zero_pad = zero_pad.cuda(self.device);
# 
#         emissions        = torch.cat((zero_pad,emissions), dim=1);
# 
#         return emissions;
# 
# class motif_state(torch.nn.Module):
#     def __init__(self, length):
#         super().__init__();
# 
#         self.length       = length;
#         self.emissions    = lstm_network([(4,16,16),(32,4,"softmax")]);
# 
#     def forward(self, items):
#         [coded_sequences, embeddings_sequences, mask_tensor] = items;
# 
#         emissions       = [];
#         sequence_length = coded_sequences.size()[1];
#         batch_size      = coded_sequences.size()[0];
# 
#         # At every position 'i' in the sequence, create a window of length self.length 
#         for i in range(sequence_length - self.length + 1):
#             subsequence  = coded_sequences[:,i:i+self.length,:];
#             submask      = mask_tensor[:,i:i+self.length,:];
#             subemissions = self.emissions(subsequence);
# 
#             subemissions = torch.sum(subemissions * submask, dim=2);
# 
#             # Collapse into single value along the sequence_length dimension
#             subemissions = torch.prod(subemissions, dim=1);
#             emissions.append(subemissions);
# 
#         """
#         An elements in the emissions list is [batch_size x 1]. Add (self.length - 1) number of such elements at the start
#         """
#         for i in range(self.length-1):
#             emissions.insert(0,torch.autograd.Variable(torch.zeros(batch_size)));
# 
#         return torch.stack(emissions, dim=1);

# class notif_state(torch.nn.Module):
#     def __init__(self):
#         super().__init__();
# 
#         self.length       = 1;
#         self.emissions    = dense_network([(16,32,"ReLu"),(32,4,"softmax")]);
# 
#     # @profile
#     def forward(self, items):
#         [coded_sequences, embeddings_sequences, mask_tensor] = items;
#         
#         mask      = mask_tensor.float();
#         emissions = torch.sum(self.emissions(embeddings_sequences) * mask, dim=2);
# 
#         return emissions;
