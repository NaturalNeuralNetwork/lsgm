# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import torch
import numpy as np

class convnet(torch.nn.Module):
    def __init__(self, num_motifs, motif_length, analysis="classify", dropout=0.0):
        super().__init__();

        num_outputs  = 1 if analysis == "regress" else 2;

        self.layer1  = torch.nn.Conv1d(4, num_motifs, motif_length);
        self.layer2  = torch.nn.Linear(num_motifs*2, 32);
        self.layer3  = torch.nn.Linear(32, num_outputs);

        if dropout > 0:
            self.dropout = torch.nn.Dropout(p=dropout);
        else:
            self.dropout = torch.nn.Dropout();

        self.dropout_value = dropout;

        self.layer1.weight.data.normal_(0.0,0.1);
        self.layer2.weight.data.normal_(0.0,0.1);
        self.layer3.weight.data.normal_(0.0,0.1);
        self.layer1.bias.data.fill_(0.1);
        self.layer2.bias.data.fill_(0.1);
        self.layer3.bias.data.fill_(0.1);

    def forward(self, items):
        mask_tensor = items[2];
        mask_kernel = torch.transpose(mask_tensor, 1, 2);
        batch_size  = mask_kernel.size()[0];

        l1  = torch.nn.functional.relu(self.layer1(mask_kernel));
        l2a = torch.nn.functional.avg_pool1d(l1, l1.size()[2]).view(batch_size,-1);
        l2b = torch.nn.functional.max_pool1d(l1, l1.size()[2]).view(batch_size,-1);
        l3  = self.layer2(torch.cat((l2a, l2b), dim=1));

        if self.dropout_value > 0: l3 = self.dropout(l3);

        l4  = self.layer3(l3);

        return l4;
