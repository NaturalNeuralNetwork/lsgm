# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import torch
from functools import reduce

def rms_grad(model):
    mean_grad_norm      = 0;
    num_items_grad_norm = 0;

    for item in model.parameters():
        if hasattr(item.grad, 'data'):
            size_grad = reduce(lambda x, y : x * y, item.size(), 1);

            mean_grad_norm = mean_grad_norm * (num_items_grad_norm / (num_items_grad_norm + size_grad)) + \
                                (item.grad.data ** 2).sum() / (num_items_grad_norm + size_grad);

            num_items_grad_norm += size_grad;

    return mean_grad_norm ** (0.5);
