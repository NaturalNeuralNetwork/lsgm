# © 2019 University of Illinois Board of Trustees.  All rights reserved.
from progress import progress
import sys
import torch
import numpy as np
import warnings
from gradient_clip import rms_grad

def ml_iteration(
        model,
        reader,
        optim=None,
        determine_rms_grad=False,
        clip_threshold=None,
        reg_ratio=0,
        find_posterior_map=False,
    ):
    train = optim is not None;

    model.train(train);
    reader.train(train);

    if train:
        sys.stdout.write("Train iteration progress: ");
        sys.stdout.flush();
    else:
        sys.stdout.write("Val iteration progress  : ");
        sys.stdout.flush();

    progressbar = progress(len(reader),0.01);
    batches     = iter(reader);
    total_value = 0;
    total_size  = 0;

    posterior_maps = [];

    for i, batch in enumerate(batches):
        data    = batch;
        lengths = None;
        labels  = None;

        if (type(data) is list) or (type(data) is tuple):
            assert(len(data) in [2, 3]);

            data    = batch[0];
            lengths = batch[1];

            if len(batch) == 3:
                labels = batch[2];

        if next(model.parameters()).is_cuda:
            data = data.cuda(non_blocking=True);

        data            = torch.autograd.Variable(data);
        batch_size      = data.size(0);
        total_size     += batch_size;
        total_rms_grad  = 0;

        # values_         = model(data) if lengths is None \
        #                    else model(data, lengths);

        args = [];

        if labels is not None:
            args.insert(0, labels);

        if lengths is not None:
            args.insert(0, lengths);

        args.insert(0, data);

        values_ = model(*args);

        reg = None;

        if (type(values_) is list) or (type(values_) is tuple):
            assert(len(values_) == 2);

            cost = values_[0];
            reg  = values_[1];
        else:
            cost = values_;

        values_ = cost;

        if lengths is not None:
            lengths = torch.autograd.Variable(lengths.float());

            if values_.is_cuda:
                lengths = lengths.cuda(device = values_.get_device());

        value = torch.sum(values_) / batch_size if lengths is None else \
                   torch.sum(values_ / lengths) / batch_size;

        if (reg_ratio != 0) and not determine_rms_grad:
            if lengths is not None:
                value += torch.sum(reg / lengths) / batch_size * reg_ratio;
            else:
                value += torch.sum(reg) / batch_size * reg_ratio;

        if train:
            optim.zero_grad();

            (-value).backward();

            if clip_threshold is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold);

            if not determine_rms_grad:
                optim.step();
            else:
                total_rms_grad += rms_grad(model) * batch_size;

        adding_      = value * batch_size;
        total_value += float(adding_.cpu().data.numpy());

        if find_posterior_map and reg is not None:
            posterior_maps.append(reg.cpu().data.numpy());

        progressbar(i);

    progressbar(i+1);

    print();

    if determine_rms_grad:
        return total_rms_grad / total_size;
    else:
        if find_posterior_map:
            return total_value / total_size, posterior_maps;
        else:
            return total_value / total_size;
