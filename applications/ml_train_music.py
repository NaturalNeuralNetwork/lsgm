# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(dir_path + "/../");
sys.path.append(dir_path + "/../networks");
sys.path.append(dir_path + "/../misc");

import torch
from data_reader import training_dataset_npy_discrete as music_reader
from networks import lsgm 
import argparse
from misc import ml_iteration
import _pickle as pickle
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description = "Train a music model through ML estimation"
             );

    parser.add_argument(
        "--val_data", 
        action   = "store",
        help     = "Data in the validation list",
        required = True,
    );

    parser.add_argument(
        "--train_data",
        action   = "store",
        help     = "List of audio files in the training list",
        required = True,
    );

    parser.add_argument(
        "--learning_rate",
        action   = "store",
        type     = float,
        help     = "Learning rate",
        default  = 1e-4
    );

    parser.add_argument(
        "--num_epochs",
        action   = "store",
        type     = int,
        help     = "Number of epochs to train",
        default  = 100
    );

    parser.add_argument(
        "--batch_size",
        action   = "store",
        type     = int,
        dest     = "batch_size",
        help     = "Batch size to use",
        default  = 100
    );

    parser.add_argument(
        "--dimension",
		 action  = "store",
		 type    = int,
		 help    = "Dimension of emissions",
		 default = 50
    );

    parser.add_argument(
        "--dropout",
		 action  = "store",
		 type    = float,
		 help    = "Dropout rate",
		 default = 0.0,
    );

    parser.add_argument(
        "--clip_gradients",
        action  = "store_true",
        help    = "Enable gradient clipping",
        default = False,
    );

    parser.add_argument(
        "--clip_threshold",
        action  ="store",
        type    = float,
        help    = "Threshold for gradient clipping",
        default = None,
    );

    parser.add_argument(
        "--num_emission_layers",
        action   = "store",
        help     = "Number of emission layers",
        type     = int,
        required = False,
        default  = 2,
    );

    parser.add_argument(
        "--num_stochastic_nodes",
        action   = "store",
        help     = "Number of stochastic nodes per graph",
        type     = int,
        required = True
    );

    parser.add_argument(
        "--num_recurrent_layers",
        action   = "store",
        help     = "Number of recurrent layers",
        type     = int,
        required = True
    );

    parser.add_argument(
        "--num_recurrent_units",
        action   = "store",
        help     = "Number of recurrent nodes",
        type     = int,
        required = True
    );

    parser.add_argument(
        "--num_dense_layers",
        action   = "store",
        help     = "Number of dense layers",
        type     = int,
        required = True
    );

    parser.add_argument(
        "--num_dense_units",
        action   = "store",
        help     = "Number of dense nodes",
        type     = int,
        required = True
    );

    parser.add_argument(
        "--cuda",
        action   = "store_true",
        help     = "Flag to move computation to GPU",
        default  = False
    );

    parser.add_argument(
        "--do_checks",
        action   = "store_true",
        help     = "Flag to enable checks during computations",
        default  = False
    );

    parser.add_argument(
        "--output_prefix",
        action   = "store",
		help     = "Prefix of the output file",
		required = True
    );

    parser.add_argument(
        "--model",
        action   = "store",
		help     = "Pre-trained model to use, if any",
		required = False
    );

    parser.add_argument(
        "--activation",
        action="store",
        choices=["leaky_relu", "relu", "clipped_leaky_relu", "softplus"],
        default="softplus",
        help="Activations for non-recurrent layers",
    );

    parser.add_argument(
        "--use_nonlin_emission",
        action="store_true",
        default=False,
        help="Use nonlinearity on emission layer",
    );

    parser.add_argument(
         "--order",
		 action   = "store",
		 type     = int,
		 help     = "Markov order of graphs",
		 default  = 1,
    );

    parser.add_argument(
         "--testmode",
		 action   = "store_true",
		 help     = "Enable testmode operation",
		 default  = False,
    );

    parser.add_argument(
        "--do_not_shuffle",
        action    = "store_true",
        help      = "Do not shuffle the database if it has been pre-shuffled",
        default   = False
    );

    parser.add_argument(
        "--entropy_regularizer",
        action    = "store_true",
        help      = "Regularize cost function using entropy of states",
        default   = False,
    );

    parser.add_argument(
        "--entropy_type",
        action    = "store",
        help      = "Entropy type for regularization",
        default   = "forward",
        choices   = ["forward","posterior","multimodal"],
    );

    parser.add_argument(
        "--regularizer_ratio",
        action    = "store",
        type      = float,
        help      = "How much to scale regularizer cost by",
        default   = 1,
    );

    parser.add_argument(
        "--decay_rate",
        action    = "store",
        type      = float,
        help      = "How does the regularizer ratio decay",
        default   = 0,
    );

    parser.add_argument(
        "--heating_iterations",
        action    = "store",
        type      = int,
        help      = "Number of iterations to heat",
        default   = 10,
    );

    parser.add_argument(
        "--cooling_iterations",
        action    = "store",
        type      = int,
        help      = "Number of iterations to cool",
        default   = 10,
    );

    parser.add_argument(
        "--min_consecutive_misses",
        action    = "store",
        type      = int,
        help      = "Number of cycles when likelihood doesn't improve to terminate training",
        default   = 3,
    );

    parser.add_argument(
      "--heat_until_likelihood_hit",
        help      = "When entropy regularizer is used, heat only until likelihood stops increasing",
        action    = 'store_true',
        default   = False,
    );

    parser.add_argument(
        "--early_stop",
        action    = "store_true",
        help      = "Stop early after heating and cooling",
        default   = False,
    );

    parser.add_argument(
        "--regularization_rate",
        action     = "store",
        type       = float,
        help       = "Regularization rate for \
                                transition probabilities",
        default    = 0.0,
    );

    parser.add_argument(
        "--emission_type",
        action     = 'store',
        help       = 'Type of emission',
        choices    = ['binary', 'continuous'],
        default    = 'binary',
    );

    parser.add_argument(
        "--embedding_dim",
        action     = 'store',
        type       = int,
        default    = 256,
        help       = 'Dimension of state embedding',
    );

    parser.add_argument(
        "--num_workers",
        action     = 'store',
        type       = int,
        default    = 12,
        help       = 'Number of worker threads to use',
    );

    parser.add_argument(
        "--use_glorot",
        action     = 'store_true',
        default    = False,
        help       = "Use Xavier Glorot initialization",
    );

    args = parser.parse_args();

    if args.num_epochs == 0:
        lsgm.SMALL = 0;

    regularizer_ratio = args.regularizer_ratio;

    module = lsgm.lsgm;

    model = module(
        num_stochastic_nodes    = args.num_stochastic_nodes,
        num_recurrent_layers    = args.num_recurrent_layers,
        num_dense_layers        = args.num_dense_layers,
        num_recurrent_units     = args.num_recurrent_units,
        num_dense_units         = args.num_dense_units,
        dimension               = args.dimension,
        dropout                 = args.dropout,
        dense_activations       = args.activation,
        use_emission_activation = args.use_nonlin_emission,
        order                   = args.order,
        testmode                = args.testmode,
        regularization_rate     = args.regularization_rate,
        entropy_type            = args.entropy_type,
        emission_type           = args.emission_type,
        embedding_dim           = args.embedding_dim,
        num_emission_layers     = args.num_emission_layers,
    );

    reader = music_reader(
                train_npy  = args.train_data,
                val_npy    = args.val_data,
                batch_size = args.batch_size,
                shuffle    = not args.do_not_shuffle,
                cuda       = args.cuda,
                dim        = args.dimension,
                num_workers= args.num_workers,
    );

    if args.model is not None:
        with open(args.model, 'rb') as file:
            params = torch.load(file);
            model.load_state_dict(params);

    if args.cuda:
        model.cuda();

    optimizer  = torch.optim.Adam(model.parameters(), lr=args.learning_rate);
    prev_val   = None;

    print("Evaluating likelihood on validation data before training ... ");

    prev_val, posterior_map = ml_iteration(
                model,
                reader,
                find_posterior_map = True,
               );

    print("Likelihood = %f\n"%prev_val);
    print();

    if args.num_epochs == 0:
        with open(args.output_prefix + "_posterior_map.data", 'wb') as fhandle:
            pickle.dump(posterior_map, fhandle);

        sys.exit(-1) 

    clip_threshold = None;

    num_consecutive_misses = 0;

    if args.clip_gradients and (args.clip_threshold is None):
        print("Determining clip threshold as average RMS value per input");
        clip_threshold = ml_iteration(
            model,
            reader,
            optim              = optimizer,
            determine_rms_grad = True
        );
        print("Obtained clip threshold = %f"%(clip_threshold));
    else:
        clip_threshold = args.clip_threshold;

    num_iterations_min = 0;

    if args.entropy_regularizer:
        num_iterations_min = args.heating_iterations + args.cooling_iterations + 1;
    else:
        num_iterations_min = 1;

    likelihood_hit = False;

    for i in range(args.num_epochs):
        print("Commencing training iterations");

        reg_ratio = 0;

        if args.entropy_regularizer:
            if args.heat_until_likelihood_hit:
                if not likelihood_hit:
                    reg_ratio = regularizer_ratio;
                else:
                    print("Likelihood has been hit, not heating anymore");
            else:
                if i < args.heating_iterations:
                    reg_ratio = regularizer_ratio;
                elif i < args.heating_iterations + args.cooling_iterations:
                    reg_ratio = -regularizer_ratio;
                else:
                    reg_ratio = 0;

        ml_iteration(
            model,
            reader,
            optim          = optimizer,
            clip_threshold = clip_threshold,
            reg_ratio      = reg_ratio,
        );

        total_prob_val = ml_iteration(model, reader);

        print("Completed epoch %d, validation likelihood is %f"%(i, total_prob_val));

        params = model.state_dict();

        if prev_val <= total_prob_val:
            with open(args.output_prefix + ".lsgm", 'wb') as file:
                print("Saving model ... ");
                torch.save(params, file);

            prev_val = total_prob_val;

            num_consecutive_misses = 0;
        else:
            if i >= num_iterations_min: num_consecutive_misses += 1;

            if args.early_stop and (i >= num_iterations_min) and (num_consecutive_misses >= args.min_consecutive_misses):
                print("Model overfits. Stopping early.");
                break;
            else:
                print("Model overfits in this iteration, not saving");

            likelihood_hit = True;

        regularizer_ratio = args.regularizer_ratio * math.e ** (-args.decay_rate * i);
