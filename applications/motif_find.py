# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../");
sys.path.append(dir_path + "/../networks");
sys.path.append(dir_path + "/../misc");
import torch
import numpy as np
from motif_finder import motif_finder
import argparse
import re
from data_generator import data_generator
from operator import itemgetter
from convnet import convnet
import scipy.stats
import lstm_network # Anand: change for 1st-order LSGM

def classifier_accuracy(output_tensor, target):
    output_labels      = output_tensor;
    output_non1h       = np.argmax(output_labels, axis=1);
    correct_prediction = np.add.reduce(np.array(output_non1h == target, dtype=np.float32)) / output_labels.shape[0];
    return correct_prediction;

class progress:
    def __init__(self, total, step_size, character = "#"):
        self.total         = total;
        self.step_size     = step_size;
        self.__last_update = 0;
        self.character     = character;
        message            = "{0:.2f}".format(0);

        sys.stdout.write(message);
        sys.stdout.flush();

    def __call__(self, current_progress):
        num_steps = int((current_progress / self.total) / self.step_size) - self.__last_update;

        if num_steps > 0:
            message   = self.character * num_steps;
            message   = message + " " + "{0:.2f}".format(current_progress/self.total);

            sys.stdout.write("\b\b\b\b\b");

            sys.stdout.write(message);
            sys.stdout.flush();

            self.__last_update = int((current_progress / self.total) / self.step_size);

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train and test a motif classifier");

    parser.add_argument("--train_data", action="store", dest="train_data", help="Path of train data file", required=True);
    parser.add_argument("--test_data", action="store", dest="test_data", help="Path of test data file", required=True);
    parser.add_argument("--motif_length", action="store", type=int, dest="motif_length", help="Length of the motifs", required=True);
    parser.add_argument("--num_motifs", action="store", type=int, dest="num_motifs", help="Number of motifs", required=True);
    parser.add_argument("--long_term_recurrence", action="store_true", dest="lt_recur", help="Turn on long term recurrence", default=False);
    parser.add_argument("--bidirectional", action="store_true", dest="bidirectional", help="Enable bidirectional analysis", default=False);
    parser.add_argument("--recurrence_type", action="store", dest="recurrence_type", help="Type of long term recurrence", default="LSTM", choices=["LSTM","GRU"]);
    parser.add_argument("--graph_order", action="store", type=int, dest="hmm_order", help="The order of the graphical model to be used", default=0);
    parser.add_argument("--model", action="store", dest="model", help="Pre-trained model parameter file for initialization", required=False, default=None);
    parser.add_argument("--learning_rate", action="store", dest="learning_rate", help="Learning rate", default=1e-4, type=float);
    parser.add_argument("--batch_size", type=int, action="store", dest="batch_size", help="Batch size for training", default=10);
    parser.add_argument("--num_epochs", type=int, action="store", dest="num_epochs", help="Number of epochs to train for", default=1);
    parser.add_argument("--cuda", action="store_true", dest="cuda", help="Move all parameters to GPU", default=False);
    parser.add_argument("--device", action="store", dest="device", help="Device id if using cuda", type=int, default=0);
    parser.add_argument("--output_prefix", action="store", dest="output_prefix", help="Prefix for output files", required=True);
    parser.add_argument("--sequence_length", action="store", dest="sequence_length", help="Sequence length", type=int, default=35);
    parser.add_argument("--preactivation", action="store", choices=["relu", "exp", "square", "cube", "quadruple", "linear"],
                            dest="preactivation", help="Type of activation to use in the penultimate layer", default="linear");
    parser.add_argument("--suffix_primer", action="store", type=int, dest="suffix_primer", help="Length of suffix primer", default=25);
    parser.add_argument("--num_long_term_states", action="store", help="Number of long term states per layer", dest="num_long_term_states", type=int, default=128);
    parser.add_argument("--use_likelihood_ratio", action="store_true", help="Use likelihood ratio", dest="use_likelihood_ratio", default=False);
    parser.add_argument("--init_pwms", action="store", help="Initialization for PWMs", dest="init_pwms");
    parser.add_argument("--freeze_pwms", action="store_true", help="Train PWMs", dest="freeze_pwms", default=False);
    parser.add_argument("--remove_final_layer", action="store_true", help="Remove the final layer in the NN classifier", dest="remove_final_layer", default=False);
    parser.add_argument("--use_log_forward", action="store_true", help="Enable alpha version log forward computation", default=False);

    args = parser.parse_args();

    # Sept 30 2018: Change for first order LSGM
    if args.hmm_order == 1:
        lstm_network.new_model = True;

    train_pwms = not args.freeze_pwms;

    regressor      = True;
    train_patterns = [];
    train_labels   = [];

    train_data = data_generator(args.train_data, max_batch_size=args.batch_size, \
                    cuda=args.cuda, regressor = regressor, max_sequence_length = args.sequence_length, suffix_length=args.suffix_primer);
    test_data  = data_generator(args.test_data, max_batch_size=args.batch_size, \
                    cuda=args.cuda, regressor = regressor, max_sequence_length = args.sequence_length, suffix_length=args.suffix_primer);

    motif_discoverer = None;

    motif_discoverer = \
            motif_finder(\
                args.num_motifs, \
                args.motif_length, \
                mode               = 0, \
                order              = args.hmm_order, \
                lt_recur           = args.lt_recur, \
                bidirectional      = args.bidirectional, \
                recurrence_type    = args.recurrence_type, \
                device             = args.device, \
                preactivation      = args.preactivation, \
                regressor          = regressor, \
                num_lt_states      = args.num_long_term_states, \
                train_pwms         = train_pwms, \
                remove_final_layer = args.remove_final_layer, \
                use_likelihood_ratio = args.use_likelihood_ratio, \
                use_log_forward    = args.use_log_forward,
            );

    if args.init_pwms is not None:
        pwms = np.load(args.init_pwms);
        motif_discoverer.motif_model.motif_state.weights.data = torch.from_numpy(pwms).float();

    print("Creating motif classifier instance");

    if args.model is not None:
        with open(args.model, 'rb') as file:
            params = torch.load(file);

        motif_discoverer.load_state_dict(params);

    if args.cuda:
        motif_discoverer.cuda(args.device);

    optimizer        = torch.optim.Adam(filter(lambda p: p.requires_grad, motif_discoverer.parameters()), lr=args.learning_rate);
    prev_accuracy    = -1;
    prev_index       = -1;

    for j in range(args.num_epochs):
        accuracies = [];

        for iterator, data in enumerate([train_data, test_data]):
            if iterator == 0:
                motif_discoverer.train(True);
            else:
                motif_discoverer.train(False);

            num_batches = data.size() // args.batch_size;

            if num_batches * args.batch_size < data.size():
                num_batches += 1;

            num_correct = 0;

            sys.stdout.write("%s iteration progress: "%(["train","test "][iterator]));
            sys.stdout.flush();

            progressbar = progress(num_batches,0.01);

            all_predicted_labels = [];
            all_target_labels    = [];

            for i in range(num_batches):
                batch_start = i * args.batch_size;
                batch_end   = min((i + 1) * args.batch_size, data.size());

                coded_patterns, embeddings_patterns, mask_tensor, labels_torch = data[batch_start:batch_end];

                if args.cuda:
                    coded_patterns      = coded_patterns.cuda(device=args.device, async=True);
                    embeddings_patterns = embeddings_patterns.cuda(device=args.device, async=True);
                    mask_tensor         = mask_tensor.cuda(device=args.device, async=True);

                codes  = torch.autograd.Variable(coded_patterns);
                embeds = torch.autograd.Variable(embeddings_patterns);
                masks  = torch.autograd.Variable(mask_tensor);

                predicted_labels = motif_discoverer([codes, embeds, masks]);

                if iterator == 0:
                    criterion = None;

                    criterion = torch.nn.MSELoss();

                    if args.cuda:
                        criterion.cuda(args.device);

                    labels_slice = labels_torch;

                    if args.cuda:
                        labels_slice = labels_slice.cuda(device=args.device);

                    target              = torch.autograd.Variable(labels_slice, requires_grad=False);
                    loss_function       = criterion(predicted_labels.view(predicted_labels.size(0)), target);

                    optimizer.zero_grad();
                    loss_function.backward();
                    optimizer.step();

                progressbar(i);

                predicted_labels_np_ = predicted_labels.cpu().data.numpy();
                target_labels_np_    = labels_torch.cpu().numpy();
                predicted_labels_np  = None;

                predicted_labels_np = np.reshape(predicted_labels_np_, (predicted_labels_np_.shape[0],))

                target_labels_np     = np.reshape(target_labels_np_, (target_labels_np_.shape[0],));

                all_predicted_labels.append(predicted_labels_np);
                all_target_labels.append(target_labels_np);

            iteration_name = ["train", "test "][iterator];

            print("");

            predictions_np   = np.concatenate(all_predicted_labels, axis=0);
            targets_np       = np.concatenate(all_target_labels);
            accuracy_measure = None;

            (correlation, p_value) = scipy.stats.spearmanr(predictions_np, targets_np);

            accuracy_measure = correlation;

            accuracies.append(accuracy_measure);

            if iterator == 1:
                print("Completed validation iteration, obtained overall spearman score %f ... "%(accuracy_measure));

        print("Completed epoch %d"%(j));

        if accuracies[1] > prev_accuracy:
            params = motif_discoverer.state_dict();

            with open(args.output_prefix + ".dnn", "wb") as file:
                torch.save(params, file);

            prev_accuracy = accuracies[1];
            prev_index    = j;
    
    print("Best model set had spearman score %f, at iteration %d"%(prev_accuracy, prev_index));
