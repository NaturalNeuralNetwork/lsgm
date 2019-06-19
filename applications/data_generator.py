# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import torch
import numpy as np
import re

def dataset(filename):
    pattern = re.compile("\S+\s([ACGT]+)\s([01])");

    patterns = [];
    labels   = [];

    fhandle = open(filename, 'r');

    for line in fhandle:
        result = pattern.match(line.rstrip());

        if result is not None:
            sequence = list(result.group(1));
            label    = int(result.group(2));

            patterns.append(sequence);
            labels.append(label);
        else:
            print(line);
            raise ValueError("File format not recognized");

    return (patterns, labels);

def dataset_regress(filename, num_primer_positions = 0):
    pattern = re.compile("([ACGT]+)\s+(-*\d+.\d+)");

    patterns = [];
    labels   = [];

    fhandle = open(filename, 'r');

    for line in fhandle:
        result = pattern.match(line.rstrip());

        if result is not None:
            sequence = list(result.group(1));
            label    = float(result.group(2));

            if num_primer_positions > 0:
                sequence = sequence[:-num_primer_positions];

            patterns.append(sequence);
            labels.append(label);

    if len(patterns) == 0:
        raise ValueError("No patterns found!");

    print("Found %d patterns, and %d labels"%(len(patterns), len(labels)));

    return (patterns, labels);

class data_generator(torch.nn.Module):
    def __init__(self, datafile, train=False, max_batch_size = 100, max_sequence_length = 101, cuda = False, regressor = False, suffix_length = 0):
        super().__init__();

        if regressor:
            self.patterns, self.labels = dataset_regress(datafile, num_primer_positions = suffix_length);
        else:
            self.patterns, self.labels = dataset(datafile);

        self.embedding = {      "A": torch.from_numpy(np.array([1,0,0,0])).float(), \
                                "C": torch.from_numpy(np.array([0,1,0,0])).float(), \
                                "G": torch.from_numpy(np.array([0,0,1,0])).float(), \
                                "T": torch.from_numpy(np.array([0,0,0,1])).float()    };

        for key, value in self.embedding.items(): setattr(self, "base" + key, value);

        self.max_batch_size         = max_batch_size;

        self.__embeddings_sequences = torch.zeros(max_batch_size, max_sequence_length, 16);
        self.__coded_sequences      = torch.zeros(max_batch_size, max_sequence_length, 4);
        self.__mask_tensor          = torch.zeros(max_batch_size, max_sequence_length, 4);

        if cuda:
            self.__embeddings_sequences = self.__embeddings_sequences.pin_memory();
            self.__coded_sequences      = self.__coded_sequences.pin_memory();
            self.__mask_tensor          = self.__mask_tensor.pin_memory();

        self.cuda = True;

    def size(self):
        return len(self.patterns);

    def coded_sequences(self, sequences):
        embedded_sequences = [torch.stack([self.embedding[i] for i in sequence], dim=0) for sequence in sequences];

        for i, sequence in enumerate(sequences):
            for j, base in enumerate(sequence):
                self.__coded_sequences[i,j,:] = self.embedding[base];

    def embeddings_sequences(self, sequences):
        embed_vector    = torch.cat([self.embedding[i] for i in ["A", "C", "G", "T"]], dim=0);

        for i, sequence in enumerate(sequences):
            for j, base in enumerate(sequence):
                self.__embeddings_sequences[i,j,:] = embed_vector;

    def mask_tensor(self, sequences):
        acgt = { \
            'A' : torch.from_numpy(np.array([1, 0, 0, 0])).float(), \
            'C' : torch.from_numpy(np.array([0, 1, 0, 0])).float(), \
            'G' : torch.from_numpy(np.array([0, 0, 1, 0])).float(), \
            'T' : torch.from_numpy(np.array([0, 0, 0, 1])).float(), \
        };

        for i, sequence in enumerate(sequences):
            for j, base in enumerate(sequence):
                self.__mask_tensor[i,j,:] = acgt[base];

    def __getitem__(self, key):
        vectors    = self.patterns[key];
        labels     = torch.from_numpy(np.array(self.labels[key])).float();
        batch_size = len(vectors);

        self.coded_sequences(vectors);
        self.embeddings_sequences(vectors);
        self.mask_tensor(vectors);

        return self.__coded_sequences[:batch_size], self.__embeddings_sequences[:batch_size], self.__mask_tensor[:batch_size], labels;

    def cuda(self, device=None):
        raise AttributeError("Cannot move module to GPU!");
