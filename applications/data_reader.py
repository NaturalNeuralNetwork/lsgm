# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import scipy.io.wavfile as wavfile
import torch
import torch.utils.data
import numpy as np
import warnings
import warnings
# from  wavmanager import memmap_manager
from functools import reduce

def update_mean(mean, num_items, data, f, offset):
    num_items_data   = reduce(lambda x, y: x * y, data.shape, 1.0);
    incremental_mean = np.add.reduce(f(data - offset).flatten()) / (num_items + num_items_data);
    mean             = mean * (num_items / (num_items_data + num_items)) + incremental_mean;
    num_items        = num_items + num_items_data;

    return mean, num_items;

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__( \
            self, \
            filelist, \
            dim, \
            length=16000, \
            normalize=False, \
            normalization_type="individual", \
            mean=None, \
            var=None, \
            spectrogram=False,\
            windowsize=50,\
            max_=None, \
            min_=None \
        ):
        self.filelist  = filelist;
        self.indices   = np.arange(len(filelist)).reshape((len(filelist),1));
        self.dim       = dim;
        self.length    = length;
        self.normalize = normalize;

        self.normalization_type = normalization_type;

        assert(normalization_type in ["global", "individual", "global_scale"]);

        if spectrogram is True:
            warnings.warn("To use spectrogram, will turn off normalization");
            self.normalize = False;

        self.spectrogram = spectrogram;
        self.windowsize  = windowsize;
        self.mean        = mean;
        self.var         = var;
        self.max_        = max_;
        self.min_        = min_;

        if normalize:
            if normalization_type == "global":
                running_sum  = 0;
                total_length = 0;

                if (mean is None) or (var is None):
                    for filename in self.filelist:
                        data          = np.array(wavfile.read(filename)[1].flatten(), dtype=np.float32);
                        running_sum  += np.add.reduce(data);
                        total_length += data.shape[0];

                    mean = running_sum / total_length;

                    running_square_sum = 0;

                    for filename in self.filelist:
                        data                 = np.array(wavfile.read(filename)[1].flatten(), dtype=np.float32);
                        running_square_sum  += np.add.reduce((data - mean)**2);

                    var  = running_square_sum / (total_length - 1);
                    var  = var ** (0.5);

                    print("Computed global mean and standard deviation %f, %f"%(mean, var));

                self.mean = mean;
                self.var  = var;
            elif normalization_type == "global_scale":
                if (max_ is None) or (min_ is None):
                    max_ = float(np.finfo(np.float32).min);
                    min_ = float(np.finfo(np.float32).max);

                    for filename in self.filelist:
                        data = np.array(wavfile.read(filename)[1].flatten(), dtype=np.float32)

                        max_data = float(np.amax(data));
                        min_data = float(np.amin(data));

                        if max_data > max_:
                            max_ = max_data;
                        
                        if min_data < min_:
                            min_ = min_data;

                    self.max_ = max_;
                    self.min_ = min_;

    def __getitem__(self, index):
        indices    = self.indices[index].flatten();

        filenames  = [self.filelist[i] for i in indices];

        tensors    = [];

        for filename in filenames:
            data = np.array(wavfile.read(filename)[1].flatten(), dtype=np.float32);

            if self.normalize:
                if self.normalization_type == "global_scale":
                    data = data / (self.max_ - self.min_);
                elif self.normalization_type == "individual":
                    data = data / (np.amax(data) - np.amin(data));
                else:
                    data = (data - self.mean) / self.var;

            if data.shape[0] < self.length:
                zeros = np.zeros(self.length - data.shape[0]);
                data  = np.concatenate((data, zeros), axis=0);
            else:
                data  = data[:self.length];

            if self.spectrogram:
                raise AttributeError("Come back for implementation please!");

            tensors.append(torch.from_numpy(data).float());

        tensors = torch.stack(tensors, dim=0);

        if not self.spectrogram:
            batch_size = tensors.size(0);
            length     = tensors.size(1);

            new_length = length // self.dim;

            if new_length * self.dim < length:
                new_length = new_length + 1;
                pad        = new_length * self.dim - length;
                zeros      = torch.zeros(batch_size, pad);
                tensors    = torch.cat((tensors, zeros), dim=1);

            tensors = tensors.contiguous().view(batch_size, new_length, self.dim);

        if len(indices) == 1:
            tensors = torch.squeeze(tensors, dim=0);

        return tensors;

    def __len__(self):
        return len(self.filelist);

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(\
            self,\
            filelist,\
            labels,\
            dim,\
            length=16000,\
            normalize=False,\
            normalization_type="individual",\
            mean=None,\
            var=None,\
            spectrogram=False,
            windowsize=50,\
            max_=None,
            min_=None,
        ):
        self.wavdata  = UnlabeledDataset( \
                            filelist, \
                            dim, \
                            length, \
                            normalize, \
                            normalization_type=normalization_type, \
                            mean=mean, \
                            var=var, \
                            spectrogram=spectrogram, \
                            windowsize=windowsize, \
                            max_=max_,
                            min_=min_,
                        );
        self.labels   = torch.Tensor(labels).long();
        assert(len(self.wavdata) == self.labels.size(0)), "Lengths are %d, %d"%(len(self.wavdata), self.labels.size(0));

    def __getitem__(self, index):
        return self.wavdata[index], self.labels[index];

    def __len__(self):
        return self.labels.size(0);

class training_dataset:
    def __init__(self, \
                train_list, \
                val_list, \
                train_labels=None, \
                val_labels=None, \
                batch_size=100, \
                dimension=200, \
                cuda=False, \
                normalize=False, \
                normalization_type="individual", \
                shuffle=True, \
                length=16000, \
                mean=None,\
                var=None, \
                spectrogram=False, \
                windowsize=50, \
                max_=None, \
                min_=None, \
        ):

        self.train_list   = [];
        self.val_list     = [];
        self.train_labels = train_labels;
        self.val_labels   = val_labels;
        self.mean         = mean;
        self.var          = var;
        self.max_         = max_;
        self.min_         = min_;

        if train_list is not None:
            with open(train_list, 'r') as trainhandle:
                for line in trainhandle:
                    self.train_list.append(line.rstrip());

        with open(val_list, 'r') as valhandle:
            for line in valhandle:
                self.val_list.append(line.rstrip());

        if (train_list is not None) and (train_labels is not None):
            self.training_set = LabeledDataset(\
                                    self.train_list, \
                                    train_labels, \
                                    dim                 = dimension, \
                                    normalize           = normalize, \
                                    length              = length, \
                                    normalization_type  = normalization_type, \
                                    mean                = self.mean, \
                                    var                 = self.var, \
                                    spectrogram         = spectrogram, \
                                    windowsize          = windowsize, \
                                    max_                = self.max_, \
                                    min_                = self.min_, \
                                );
            self.train_loader = torch.utils.data.DataLoader(self.training_set, batch_size, shuffle=shuffle, num_workers=0, drop_last=False, pin_memory=cuda);

            if normalization_type == "global":
                self.mean, self.var  = self.training_set.wavdata.mean, self.training_set.wavdata.var;

            if normalization_type == "global_scale":
                self.max_, self.min_ = self.training_set.wavdata.max_, self.training_set.wavdata.min_;

        if (train_list is not None) and (train_labels is None):
            self.training_set = UnlabeledDataset(\
                                    self.train_list, \
                                    dim                = dimension, \
                                    normalize          = normalize, \
                                    length             = length, \
                                    normalization_type = normalization_type, \
                                    mean               = self.mean, \
                                    var                = self.var, \
                                    spectrogram        = spectrogram, \
                                    windowsize         = windowsize, \
                                    max_                = self.max_, \
                                    min_                = self.min_, \
                                );
            self.train_loader = torch.utils.data.DataLoader(self.training_set, batch_size, shuffle=shuffle, num_workers=0, drop_last=False, pin_memory=cuda);

            if normalization_type == "global":
                self.mean, self.var = self.training_set.mean, self.training_set.var;

            if normalization_type == "global_scale":
                self.max_, self.min_ = self.training_set.max_, self.training_set.min_;

        if (val_list is not None) and (val_labels is not None):
            self.val_set      = LabeledDataset( \
                                    self.val_list, \
                                    val_labels, \
                                    dim                 = dimension, \
                                    normalize           = normalize, \
                                    length              = length, \
                                    normalization_type  = normalization_type, \
                                    mean                = self.mean, \
                                    var                 = self.var, \
                                    spectrogram         = spectrogram, \
                                    windowsize          = windowsize, \
                                    max_                = self.max_, \
                                    min_                = self.min_, \
                                );

            self.val_loader   = torch.utils.data.DataLoader(self.val_set, batch_size, shuffle=shuffle, num_workers=0, drop_last=False, pin_memory=cuda);

        if (val_list is not None) and (val_labels is None):
            self.val_set      = UnlabeledDataset( \
                                    self.val_list, \
                                    dim                 = dimension, \
                                    normalize           = normalize, \
                                    length              = length, \
                                    normalization_type  = normalization_type, \
                                    mean                = self.mean, \
                                    var                 = self.var, \
                                    spectrogram         = spectrogram, \
                                    windowsize          = windowsize, \
                                    max_                = self.max_, \
                                    min_                = self.min_, \
                                );

            self.val_loader   = torch.utils.data.DataLoader(self.val_set, batch_size, shuffle=shuffle, num_workers=0, drop_last=False, pin_memory=cuda);

        assert(hasattr(self, 'train_loader') or hasattr(self, 'val_loader')), "Training and validation data not provided!";

        if hasattr(self, 'train_loader'):
            self.__train      = True;
            self.__loader     = self.train_loader;
        else:
            self.__train      = False;
            self.__loader     = self.val_loader;

    def __iter__(self):
        return iter(self.__loader);

    def __len__(self):
        return len(self.__loader);

    def train(self, _train):
        self.__train  = _train;
        self.__loader = self.train_loader if _train is True else self.val_loader;

    def is_train(self):
        return self.__train;

class memmap_manager_dataset(torch.utils.data.Dataset):
    def __init__(
                self,
                memmap_item, 
                indices, 
                dimension, 
                mean=None, 
                var=None, 
                min_=None, 
                max_=None, 
                normalize=False, 
                normalization_type="global_scale"
        ):

        self.memmap_item = memmap_item;
        self.indices     = list(indices);
        self.dimension   = dimension;
        self.normalize   = normalize;
        self.ntype       = normalization_type;
        self.max_        = max_;
        self.min_        = min_;
        self.mean        = mean;
        self.var         = var;

        if self.normalize:
            if self.ntype == "global_scale":

                if (self.max_ is None) or (self.min_ is None):
                    print("Computing global max and min ... ");

                    self.max_ = float(np.finfo(np.float32).min);
                    self.min_ = float(np.finfo(np.float32).max);

                    for index in indices:
                        data = self.memmap_item[index];
                        max_ = float(np.amax(data.flatten()));
                        min_ = float(np.amin(data.flatten()));

                        if self.max_ < max_:
                            self.max_ = max_;

                        if self.min_ > min_:
                            self.min_ = min_;

                    print("Obtained global max and min ", self.max_, self.min_);

            elif self.ntype == "global":

                if (self.mean is None) or (self.var is None):
                    print("Computing global mean and standard deviation");

                    mean      = 0;
                    num_items = 0;

                    for index in indices:
                        data = self.memmap_item[index];
                        mean, num_items = update_mean(mean, num_items, data, lambda x : x, offset = 0);

                    var       = 0;
                    num_items = 0;

                    for index in indices:
                        data = self.memmap_item[index];
                        var, num_items = update_mean(var, num_items, data, lambda x : x**2, offset = mean);

                    var = var ** (0.5);

                    print("Obtained global mean and standard deviation", mean, var);

                self.mean = mean;
                self.var  = var;

    def __getitem__(self, index):
        indices = self.indices[index];
        npy     = self.memmap_item[indices];

        if len(npy.shape) == 1:
            batch_size = 1;
            length     = npy.shape[0];
        elif len(npy.shape) == 2:
            batch_size = npy.shape[0];
            length     = npy.shape[1];
        else:
            raise ValueError("Unknown shape!");

        new_length = length // self.dimension;

        if new_length * self.dimension < length:
            new_length += 1;

        data  = npy;

        if new_length * self.dimension > length:
            zeros = np.zeros((batch_size, new_length * self.dimension - length), dtype=np.float32);
            data  = np.concatenate((npy, zeros), axis=1);

        newdata = np.reshape(data, (batch_size, new_length, self.dimension));

        if self.normalize:
            if self.ntype == "global":
                newdata = (newdata - self.mean) / self.var;
            elif self.ntype == "global_scale":
                newdata = newdata / (self.max_ - self.min_);
            elif self.ntype == "individual":
                newdata = newdata / (np.amax(newdata, axis=(1,2), keepdims=True) - np.amin(newdata, axis=(1,2), keepdims=True));
            else:
                raise ValueError("Cannot normalize! Unknown normalization type " + str(self.ntype));

        return torch.squeeze(torch.from_numpy(newdata), dim=0);

    def __len__(self):
        return len(self.indices);

class training_dataset_memmap_manager:
    def __init__(self,
                memmap_filename, 
                total_num_sequences, 
                sample_length, 
                train_indices, 
                val_indices, 
                batch_size, 
                cuda,
                dimension,
                min_,
                max_,
                mean,
                var,
                normalize,
                normalization_type,
                shuffle=False,
        ):

        self.mean = mean;
        self.var  = var;
        self.min_ = min_;
        self.max_ = max_;

        self.dataset = np.memmap(
                           memmap_filename,
                           dtype=np.float32,
                           shape=(total_num_sequences, sample_length),
                           mode='r',
                       );

        self.train_dataset = memmap_manager_dataset(
                                self.dataset, 
                                train_indices,
                                dimension,
                                mean = self.mean,
                                var  = self.var,
                                min_ = self.min_,
                                max_ = self.max_,
                                normalize = normalize,
                                normalization_type = normalization_type,
                            );

        self.mean          = self.train_dataset.mean;
        self.var           = self.train_dataset.var;
        self.min_          = self.train_dataset.min_;
        self.max_          = self.train_dataset.max_;

        self.val_dataset   = memmap_manager_dataset(
                                self.dataset, 
                                val_indices,
                                dimension,
                                mean = self.mean,
                                var  = self.var,
                                min_ = self.min_,
                                max_ = self.max_,
                                normalize = normalize,
                                normalization_type = normalization_type,
                            );

        self.train_loader  = torch.utils.data.DataLoader(self.train_dataset, batch_size, shuffle=shuffle, num_workers=0, drop_last=False, pin_memory=cuda);
        self.val_loader    = torch.utils.data.DataLoader(self.val_dataset, batch_size, shuffle=shuffle, num_workers=0, drop_last=False, pin_memory=cuda);

        self.__train       = True;
        self.__loader      = self.train_loader;

    def train(self, _train):
        self.__train  = _train;
        self.__loader = self.train_loader if _train else self.val_loader;

    def __iter__(self):
        return iter(self.__loader);

    def __len__(self):
        return len(self.__loader);

    def is_train(self):
        return self.__train;

class DatasetNpy(torch.utils.data.Dataset):
    def __init__(self, npy, dim=88, zeropad=None):
        self.npy     = np.load(npy);
        self.maxlen  = max([item.shape[0] for item in self.npy]);
        self.dim     = dim;
        self.lengths = torch.Tensor([int(item.shape[0]) for item in self.npy]).int();
        self.zeropad = zeropad;

    def __getitem__(self, index):
        items     = self.npy[index];
        indices   = list(range(len(self.npy)))[index];
        num_items = 1 if type(indices) is int else len(indices);

        if self.dim == 0:
            return_array = np.zeros((self.maxlen,));
        else:
            return_array = np.zeros((self.maxlen, self.dim));

        if num_items > 1:
            if self.dim == 0:
                return_array = np.zeros((num_items, self.maxlen,));
            else:
                return_array = np.zeros((num_items, self.maxlen, self.dim));

            for i, item in enumerate(items):
                if self.zeropad == "prepend":
                    return_array[i,-len(item):] = item;
                else:
                    return_array[i,:len(item)] = item;
        else:
            if self.zeropad == "prepend":
                return_array[-len(items):] = items;
            else:
                return_array[:len(items)] = items;

        return torch.Tensor(return_array), self.lengths[index];

    def __len__(self):
        return self.npy.shape[0];

class DatasetNpyLabeled(torch.utils.data.Dataset):
    def __init__(self, npy, lab, dim=88, zeropad=None):
        self.dataset = DatasetNpy(npy, dim, zeropad);
        self.labels  = DatasetNpy(lab, 0, zeropad); # torch.Tensor(np.load(lab)).int();

    def __getitem__(self, index):
        data, length = self.dataset[index];
        labels, length_ = self.labels[index];

        return data, length, labels;

    def __len__(self):
        return len(self.dataset);

class training_dataset_npy_discrete_labeled:
    def __init__(
                self,
                train_npy,
                train_lab,
                val_npy,
                val_lab,
                batch_size,
                shuffle,
                cuda,
                dim,
                zeropad="append",
                num_workers=0,
                ):

        if train_npy is not None:
            traindataset      = DatasetNpyLabeled(train_npy, train_lab, dim, zeropad=zeropad);
            self.train_loader = torch.utils.data.DataLoader(traindataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False, pin_memory=cuda);

        if val_npy is not None:
            valdataset      = DatasetNpyLabeled(val_npy, val_lab, dim, zeropad=zeropad);
            self.val_loader = torch.utils.data.DataLoader(valdataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False, pin_memory=cuda);

        if hasattr(self, 'train_loader'):
            self.__loader = self.train_loader;
            self.__train  = True;
        elif hasattr(self, 'val_loader'):
            self.__loader = self.val_loader;
            self.__train  = False;
        else:
            raise ValueError("Should provide either train or validation datasets");

    def __iter__(self):
        return iter(self.__loader);

    def __len__(self):
        return len(self.__loader);

    def is_train(self):
        return self.__train;

    def train(self, _train):
        self.__train  = _train;
        self.__loader = self.train_loader if _train else self.val_loader;

class training_dataset_npy_discrete:
    def __init__(
                self,
                train_npy,
                val_npy,
                batch_size,
                shuffle,
                cuda,
                dim,
                zeropad="append",
                num_workers=0,
                ):

        if train_npy is not None:
            traindataset      = DatasetNpy(train_npy, dim, zeropad=zeropad);
            self.train_loader = torch.utils.data.DataLoader(traindataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False, pin_memory=cuda);

        if val_npy is not None:
            valdataset      = DatasetNpy(val_npy, dim, zeropad=zeropad);
            self.val_loader = torch.utils.data.DataLoader(valdataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False, pin_memory=cuda);

        if hasattr(self, 'train_loader'):
            self.__loader = self.train_loader;
            self.__train  = True;
        elif hasattr(self, 'val_loader'):
            self.__loader = self.val_loader;
            self.__train  = False;
        else:
            raise ValueError("Should provide either train or validation datasets");

    def __iter__(self):
        return iter(self.__loader);

    def __len__(self):
        return len(self.__loader);

    def is_train(self):
        return self.__train;

    def train(self, _train):
        self.__train  = _train;
        self.__loader = self.train_loader if _train else self.val_loader;

class DatasetTIMIT(torch.utils.data.Dataset):
    def __init__(self, npz, dim=200, slide_back=0, normalize=True, mean=None, std=None):
        self.dim        = dim;
        self.normalize  = normalize;
        self.slide_back = slide_back;

        self.vectors                = np.load(npz)['vectors'];
        self.labels                 = np.load(npz)['labels'];
        self.sequence_lengths       = np.load(npz)['sequence_lengths'];
        self.state_sequence_lengths = np.load(npz)['state_sequence_lengths'];

        # Determine global mean and std if not provided
        if normalize:
            if (mean is None) or (std is None):
                total   = 0;
                totalsq = 0;
                num     = 0;

                for length, sequence in zip(self.sequence_lengths, self.vectors):
                    total   += np.add.reduce(sequence.flatten());
                    totalsq += np.add.reduce((sequence**2).flatten());
                    num     += length;

                mean = total / num;
                std  = (totalsq / num - mean**2) ** 0.5;

                print("Computed global mean and std %f, %f"%(mean, std));

            self.mean = mean;
            self.std  = std;

    def __getitem__(self, index):
        data_        = self.vectors[index];
        labels       = self.labels[index];
        seqlengths   = self.sequence_lengths[index];
        statelengths = self.state_sequence_lengths[index];
        data         = data_;
        indices      = list(range(len(self)))[index];
        num_items    = 1 if type(indices) is int else len(indices);

        sequence_lengths = seqlengths if num_items > 1 else seqlengths;

        if self.normalize:
            data = (data_ - self.mean) / self.std;

        if num_items > 1:
            length    = data.shape[1];
        else:
            length    = data.shape[0];

        num_steps = length // self.dim;

        if num_steps * self.dim < length:
            num_steps += 1;

        num_zeros = num_steps * self.dim - length;

        if num_items > 1:
            data = np.concatenate((data, np.zeros((num_items, num_zeros)).astype(np.float32)), axis=1);
            data = np.reshape(data, (num_items, num_steps, self.dim));

            sequence_lengths_ = sequence_lengths // self.dim;

            sequence_lengths = np.where(sequence_lengths_ * self.dim < sequence_lengths, sequence_lengths_+1, sequence_lengths_);

            sequence_lengths = torch.Tensor(sequence_lengths);
            statelengths     = torch.Tensor(statelengths);
        else:
            data = np.concatenate((data, np.zeros((num_zeros)).astype(np.float32)), axis=0);
            data = np.reshape(data, (num_steps, self.dim));

            sequence_lengths_ = sequence_lengths // self.dim;

            if sequence_lengths_ * self.dim < sequence_lengths:
                sequence_lengths = sequence_lengths_ + 1;
            else:
                sequence_lengths = sequence_lengths_;

        return torch.Tensor(data).float(), torch.Tensor(labels).long(), sequence_lengths, statelengths;

    def __len__(self):
        return len(self.vectors); 

class DatasetTIMITfbank(torch.utils.data.Dataset):
    def __init__(self, npz, dim=40, slide_back=0, normalize=True, normalization_type="global", mean=None, std=None):
        self.dim        = dim;
        self.normalize  = normalize;
        self.slide_back = slide_back;

        self.vectors                = np.load(npz)['vectors'];
        self.labels                 = np.load(npz)['labels'];
        self.sequence_lengths       = np.load(npz)['sequence_lengths'];
        self.state_sequence_lengths = np.load(npz)['state_sequence_lengths'];
        self.normalization_type     = normalization_type;

        # Determine global mean and std if not provided
        if normalize:
            if normalization_type == "global":
                if (mean is None) or (std is None):
                    total   = 0;
                    totalsq = 0;
                    num     = 0;

                    for length, sequence in zip(self.sequence_lengths, self.vectors):
                        total   += np.add.reduce(sequence.flatten());
                        totalsq += np.add.reduce((sequence**2).flatten());
                        num     += length * self.dim;

                    mean = total / num;
                    std  = (totalsq / num - mean**2) ** 0.5;

                    print("Computed global mean and std %f, %f"%(mean, std));

                self.mean = mean;
                self.std  = std;
            elif normalization_type == "frame_wise":
                if (mean is None) or (std is None):
                    """ This one doesn't care whether you provide a mean and a variance """
                    total   = 0;
                    totalsq = 0;
                    num     = 0;

                    for length, sequence in zip(self.sequence_lengths, self.vectors):
                        total   += np.add.reduce(sequence, axis=0, keepdims=True); # Add along the sequence length (preserve dimension)
                        totalsq += np.add.reduce(sequence**2, axis=0, keepdims=True);
                        num     += length;

                    mean = total / num;
                    std  = (totalsq / num - mean ** 2) ** 0.5;

                    print("Found framewise mean", mean, "framewise standard deviation", std);

                self.mean = mean;
                self.std  = std;

    def __getitem__(self, index):
        vectors      = self.vectors[index];
        labels       = self.labels[index];
        seqlengths   = self.sequence_lengths[index];
        statelengths = self.state_sequence_lengths[index];

        if self.normalize:
            if self.normalization_type == "frame_wise":
                mean = std = None;

                if len(vectors.shape) > 2:
                    mean = np.expand_dims(self.mean, 0);
                    std  = np.expand_dims(self.std, 0);
                else:
                    mean = self.mean;
                    std  = self.std;

                vectors = (vectors - mean) / std;
            else:
                vectors = (vectors - self.mean) / self.std;

        return torch.Tensor(vectors).float(), torch.Tensor(labels).long(), seqlengths, statelengths;

    def __len__(self):
        return len(self.vectors);

class TIMIT:
    def __init__(self, train_npz, val_npz, dim=200, slide_back=100, normalize=True, normalization_type="global", mean=None, std=None, batch_size=10, fbank=False, normalize_separately=False):
        DataType   = DatasetTIMIT;

        if fbank:
            DataType = DatasetTIMITfbank;

        train      = DataType(train_npz, dim, slide_back, normalize, normalization_type, mean, std);
        self.mean  = train.mean if (not normalize_separately) else None;
        self.std   = train.std if (not normalize_separately) else None;
        val        = DataType(val_npz, dim, slide_back, normalize, normalization_type, self.mean, self.std);

        self.train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, num_workers=0, drop_last=False, pin_memory=True);
        self.val_loader   = torch.utils.data.DataLoader(val, batch_size, shuffle=True, num_workers=0, drop_last=False, pin_memory=True);
        
        self.__train  = True;
        self.__loader = self.train_loader;

    def __iter__(self):
        return iter(self.__loader);

    def __len__(self):
        return len(self.__loader);

    def is_train(self):
        return self.__train;

    def train(self, _train):
        self.__train  = _train;
        self.__loader = self.train_loader if _train else self.val_loader;

def arrange_data(sequences, labels, sequence_lengths):
    batch_size = sequences.size(0);
    max_length = sequences.size(1);

    # Frist concatenate labels into sequences
    combined = torch.stack((sequences, labels), dim=2);

    # Now obtain the sorted ordering of sequence_lengths
    sorted_lengths, sorting_order = sequence_lengths.sort();

    # Re-order the data
    reordered_data = combined[sorting_order];

    return reordered_data, sorted_lengths;
