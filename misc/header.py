import torch
import numpy
import scipy.stats
from functools import reduce
import numpy as np

sigmoid     = torch.nn.functional.sigmoid;
relu        = torch.nn.functional.relu;
tanh        = torch.nn.functional.tanh;
softmax     = torch.nn.functional.softmax;

nonlinear   = {'sigmoid':sigmoid, 'ReLu':relu, 'tanh':tanh, 'softmax':lambda x : softmax(x, dim=1), 'noop':lambda x : x};

small_value = np.nextafter(0,1);

def weight_variable(shape):
    num_elems = reduce(lambda x, y: x * y, shape, 1);
    elems     = scipy.stats.truncnorm.rvs(-0.1,0.1,size=num_elems).reshape(shape);

    return torch.from_numpy(elems).float();
