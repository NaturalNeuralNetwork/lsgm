# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import torch
import numpy as np
import warnings

SMALL = float(np.finfo(np.float32).tiny);

try:
    profile
except NameError:
    profile = lambda x : x

class ClippedLeakyReLU(torch.nn.Module):
    def __init__(self, leakiness, min_threshold, max_threshold):
        super().__init__();

        self.leakiness     = leakiness;
        self.min_threshold = min_threshold;
        self.max_threshold = max_threshold;

    def forward(self, tensor):
        return torch.clamp(
                torch.nn.functional.leaky_relu(tensor, negative_slope=self.leakiness),
                min = self.min_threshold,
                max = self.max_threshold
        );

class Activations:
    def __init__(self, activation_type):
        self.activation_type = activation_type;

    def set_activation_type(self, activation_type):
        self.activation_type = activation_type;

    def activation(self):
        if self.activation_type == "relu":
            return torch.nn.functional.relu;
        elif self.activation_type == "leaky_relu":
            return lambda tensor : torch.nn.functional.leaky_relu(tensor, negative_slope=1/3);
        elif self.activation_type == "clipped_leaky_relu":
            return lambda tensor : \
                            torch.clamp(
                                torch.nn.functional.leaky_relu(tensor, negative_slope=1/3),
                                min = -3,
                                max = 3
                            );
        elif self.activation_type == "softplus":
            return torch.nn.functional.softplus;

    def Activation(self):
        if self.activation_type == "relu":
            return torch.nn.ReLU();
        elif self.activation_type == "leaky_relu":
            return torch.nn.LeakyReLU(negative_slope=1/10.0);
        elif self.activation_type == "clipped_leaky_relu":
            return ClippedLeakyReLU(leakiness=1/3, min_threshold=-3, max_threshold=3);
        elif self.activation_type == "softplus":
            return torch.nn.Softplus();

    def scale(self):
        if self.activation_type == "clipped_leaky_relu":
            return 1e-3;
        else:
            return 1e-1;

    def bias(self):
        if self.activation_type == "clipped_leaky_relu":
            return 0.0;
        elif self.activation_type == "leaky_relu":
            return 0.0;
        else:
            return 0.1;

def soft_threshold(tensor, threshold):
    """ Verified using a numpy plot """
    return -torch.nn.functional.softplus(-(tensor-threshold)) + threshold;

def log_add(tensor, dim, keepdim=False):
    """What you are exponentiating may be too large. So let's take a step back and think ...
    log(a + b) = log(b(1 + a/b)) = log(b) + log(1 + a/b) = log(b) + log(1 + exp(log(a) - log(b)))
    Now, we need to ensure that log(a) - log(b) is within range of exponentiation,
    so we need to choose a and b a appropriately. This means, b should always be the larger quantity.
    Now, what do we do when we need to do this for more than two numbers. Then we simply pick the largest
    quantity. So, for n numbers where a_l is the largest, this will look like:
    log[a_1 + ... a_n] = log[a_l (1 + a_1 + ... a_n)/a_l] = log(a_l) + log(1 + a_1/a_l + ... a_n/a_l)
    This will look like log(a_) + log(1 + exp(log a_1 - log a_l) + exp(log a_2 - log a_l) + ... exp(log a_n - log a_l))
    None of the exponential terms overflow since a_l is the largest."""
    max_value, _ = torch.max(tensor, dim=dim, keepdim=True);
    tensor_      = tensor - max_value;

    if keepdim:
        result = torch.log(torch.sum(torch.exp(tensor_), dim=dim, keepdim=True)) + max_value;
    else:
        result = torch.log(torch.sum(torch.exp(tensor_), dim=dim)) + torch.squeeze(max_value, dim=dim);

    return result;

def rnn_init(rnn, num_layers):
    for i in range(num_layers):
        weight_ih_l = getattr(rnn, 'weight_ih_l' + str(i));
        weight_hh_l = getattr(rnn, 'weight_hh_l' + str(i));
        bias_ih_l   = getattr(rnn, 'bias_ih_l' + str(i));
        bias_hh_l   = getattr(rnn, 'bias_hh_l' + str(i));

        weight_ih_l.data.normal_(0.0,0.1);
        weight_hh_l.data.normal_(0.0,0.1);
        bias_ih_l.data.fill_(0.1);
        bias_hh_l.data.fill_(0.1);

def create_dense_layer(
        num_inputs,
        num_outputs,
        dropout,
        activation=None,
        scale=1e-1,
        bias=None,
        package=False,
    ):

    if bias is None:
        bias = torch.Tensor(num_outputs).float();
        bias.fill_(0.1);
    elif type(bias) is float:
        bias_ = torch.Tensor(num_outputs).float();
        bias_.fill_(bias);
        bias = bias_;

    dense = torch.nn.Linear(num_inputs, num_outputs);

    dense.weight.data.normal_(0.0,scale);
    dense.bias.data[:] = bias;

    dropout = torch.nn.Dropout(p=dropout);

    if activation is not None:
        items = [dropout, dense, activation];
    else:
        items = [dropout, dense];

    if package:
        items = torch.nn.Sequential(*items);

    return items;

def check_tensor(tensor, name):
    tensor_np = tensor.cpu().data.numpy();

    if np.logical_or.reduce(np.isnan(tensor_np).flatten(), axis=0) or \
       (not np.logical_and.reduce(np.isfinite(tensor_np).flatten(), axis=0)):
        raise ValueError("%s tensor is not finite!"%name);

@profile
def normalized_likelihoods(sequences, transitions, emissions, start_transitions=None):
    transitions      = torch.unbind(transitions, dim=0);
    emissions        = torch.unbind(torch.exp(emissions), dim=0);
    sequence_length  = sequences.size(0);
    batch_size       = sequences.size(1);
    dimension        = sequences.size(2);
    forward_matrix   = [];
    forward_norms    = [];

    if start_transitions is None:
        start_transitions = torch.nn.functional.softmax(
                                torch.autograd.Variable(
                                    torch.zeros(1,emissions[0].size(1))
                                ),
                            dim=1);

        if sequences.is_cuda:
            start_transitions = start_transitions.cuda(device=sequences.get_device());

    for i, e in enumerate(emissions):
        t = transitions[i-1];

        forward_entry = None;

        if i == 0:
            """ Initialization """
            # [1 x num_states] * [batch_size x num_states], torch broadcast
            forward_entry     = start_transitions * e;
                                # torch.log(start_transitions + \
                                #     float(np.finfo(np.float32).tiny)) + e;
        else:
            """ Recursion """
            forward_entry     = torch.squeeze(
                                    torch.matmul(
                                        torch.unsqueeze(forward_matrix[i-1],dim=1),
                                        t,
                                    ),
                                    dim=1
                                ) * e;

        normalization = torch.sum(forward_entry, dim=1, keepdim=True);
        forward_matrix.append(forward_entry/normalization);
        forward_norms.append(torch.squeeze(normalization, dim=1));

    forward_matrix = torch.stack(forward_matrix, dim=0);
                        # Sequence length x batch x num_states

    norms_log      = torch.log(torch.stack(forward_norms, dim=0));
                        # Sequence length x batch

    forward_log    = torch.log(forward_matrix) + \
                        torch.unsqueeze(norms_log, dim=2);

    return norms_log, forward_log; 

@profile
def likelihoods(sequences, transitions, emissions, start_transitions=None):
    transitions      = torch.unbind(transitions, dim=0);
    emissions        = torch.unbind(emissions, dim=0);
    sequence_length  = sequences.size(0);
    batch_size       = sequences.size(1);
    dimension        = sequences.size(2);
    forward_matrix   = [];

    # print('\nsmall=',SMALL,'\n');
    # assert(1==2);

    if start_transitions is None:
        start_transitions = torch.nn.functional.softmax(
                                torch.autograd.Variable(
                                    torch.zeros(1,emissions[0].size(1))
                                ),
                            dim=1);

        if sequences.is_cuda:
            start_transitions = start_transitions.cuda(device=sequences.get_device());

    for i, e in enumerate(emissions):
        t = transitions[i-1];

        forward_entry = None;

        if i == 0:
            """ Initialization """
            # [1 x num_states] + [batch_size x num_states], torch broadcast
            forward_entry     = torch.log(start_transitions + \
                                    SMALL) + e;
        else:
            """ Recursion """
            """ Be wary of dynamic range overflow (use log_add function) """
            # Step 1: Add forward terms to the source states in the transition matrix
            log_sum       = torch.log(t + SMALL) + \
                            torch.unsqueeze(forward_matrix[i-1], dim=2);
            # Step 2: Add log-domain values along the source domain (dim=1)
            fwdtr         = log_add(log_sum, dim=1);
            # Step 3: Add to log emissions
            forward_entry = fwdtr + e;

        forward_matrix.append(forward_entry);

    forward_matrix = torch.stack(forward_matrix, dim=0);

    """ Compute \log P[X_{1:i}] = \log \sum_k P[X_{1:i},\pi_i=k] """
    prefix_prob = log_add(forward_matrix, dim=2); 

    """ Shift prefix_prob by one position """ 
    zeros = torch.autograd.Variable(torch.zeros(1,batch_size).float());

    if sequences.is_cuda:
        zeros = zeros.cuda(device=sequences.get_device(), non_blocking=True);

    prefix_prob_ = torch.cat((zeros, prefix_prob), dim=0);

    """ Compute \log P[X_{1:i}] - \log P[X_{1:i-1}] """
    return prefix_prob - prefix_prob_[:-1], forward_matrix; 

def log_backward(
    sequences,
    transitions,
    emissions,
    start_transitions = None,
    lengths = None
):
    transitions      = torch.unbind(transitions, dim=0);
    emissions        = torch.unbind(emissions, dim=0);
    sequence_length  = sequences.size()[0];
    batch_size       = sequences.size()[1];
    dimension        = sequences.size()[2];
    num_states       = emissions[0].size(1);

    if start_transitions is None:
        start_transitions = torch.nn.functional.softmax(
                                torch.autograd.Variable(
                                    torch.zeros(1,num_states)
                                ),
                            dim=1);

        if sequences.is_cuda:
            start_transitions = start_transitions.cuda(device=sequences.get_device());

    backward_matrix   = [];
    prob              = None;

    """ First, we need to right align transitions and emissions """
    def right_align_array(array, array_length, individual_lengths):
        if individual_lengths is not None:
            stacked_array = torch.stack(array, dim=0);
            target_array  = torch.autograd.Variable(
                                torch.zeros(stacked_array.size())
                            );

            assert(stacked_array.size(0) == array_length);

            if stacked_array.is_cuda:
                target_array = target_array.cuda(device=stacked_array.get_device());

            for i, l in enumerate(individual_lengths):
                start = array_length - l;
                target_array[start:,i] = stacked_array[:l,i];

            return torch.unbind(target_array, dim=0);
        else:
            return array;

    # Keep the original emissions for convenience
    orig_emissions = emissions;
    transitions    = right_align_array(transitions, sequence_length, 
                                                                lengths);
    emissions      = right_align_array(emissions, sequence_length, 
                                                                lengths);

    for i in reversed(range(len(emissions))):
        """
        Computing \beta_i(k) = \sum_j \beta_{i+1}(j) e_{i+1}(j) \gamma_{i}(k,j)
        """
        t = transitions[i];
        e = emissions[i+1] if i+1 < len(emissions) else None;

        if i == len(emissions) - 1:
            """ Initialization """
            backward_entry = torch.autograd.Variable(
                                torch.zeros(batch_size, num_states)
                             );
                # torch.log(t[:,:,-1] + 
                    # float(np.finfo(np.float32).tiny)); # [batch_size x num_states]

            if sequences.is_cuda:
                backward_entry = backward_entry.cuda(
                                    device=sequences.get_device(), non_blocking=True
                                );
        else:
            """ Add emissions and transitions """
            te = torch.log(t + SMALL) + \
                                                torch.unsqueeze(e, dim=1);
                                            # [batch_size x num_states x num_states]

            """ Add the backward entry (pre matrix multiply) """
            backward_pre = te + torch.unsqueeze(backward_matrix[-1], dim=1);

            """ Use log-add (effecting matrix multiply without overflow) """
            backward_entry = log_add(backward_pre, dim=2);

        backward_entry = \
            backward_entry.contiguous().view(batch_size, num_states);

        backward_matrix.append(backward_entry);

    """ Right align backward matrix and reverse """
    backward_matrix = list(
                       reversed(
                        right_align_array(
                            backward_matrix, 
                            sequence_length, 
                            lengths
                        )
                       )
                      );

    """
    Termination.

    t - start probability
    e - emission probability of the first symbol in the sequence (0-th with 0-based numbering)
    b - backward matrix after computing l backward iterations (or backward matrix at position 0)

    Let's say we have zero-padded the initial parts of the sequences. If all the sequences are represented with
    length L, but the actual sequence length is 'l', then for backward we are looking at the (L-l)-th item, and
    for emissions we are looking at (L-l)-th item as well.
    """

    t = start_transitions.contiguous().view(1, num_states).expand(batch_size, num_states).contiguous();
    b = backward_matrix[0];
    e = orig_emissions[0];

    """ Termination """
    prob = log_add(torch.log(t + SMALL) + e + b, dim=1);
        
    backward_matrix = torch.stack(backward_matrix, dim=0); # [sequence_length x batch_size x num_states]

    return prob, backward_matrix;

def multimodal_entropy(
    forward_matrix,
    transitions,
    p_emissions = None,
    lengths = None,
    state_fraction = 1.0,
    emit_fraction = 1e-3,
):
    """
    Compute entropy of state distribution at every time step as -\sum P_s \log (P_s)
    P_s = P[s_t|X_{1:t-1}]

    =\sum_{s_{t-1}} P[s_{t-1}|X_{1:t-1}] P[s_t|s_{t-1},X_{1:t-1}]
    """
    sequence_length = forward_matrix.size(0);
    batch_size      = forward_matrix.size(1);
    num_states      = forward_matrix.size(2);

    marginal    = log_add(forward_matrix, dim=2, keepdim=True).expand(
                        sequence_length, batch_size, num_states
                    );
    conditional = forward_matrix - marginal;
                    # sequence_length x batch_size x num_states

    # transitions are [sequence_length x batch_size x num_states x num_states]
    # Make conditional of the same dimension
    conditional  = torch.unsqueeze(conditional, dim=3);

    distribution = log_add(conditional + transitions, dim=2);

    if lengths is not None:
        for i, l in enumerate(lengths):
            if sequence_length > l:
                distribution[l:,i] = 0.0;

    entropy = state_fraction * \
                torch.sum(distribution * torch.exp(distribution), dim=2);

    if p_emissions is None: return torch.sum(entropy, dim=0);

    # Next compute Jensen-Shannon divergence between state-pair emissions
    # p_emissions is [sequence_length x batch_size x dimension x num_states]
    js = 0;

    p_emissions = torch.exp(p_emissions);

    for i in range(num_states-1):
        for j in range(i+1,num_states):
            if i == j: continue;

            e1  = p_emissions[:,:,:,i];
            e2  = p_emissions[:,:,:,j];
            e3  = (e1 + e2) / 2;
            e1_ = 1 - e1;
            e2_ = 1 - e2;
            e3_ = 1 - e3;

            js += (e1 * torch.log(e1/e3) + e1_ * torch.log(e1_/e3_) + \
                   e2 * torch.log(e2/e3) + e2_ * torch.log(e2_/e3_))/2;

    entropy += emit_fraction * torch.sum(js, dim=2);
    
    return torch.sum(entropy, dim=0);

def state_entropy(forward_matrix, lengths = None):
    """
    Compute state entropy as -\sum P_s \log (P_s)
    where P_s is P[s_t|x_{1:t}]
    """
    sequence_length = forward_matrix.size(0);
    batch_size      = forward_matrix.size(1);
    num_states      = forward_matrix.size(2);

    marginal    = log_add(forward_matrix, dim=2, keepdim=True).expand(
                        sequence_length, batch_size, num_states
                    );
    conditional = forward_matrix - marginal;

    entropy_    = -torch.exp(conditional) * conditional;
                    # [ sequence_length, batch_size, num_states ]

    if lengths is not None:
        for i, l in enumerate(lengths):
            if sequence_length > l:
                entropy_[l:,i] = 0.0;

    entropy     = torch.sum(torch.sum(entropy_, dim=2), dim=0);

    return entropy;

class lsgm(torch.nn.Module):
    def __init__(
        self,
        num_stochastic_nodes    = 4,
        num_recurrent_layers    = 4,
        num_dense_layers        = 4,
        num_emission_layers     = 2,
        embedding_dim           = 256,
        num_recurrent_units     = 256,
        num_dense_units         = 128,
        dimension               = 200,
        dropout                 = 0.5,
        order                   = 1,
        dense_activations       = "softplus",
        use_emission_activation = False,
        testmode                = False,
        regularization_rate     = 0,
        tbptt                   = False,
        tbptt_length            = 100,
        entropy_type            = "forward",
        emission_type           = "binary",
    ):
        super().__init__();

        """ Store all arguments """
        self.num_stochastic_nodes    = num_stochastic_nodes;
        self.num_recurrent_layers    = num_recurrent_layers;
        self.num_recurrent_units     = num_recurrent_units;
        self.num_dense_layers        = num_dense_layers;
        self.num_dense_units         = num_dense_units;
        self.dimension               = dimension;
        self.dropout                 = dropout;
        self.dense_activations       = dense_activations;
        self.ActivationObject        = Activations(dense_activations);
        self.order                   = order;
        self.testmode                = testmode;
        self.regularization_rate     = regularization_rate;
        self.embedding_dim           = embedding_dim;
        self.num_emission_layers     = num_emission_layers;
        self.tbptt                   = tbptt;
        self.tbptt_length            = tbptt_length;
        self.entropy_type            = entropy_type;
        self.emission_type           = emission_type;

        assert(entropy_type in ["forward", "posterior", "multimodal"]);

        """ Recurrent layers """
        self.recurrent_layers = torch.nn.GRU(
                                   input_size  = dimension,
                                   hidden_size = num_recurrent_units,
                                   num_layers  = num_recurrent_layers,
                                   dropout     = dropout
                                );
        rnn_init(self.recurrent_layers, num_recurrent_layers);

        """ Recurrent network initialization state parameter """
        self.recurrent_init = torch.nn.Parameter(
                              torch.Tensor(num_recurrent_layers,1,num_recurrent_units)
                              );
        self.recurrent_init.data.normal_(0.0,0.1);

        """ Dense layers """
        dense_layers = [];
        num_inputs   = num_recurrent_units;
        num_outputs  = num_recurrent_units;
        
        for i in range(num_dense_layers):
            num_outputs   = num_dense_units;
            dense_layers += create_dense_layer(
                                num_inputs,
                                num_outputs,
                                dropout,
                                activation=self.ActivationObject.Activation(),
                                scale=self.ActivationObject.scale(),
                                bias=self.ActivationObject.bias(),
                                package=False,
                            );
            num_inputs    = num_outputs;

        self.dense_layers = torch.nn.Sequential(*dense_layers);

        """ State embeddings """
        self.state_embedding = torch.nn.Embedding(
                                num_stochastic_nodes,
                                embedding_dim
                              );

        """ Emission layer """
        num_emission_inputs  = num_outputs + embedding_dim;
        num_emission_outputs = dimension if emission_type == "binary" else 2 * dimension;
        num_emission_intermediate = 2 * max(num_emission_inputs, num_emission_outputs);

        if use_emission_activation:
            warnings.warn("Emission activation is deprecated");
        else:
            emission_layers = [];

            for i in range(self.num_emission_layers-1):
                emission_layers.append(
                    create_dense_layer(
                        num_emission_inputs,
                        num_emission_intermediate,
                        dropout = dropout,
                        activation=self.ActivationObject.Activation(),
                        scale = self.ActivationObject.scale(),
                        bias=self.ActivationObject.bias(),
                        package=True,
                    )
                );

            emission_layers.append(
                create_dense_layer(
                   num_emission_intermediate if self.num_emission_layers > 1 else num_emission_inputs,
                   num_emission_outputs,
                   dropout=0,
                   bias=0,
                   scale=1e-3,
                   activation=None,
                   package=True,
                )
            );

            self.emission_layer = torch.nn.Sequential(*emission_layers);

        """ Stochastic layers """
        # The stochastic layers
        num_stochastic_inputs   = num_outputs;
        power                   = 2 if (self.order == 1) else 1;

        num_stochastic_outputs  = num_stochastic_nodes ** power;
        num_intermediate        = 2 * max(num_stochastic_inputs, num_stochastic_outputs);

        layer1 = create_dense_layer(
                    num_stochastic_inputs,
                    num_intermediate,
                    dropout = dropout,
                    activation = self.ActivationObject.Activation(),
                    scale = self.ActivationObject.scale(),
                    bias = self.ActivationObject.bias(),
                    package = True
                );

        layer2 = create_dense_layer(
                    num_intermediate,
                    num_stochastic_outputs,
                    dropout=0,
                    scale=1e-3,
                    bias=0.0,
                    package=True,
                );

        self.stochastic_layers = [torch.nn.Sequential(layer1, layer2)];
        setattr(self, 'stochastic_layer0', self.stochastic_layers[0]);

        """ Buffer for indexing state embeddings """
        embedding_inputs = torch.LongTensor(
                                [n for n in range(
                                           self.num_stochastic_nodes
                                        )
                                ],
                            );

        self.register_buffer('embedding_inputs', embedding_inputs);

    """ Initial state of the RNN """
    def __init_state(self, batch_size):
        init_shape = (      self.recurrent_init.size(0),
                            batch_size,
                            self.recurrent_init.size(2)
                     );

        return self.recurrent_init.expand(init_shape);

    @profile
    def forward(self, sequences, lengths = None):
        sequences       = torch.transpose(sequences, 0, 1);

        sequence_length = sequences.size(0);
        batch_size      = sequences.size(1);
        dimension       = sequences.size(2);

        """ Prepend a zero """
        zeros  = torch.autograd.Variable(torch.zeros(1,batch_size,dimension).float());

        if sequences.is_cuda:
            zeros = zeros.cuda(non_blocking=True, device=sequences.get_device());

        rnn_input = torch.cat((zeros, sequences), dim=0);

        """ Send sequences through recurrent layers and dense layers """
        init_state  = self.__init_state(batch_size).contiguous();
        rnn_outputs = None;

        if self.tbptt:
            n_chunks = (sequence_length + 1) // self.tbptt_length;

            rnn_outputs = [];

            if n_chunks * self.tbptt_length < (sequence_length + 1):
                n_chunks += 1;

            chunks = torch.chunk(rnn_input, chunks=n_chunks, dim=0);

            for c_, c in enumerate(chunks):
                if c_ == 0:
                    o_, s_ = self.recurrent_layers(c, init_state);
                else:
                    o_, s_ = self.recurrent_layers(
                                c, 
                                torch.autograd.Variable(s_.data)
                             );

                rnn_outputs.append(o_);

            rnn_outputs = torch.cat(rnn_outputs, dim=0);
        else:
            rnn_outputs, state = self.recurrent_layers(rnn_input, init_state);

        dense_outputs = self.dense_layers(rnn_outputs);

        """ Compute emission layer """
        num_nodes = self.num_stochastic_nodes;
        embedding_inputs = torch.autograd.Variable(self.embedding_inputs);

        state_ids = torch.unsqueeze(
                        self.state_embedding(embedding_inputs), dim=0
                    ).expand(
                        batch_size, 
                        num_nodes,
                        self.embedding_dim
                    ).contiguous();

        ####
        state_ids = torch.unsqueeze(state_ids, dim=0).expand(
                        sequence_length+1,
                        batch_size,
                        num_nodes,
                        self.embedding_dim,
                    );

        dense_outputs_ = torch.unsqueeze(
                            dense_outputs,
                            dim=2
                        ).expand(
                            sequence_length+1,
                            batch_size,
                            num_nodes,
                            self.num_dense_units,
                        );

        dense_outputs_ = torch.cat((dense_outputs_[:-1],state_ids[:-1]), dim=3);

        emission_layer_inputs    = dense_outputs_;
        emission_layer_outputs   = self.emission_layer(emission_layer_inputs);
        emission_layer_outputs   = torch.transpose(emission_layer_outputs, 2, 3);

        """ Compute each stochastic layer on the NN side """
        stochastic_layer_outputs = self.stochastic_layers[0](dense_outputs);

        # Function to separate out the transition parameters
        def extract_transitions(params, regularization_rate = 0):
            # Remove the first item from params (corresponding to 0 input)
            params = params[1:];

            if self.order == 1:
                transition_params = params.view(
                                              sequence_length,
                                              batch_size,
                                              1,
                                              self.num_stochastic_nodes,
                                              self.num_stochastic_nodes,
                                          );
            elif self.order == 0:
                transition_params = params.view(
                                              sequence_length,
                                              batch_size,
                                              1,
                                              1,
                                              self.num_stochastic_nodes,
                                          ).expand(
                                              sequence_length,
                                              batch_size,
                                              1,
                                              self.num_stochastic_nodes,
                                              self.num_stochastic_nodes,
                                          );
            else:
                raise AttributeError("Only orders 0 and 1 supported!");

            if (regularization_rate > 0) and (self.training):
                mask = torch.zeros(transition_params.size());

                if transition_params.is_cuda:
                    mask = mask.cuda(non_blocking=True, 
                                    device=transition_params.get_device());

                mask = torch.autograd.Variable(
                                mask.bernoulli_(regularization_rate)
                       ) * float(np.finfo(np.float32).min);

                transition_params += mask;
                
            transitions = torch.nn.functional.softmax\
                                (transition_params, dim=4);

            return torch.unbind(transitions, dim=2);

        # First compute emission distributions
        graph_parameters  = emission_layer_outputs;

        if self.emission_type == "binary":
            emission_params   = torch.nn.functional.sigmoid(
                                graph_parameters.view(
                                                sequence_length,
                                                batch_size,
                                                dimension,
                                                1,
                                                self.num_stochastic_nodes,
                                            )); # order helps broadcast
            emission_dist     = torch.distributions.bernoulli.Bernoulli(probs=emission_params);
        else:
            mean, std         = torch.chunk(graph_parameters.view(
                                    sequence_length,
                                    batch_size,
                                    2 * self.dimension,
                                    1,
                                    self.num_stochastic_nodes,
                                ), chunks=2, dim=2);
            emission_dist     = torch.distributions.normal.Normal(mean, torch.sqrt(torch.exp(std)+1e-4));

        p_emissions       = emission_dist.log_prob(
                                torch.unsqueeze(
                                    torch.unsqueeze(sequences, dim=3),
                                    dim=4,
                                ) # Enable broadcast
                            );

        emissions         = torch.unbind(
                                torch.sum(
                                    p_emissions,
                                dim=2), # Sum along 'dimension'
                            dim=2); # Unbind across graphs (only one graph)

        p_emissions       = torch.unbind(p_emissions, dim=3);
                                # Unbind across graphs (only one graph)

        Likelihoods       = emissions;

        entropy_measures  = [];

        """ Function for determining posterior entropy """
        def posterior_entropy(f_, b_, prob_, lengths):
            fb        = f_ + b_;
            posterior = fb - torch.unsqueeze(
                                torch.unsqueeze(
                                    prob_,
                                    dim=0
                                ),
                                dim=2
                            ).expand(fb.size());

            if lengths is not None:
                for i_, l in enumerate(lengths):
                    if posterior.size(0) > l:
                        posterior[l:,i_] = 0.0;

            entropy = torch.sum(-posterior * torch.exp(posterior), dim=2);

            return torch.sum(entropy, dim=0);

        # Next, compute the stochastic graph
        graph_params = stochastic_layer_outputs;
        transitions  = extract_transitions(graph_params,
                                    regularization_rate=self.regularization_rate)[0];
        l_, f_       = likelihoods(sequences, transitions, Likelihoods[0]);

        if self.entropy_type == "forward":
            entropy = state_entropy(f_, lengths);
        elif self.entropy_type == "multimodal":
            entropy = multimodal_entropy(
                            f_,
                            transitions,
                            p_emissions = p_emissions[0],
                            lengths = lengths,
                        );
        else:
            prob_, b_ = log_backward(
                            sequences,
                            transitions,
                            Likelihoods[0],
                            lengths = lengths
                        );

            entropy = posterior_entropy(
                            f_,
                            b_,
                            prob_,
                            lengths
                        );

        entropy_measures.append(entropy);

        # Apply length masking
        if lengths is not None:
            for i, l in enumerate(lengths):
                if sequence_length > l:
                    l_[l:,i] = 0.0;

        Likelihoods  = l_;

        return torch.sum(Likelihoods, dim=0), sum(entropy_measures);
