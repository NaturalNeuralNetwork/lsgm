# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import torch
import numpy as np
import scipy
from state_parallel import notif_state, motif_collection
from header import small_value
from dense_network import dense_network
from header import weight_variable
from lstm_network import lstm_network
from gru_network import gru_network

try:
    profile
except NameError:
    profile = lambda x : x

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

class bidirectional_lsgm(torch.nn.Module):
    def __init__(self, motif_length=24, num_motifs=16, order = 1, transition_type="Markov", recurrence_type="LSTM", device = 0, num_lt_states = 128, train_pwms=True, use_log_forward=False):
        super().__init__();

        self.use_log_forward = use_log_forward;

        self.motif_length = motif_length;
        self.num_motifs   = num_motifs;
        self.order        = order;
        self.device       = device;
        self.mode         = 0;

        """ State emisions can be modeled using a log->CNN->exp layer """
        self.motif_state  = motif_collection(motif_length, num_motifs, device = device, train_pwms=train_pwms);
        self.notif_state  = notif_state();
        self.states       = [self.motif_state, self.notif_state];

        """
        Transition from start state to non-silent states
        """
        # start_transition     = torch.ones(self.num_motifs+1).double() / (self.num_motifs+1);
        start_transition     = torch.ones(self.num_motifs+1) / (self.num_motifs+1);   # TBD: Permanently remove double precision
        self.start_weights_f = torch.nn.Parameter(start_transition.view(start_transition.size()[0],1));
        self.start_weights_r = torch.nn.Parameter(start_transition.clone().view(start_transition.size()[0],1));

        """
        Transition matrices for Hidden Markov Model functionality
        """
        self.transition_type = transition_type;
        self.lstm_network_   = lstm_network if recurrence_type == "LSTM" else gru_network;

        # transition_matrix_np = [];

        # for i in range((self.num_motifs+1)):
        #     transition_matrix_row = np.array([0.0 for j in range(self.num_motifs+1+1)]);
        #     transition_matrix_row[i] = 7.0;
        #     transition_matrix_np.append(transition_matrix_row);

        # transition_matrix_torch   = torch.from_numpy(np.array(transition_matrix_np)).double();
        # transition_matrix_torch   = torch.from_numpy(np.array(transition_matrix_np)); # TBD: permanently remove double precision
        transition_matrix_torch   = torch.Tensor(self.num_motifs+1,self.num_motifs+2); # TBD: permanently remove double precision
        transition_matrix_torch.normal_(0.0,0.01);
        self.transition_weights_f = torch.nn.Parameter(transition_matrix_torch);
        self.transition_weights_r = torch.nn.Parameter(transition_matrix_torch.clone());

        transition_matrix_0order  = weight_variable(shape=(num_motifs+2,));
        # self.zero_order_f         = torch.nn.Parameter(transition_matrix_0order.double());
        # self.zero_order_r         = torch.nn.Parameter(transition_matrix_0order.clone().double());
        self.zero_order_f         = torch.nn.Parameter(transition_matrix_0order);   # TBD: permanently remove double precision
        self.zero_order_r         = torch.nn.Parameter(transition_matrix_0order.clone());

        """
        Transition networks for long-term recurrence
        """
        num_neurons_intermediate_lstm  = num_lt_states;
        num_neurons_intermediate_dense = num_lt_states;

        num_inputs = 4;

        # Anand - commented out on Sept 30 2018
        # self.lstm_f = self.lstm_network_([ \
        #     (num_inputs,num_neurons_intermediate_lstm,2), \
        #     (num_neurons_intermediate_lstm,num_neurons_intermediate_dense,"ReLu"), \
        #     (num_neurons_intermediate_dense, self.num_motifs+2,"softmax")\
        # ], bidirectional=False, device=self.device);

        # self.lstm_r = self.lstm_network_([ \
        #     (num_inputs,num_neurons_intermediate_lstm,2), \
        #     (num_neurons_intermediate_lstm,num_neurons_intermediate_dense,"ReLu"), \
        #     (num_neurons_intermediate_dense, self.num_motifs+2,"softmax")\
        # ], bidirectional=False, device=self.device);

        # Anand - changes on Sept 30 2018 for order = 1 and dense network performance
        num_outputs = (self.num_motifs + 1) * (self.num_motifs + 2) if order == 1 else self.num_motifs + 2;

        self.lstm_f = self.lstm_network_([ \
            (num_inputs,num_neurons_intermediate_lstm,2), \
            (num_neurons_intermediate_lstm,num_neurons_intermediate_dense,"ReLu"), \
            (num_neurons_intermediate_dense, num_outputs, "noop")\
        ], bidirectional=False, device=self.device);

        self.lstm_r = self.lstm_network_([ \
            (num_inputs,num_neurons_intermediate_lstm,2), \
            (num_neurons_intermediate_lstm,num_neurons_intermediate_dense,"ReLu"), \
            (num_neurons_intermediate_dense, num_outputs, "noop")\
        ], bidirectional=False, device=self.device);
        # Sept 30 2018's changes end

        self.direction = "forward";

    def start_transition(self):
        if self.direction == "forward":
            return torch.nn.functional.softmax(self.start_weights_f, dim=0);
        else:
            return torch.nn.functional.softmax(self.start_weights_r, dim=0);

    def transition_matrix(self, mask_tensor = None, emission_tensor = None):
        if self.transition_type == "Markov":
            if self.direction == "forward":
                if self.order == 0:
                    new_weights = torch.stack([self.zero_order_f] * (self.num_motifs+1), dim=0);
                    return torch.nn.functional.softmax(new_weights);

                return torch.nn.functional.softmax(self.transition_weights_f, dim=1);
            elif self.direction == "backward":
                if self.order == 0:
                    new_weights = torch.stack([self.zero_order_r] * (self.num_motifs+1), dim=0);
                    return torch.nn.functional.softmax(new_weights);

                return torch.nn.functional.softmax(self.transition_weights_r, dim=1);
            else:
                raise ValueError("Cannot understand direction");

        """ Reverse the tensor if in backward mode """
        if (self.direction == "backward"):
            mask_tensor_seq = torch.unbind(mask_tensor, dim = 1);
            mask_tensor_rsq = [m for m in reversed(mask_tensor_seq)];
            mask_tensor     = torch.stack(mask_tensor_rsq, dim=1);

        batch_size = mask_tensor.size()[0];

        """ Initialization entry """
        zero_seq   = torch.autograd.Variable(torch.ones(batch_size, 1, 4));
        if next(self.parameters()).is_cuda: zero_seq = zero_seq.cuda(self.device);
        inputs     = torch.cat((zero_seq, mask_tensor), dim=1);

        transition_tensor  = [];

        if self.order == 0:
            if self.direction == "forward":
                # transition_vectors = self.lstm_f(inputs).double(); 
                transition_vectors = self.lstm_f(inputs); # TBD: Permanently remove double precision
            else:
                # transition_vectors = self.lstm_r(inputs).double(); 
                transition_vectors = self.lstm_r(inputs); # TBD: Permanently remove double precision
  
            # Anand - changes for Sept 30th 2018 to support order = 1 LSGM
            transition_vectors = torch.nn.functional.softmax(transition_vectors, dim=2);

            transition_vectors = torch.unbind(transition_vectors, dim=1); # A (seq_length + 1) length list of [batch_size x self.num_motifs + 2]

            for t in transition_vectors:
                transition_tensor.append(torch.stack([t] * (self.num_motifs+1), dim=1)); # Each item is [batch_size x num_motifs+1 x num_motifs+2]
        else:
            # Anand - changes for Sept 30th 2018 to support order = 1 LSGM
            # Commented out on Sept 30 2018
            # raise AttributeError("Long-term recurrence not implemented for order = 1 graphs");

            # transition_vectors = [];

            # transition_vectors = torch.stack(transition_vectors, dim=0);    # state x batch x seq_length+1 x dest
            # transition_vectors = torch.transpose(transition_vectors, 0, 1); # batch x state x seq_length + 1 x dest
            # transition_tensor  = torch.unbind(transition_vectors, dim=2);   # list of, batch x num_motifs+1 x num_motifs + 2

            # if self.direction == "forward":
            #     for lstm in self.lstm_f:
            #         # transition_vectors.append(lstm(inputs).double());
            #         transition_vectors.append(lstm(inputs)); # TBD: Permanently remove double precision
            # else:
            #     for lstm in self.lstm_r:
            #         # transition_vectors.append(lstm(inputs).double());
            #         transition_vectors.append(lstm(inputs)); # TBD: Permanently remove double precision

            if self.direction == "forward":
                transition_vectors = self.lstm_f(inputs);
            else:
                transition_vectors = self.lstm_r(inputs);

            (a,b,c) = tuple(transition_vectors.size()); # [batch, sequence_length + 1, (num_motifs+1) * (num_motifs+2)

            transition_vectors = torch.nn.functional.softmax(transition_vectors.view(a,b,self.num_motifs+1,self.num_motifs+2), dim=3);

            return iter(torch.unbind(transition_vectors, dim=1));

        return iter(transition_tensor);

    def __emissions(self, items):
        emissions         = [];

        mask_tensor       = items[2];
        batch_size        = mask_tensor.size()[0];
        sequence_length   = mask_tensor.size()[1];

        motif_emissions   = self.motif_state(items, no_post_process = True); # state x batch x length
        notif_emissions   = self.notif_state(items, no_post_process = True); # 1 x batch x length

        zero_pad  = torch.autograd.Variable(torch.zeros(self.num_motifs,batch_size,self.motif_length-1));
        is_cuda   = next(self.parameters()).is_cuda;
        if is_cuda: zero_pad = zero_pad.cuda(self.device);

        """ Attach the zeros at the start if direction is forward, or at the end if the direction is reverse """
        if self.direction == "forward":
            motif_emissions = torch.cat((zero_pad, motif_emissions), dim=2); # batch x num_motifs x length
        else:
            motif_emissions = torch.cat((motif_emissions, zero_pad), dim=2); # batch x num_motifs x length

        # emissions         = torch.cat((motif_emissions, notif_emissions), dim = 0).double();
        emissions         = torch.cat((motif_emissions, notif_emissions), dim = 0); # TBD: Permanently remove double precision
        emissions         = torch.unbind(emissions, dim= 2);

        if self.direction == "forward":
            return iter(emissions);
        else:
            return reversed(emissions);

    @profile
    def __forward_lstm_log(self, items, emissions, transition_matrix_, index = None):
        """
        emissions_state?   := batch_size x sequence_length x distribution
        transitions_state? := batch_size x sequence_length x distribution
        """
        batch_size        = items[0].size()[0];
        sequence_length   = items[0].size()[1];

        # Initialization of matrices
        # transition_matrix_= list(self.transition_matrix(items[2], torch.stack(emissions, dim=2)));
        start_transition  = self.start_transition();
        is_cuda           = next(self.parameters()).is_cuda;

        forward_matrix_buffer = None;

        if hasattr(self, 'forward_matrix_buffer'):
            if  (self.forward_matrix_buffer.size(0) == self.num_motifs + 1) and \
                (self.forward_matrix_buffer.size(1) == sequence_length) and \
                (self.forward_matrix_buffer.size(2) >= batch_size):
                forward_matrix_buffer = self.forward_matrix_buffer[:,:,:batch_size].fill_(0.0);

        if forward_matrix_buffer is None:
            forward_matrix_buffer      = torch.FloatTensor((self.num_motifs+1),sequence_length,batch_size).fill_(0.0);
            if is_cuda:
                forward_matrix_buffer  = forward_matrix_buffer.cuda(self.device);
            self.forward_matrix_buffer = forward_matrix_buffer;

        forward_matrix = torch.autograd.Variable(forward_matrix_buffer);

        """
        New: Convert everything to log domain
        ########################################################################
        """
        transition_matrix_ = [torch.log(t) for t in transition_matrix_];
        emissions          = [torch.log(e) for e in emissions];
        start_transition   = torch.log(start_transition);
        """
        ########################################################################
        """

        """
        Zero values in the forward matrix are not compatible with the log domain
        When only the notif state is allowed, there will be many zero values in the transition
        matrix. So for the case where only notif state is allowed, this computation is done
        separately for log domain forward
        """
        if index is not None:
            value = None;

            """ Everything comes from the notif state """
            for i, emits in enumerate(emissions):
                if i == 0:
                    value = start_transition[-1] + emits[-1];
                else:
                    value += transition_matrix_[i][:,-1,-2];
                    value += emits[-1];

            value += transition_matrix_[-1][:,-1,-1];

            return forward_matrix, value;
                # Note: forward_matrix is incomplete, but I don't use forward_matrix, so this is okay

        # Recurrence
        for i, emits in enumerate(emissions):
            transition_matrix = transition_matrix_[i];

            """ notif state """
            if i == 0:
                forward_matrix[-1,0] = start_transition[-1] + emits[-1];
            else:
                if i <= self.motif_length-1:
                    # Note : Until all the motif stats can be used (the first motif_length positions), we need
                    # to carefully compute notif state-only forward values separately, because zero values in
                    # the forward matrix are not compatible with log domain computations
                    forward_matrix[-1,i] = forward_matrix[-1,i-1] + transition_matrix[:,-1,-2] + emits[-1];
                else:
                    transition_source    = transition_matrix[:,:,-2]; #.contiguous();                                         # [batch x states]
                    forward_source       = forward_matrix[:,i-1]; # .clone().contiguous();                                     # [states x batch]
                    forward_source       = torch.transpose(forward_source, 0, 1); # .contiguous();                             # [batch x states]
                    forward_value        = log_add(transition_source + forward_source, dim=1) + emits[-1]      

                    forward_matrix[-1,i] = forward_value;

            """ motif states """
            transition_matrix = transition_matrix_[i-self.motif_length+1];
                                            # What we want is the transition at X[i-motif_length].
                                            # However, transition_matrix_ is already shifted by 1, because the first symbol is zero.
                                            # Hence we need to shift (motif_length - 1) times more.

            if i == self.motif_length - 1:
                start_source = start_transition[:-1].view((self.num_motifs+1)-1,1); #.contiguous(); # [states - 1 x 1]
                emit_source  = emits[:-1].contiguous(); # [states - 1 x batch_size]
                forward_matrix[:-1,self.motif_length-1] = start_source + emit_source;
            elif i > self.motif_length - 1:

                if i - self.motif_length >= self.motif_length-1:
                    transition_source     = torch.transpose(transition_matrix[:,:,:-2], 1, 2) #.contiguous();  # [batch x state-1 x state]
                    forward_source        = forward_matrix[:,i-self.motif_length];                           # [states x batch]
                    forward_source        = torch.transpose(forward_source, 0, 1);                           # [batch x states]
                    forward_source        = forward_source.view(batch_size,1,self.num_motifs+1);             # [batch x 1 x states]
                    forward_prelim        = log_add(transition_source + forward_source, dim=2);              # [batch x states-1]
                    forward_prelim        = torch.transpose(forward_prelim, 0, 1);                           # [states - 1 x batch]
                    emission_source       = emits[:-1];                                                      # [states - 1 x batch]

                    forward_matrix[:-1,i] = forward_prelim + emission_source;                                # [states - 1 x batch]
                else:
                    # Note : Motif states look at forward entries motif_length positions before the current position
                    # However until i = 2 * motif_length - 1, the motif state entries in the forwrard matrix will be zero
                    # The zero values are not compatible with log domain. So, until i reaches this value we have to separately
                    # compute motif state forward values from notif state-only forward values
                    forward_source = forward_matrix[-1,i-self.motif_length].view(1,batch_size);
                    transition_src = torch.transpose(transition_matrix[:,-1,:-2],0,1);
                    emission_src   = emits[:-1];

                    forward_matrix[:-1,i] = forward_source + transition_src + emission_src;

        # Termination
        forward_last_col   = forward_matrix[:,-1,:];
        transition_vector  = transition_matrix_[-1][:,:,-1];                                              # [batch x state]
        predecessor_values = forward_last_col;                                                            # [states x batch]
        predecessor_values = torch.transpose(predecessor_values, 0, 1); #.contiguous();                   # [batch x states]
        termination        = log_add(transition_vector + predecessor_values, dim=1);                      # [batch]

        return forward_matrix, termination;

    def __forward_lstm(self, items, emissions, index = None):
        """
        emissions_state?   := batch_size x sequence_length x distribution
        transitions_state? := batch_size x sequence_length x distribution
        """
        batch_size        = items[0].size()[0];
        sequence_length   = items[0].size()[1];

        # Initialization of matrices
        transition_matrix_= list(self.transition_matrix(items[2], torch.stack(emissions, dim=2)));
        start_transition  = self.start_transition();
        forward_matrix    = torch.autograd.Variable(torch.zeros((self.num_motifs+1),sequence_length,batch_size).double());
        is_cuda           = next(self.parameters()).is_cuda;

        """
        New : This  is a double precision implementation, convert all to double
        #######################################################################
        """
        start_transition   = start_transition.double();
        transition_matrix_ = [t.double() for t in transition_matrix_];
        emissions          = [e.double() for e in emissions];
        """
        #######################################################################
        """

        if is_cuda:
            forward_matrix = forward_matrix.cuda(self.device);

        # Parallelize the operation for all motif states
        # ==============================================
        # Parallelizing the recursion
        # ---------------------------
        # 1. Take forward[:,i-self.motif_length]. This will be [states x batch_size]. Call this forward_source.
        # 2. Now take transition probabilities to every state except the notif and end states. Since the excluded states are the last two, this is simply,
        #    transition_matrix[:,:,:-2]. Let this be transition_source. This is [batch x states x states - 1]
        # 3. Rearrange transition_source such that every row in each matrix in transition_source is the destination state. 
        #    But originally, matrices in transition_source, being a slice of transition_matrix, have rows as the source or predecessor states, 
        #    and the columns as the target or destination states. Simply transposing it will have the desired effect.
        # 4. Every row of every matrix in transition_source (after step 4), should dot with every forward_source state. Note that transition_source is 
        #    [batch x state-1 x state] and forward_source is [state x batch_size]. Reshape as follows:
        #    transition_source remains at [batch x state-1 x state], forward_source reshapes to [batch_size x state x 1]. Then do
        #    torch.matmul(transition_source, forward_source) to get [batch_size x state-1] result. Transpose this to get back to format.
        # 5. Then we need to multiply this with emissions which are [states x batch_size]. Slice as appropriate.
        # 
        # Parallelizing the initialization - not affected by LSTM since this is not from an LSTM
        # --------------------------------------------------------------------------------------
        # 1. Take start_transition[:-1]. This excludes the notif state. Reshape this into [states-1 x 1].
        # 2. Take emits[:-1]. This excludes the notif state. Reshape this into [states-1 x batch_size].
        # 3. Multiply together and assign to forward_matrix[:-1,motif_length-1]

        # Recurrence
        for i, emits in enumerate(emissions):
            transition_matrix = transition_matrix_[i];

            """ notif state """
            if i == 0:
                forward_matrix[-1,0] = start_transition[-1] * emits[-1];
            else:
                transition_source    = transition_matrix[:,:,-2].contiguous();                                         # [batch x states]
                transition_source    = transition_source.view(batch_size,self.num_motifs+1,1);                         # [batch x states x 1]
                transition_source    = torch.transpose(transition_source,1,2);                                         # [batch x 1 x states]

                forward_source       = forward_matrix[:,i-1].clone().contiguous();                                     # [states x batch]
                forward_source       = torch.transpose(forward_source, 0, 1).contiguous();                             # [batch x states]
                forward_source       = forward_source.view(batch_size, (self.num_motifs+1), 1);                        # [batch x states x 1] 

                forward_value        = torch.matmul(transition_source, forward_source).view(batch_size);               # [batch_size x 1]

                emission_vector      = emits[-1].contiguous().view(batch_size);                                        # [batch x 1]

                forward_matrix[-1,i] = forward_value * emission_vector;
            if index == (self.num_motifs+1) - 1:
                continue;

            """ motif states """
            transition_matrix = transition_matrix_[i-self.motif_length+1];
                                            # What we want is the transition at X[i-motif_length].
                                            # However, transition_matrix_ is already shifted by 1, because the first symbol is zero.
                                            # Hence we need to shift (motif_length - 1) times more.

            if index is not None:
                raise ValueError("Unsupported Feature!");

                if i == self.motif_length - 1:
                    forward_matrix[index,0] = start_transition[index] * emits[index];
                elif i > self.motif_length - 1:
                    forward_source       = forward_matrix[:,i-self.motif_length].clone().contiguous();              # [states x batch]
                    transition_source    = transition_matrix[:,index].contiguous().view((self.num_motifs+1),1);     # [states x 1]
                    emission_vector      = emits[index].contiguous().view(batch_size,1);                            # [batch_size x 1]
                    forward_value        = torch.sum(forward_source * transition_source, dim=0).view(batch_size,1); # [batch_size x 1]

                    forward_matrix[index,i] = forward_value * emission_vector;

                continue;

            if i == self.motif_length - 1:
                start_source = start_transition[:-1].view((self.num_motifs+1)-1,1).contiguous(); # [states - 1 x 1]
                emit_source  = emits[:-1].contiguous(); # [states - 1 x batch_size]
                forward_matrix[:-1,self.motif_length-1] = start_source * emit_source;
            elif i > self.motif_length - 1:
                transition_source     = torch.transpose(transition_matrix[:,:,:-2], 1, 2).contiguous();  # [batch x state-1 x state]

                forward_source        = forward_matrix[:,i-self.motif_length].clone().contiguous();      # [states x batch]
                forward_source        = torch.transpose(forward_source, 0, 1).contiguous();              # [batch x states]
                forward_source        = forward_source.view(batch_size,self.num_motifs+1,1);             # [batch x states x 1]

                forward_prelim        = torch.matmul(transition_source, forward_source);                 # [batch x states - 1 x 1]
                forward_prelim        = torch.squeeze(forward_prelim, dim=2);                            # [batch x states - 1]
                forward_prelim        = torch.transpose(forward_prelim, 0, 1);                           # [states - 1 x batch]

                emission_source       = emits[:-1];                                                      # [states - 1 x batch]

                forward_matrix[:-1,i] = forward_prelim * emission_source;                                # [states - 1 x batch]

        # Termination
        forward_last_col   = forward_matrix[:,-1,:];

        transition_vector  = transition_matrix_[-1][:,:,-1].contiguous().view(batch_size,self.num_motifs+1,1); # [batch x state x 1]
        transition_vector  = torch.transpose(transition_vector,1,2);                                      # [batch x 1 x state]

        predecessor_values = forward_last_col;                                                            # [states x batch]
        predecessor_values = torch.transpose(predecessor_values, 0, 1).contiguous();                      # [batch x states]
        predecessor_values = predecessor_values.view(batch_size,self.num_motifs+1,1);                     # [batch x states x 1]

        termination        = torch.matmul(transition_vector, predecessor_values);                         # [batch x 1]

        return forward_matrix, torch.log(termination + small_value);

    def __forward_log(self, items, emissions, index = None):
        """
        emissions_state?   := batch_size x sequence_length x distribution
        transitions_state? := batch_size x sequence_length x distribution
        """
        batch_size        = items[0].size()[0];
        sequence_length   = items[0].size()[1];

        # Initialization of matrices
        transition_matrix_= self.transition_matrix();
        start_transition  = self.start_transition();
        forward_matrix    = torch.autograd.Variable(torch.zeros((self.num_motifs+1),sequence_length,batch_size).float());
        is_cuda           = next(self.parameters()).is_cuda;

        """
        New : This is a log_domain implementation, convert to logarithms
        #######################################################################
        """
        start_transition   = torch.log(start_transition);
        transition_matrix_ = torch.unsqueeze(torch.log(transition_matrix_), dim=0).expand(batch_size,transition_matrix_.size(0),transition_matrix_.size(1));
        emissions          = [torch.log(e) for e in emissions];
        """
        #######################################################################
        """

        if is_cuda:
            forward_matrix = forward_matrix.cuda(self.device);

        """
        Zero values in the forward matrix are not compatible with the log domain
        When only the notif state is allowed, there will be many zero values in the transition
        matrix. So for the case where only notif state is allowed, this computation is done
        separately for log domain forward
        """
        if index is not None:
            value = None;

            """ Everything comes from the notif state """
            for i, emits in enumerate(emissions):
                if i == 0:
                    value = start_transition[-1] + emits[-1];
                else:
                    value += transition_matrix_[:,-1,-2];
                    value += emits[-1];

            value += transition_matrix_[:,-1,-1];

            return forward_matrix, value;
                # Note: forward_matrix is incomplete, but I don't use forward_matrix, so this is okay

        # # Recurrence
        # for i, emits in enumerate(emissions):
        #     transition_matrix = transition_matrix_;

        #     """ notif state """
        #     if i == 0:
        #         forward_matrix[-1,0] = start_transition[-1] * emits[-1];
        #     else:
        #         forward_source       = forward_matrix[:,i-1].clone().contiguous();                              # [states x batch]
        #         transition_source    = transition_matrix[:,-2].contiguous().view((self.num_motifs+1),1);        # [states x 1]
        #         emission_vector      = emits[-1].contiguous().view(batch_size);                                 # [batch_size]
        #         forward_value        = torch.sum(forward_source * transition_source, dim=0).view(batch_size);   # [batch_size]
        #         forward_matrix[-1,i] = forward_value * emission_vector;

        #     if index == (self.num_motifs+1) - 1:
        #         continue;

        #     """ motif states """
        #     if index is not None:
        #         if i == self.motif_length - 1:
        #             forward_matrix[index,0] = start_transition[index] * emits[index];
        #         elif i > self.motif_length - 1:
        #             forward_source       = forward_matrix[:,i-self.motif_length].clone().contiguous();              # [states x batch]
        #             transition_source    = transition_matrix[:,index].contiguous().view((self.num_motifs+1),1);     # [states x 1]
        #             emission_vector      = emits[index].contiguous().view(batch_size,1);                            # [batch_size x 1]
        #             forward_value        = torch.sum(forward_source * transition_source, dim=0).view(batch_size,1); # [batch_size x 1]

        #             forward_matrix[index,i] = forward_value * emission_vector;

        #         continue;

        #     if i == self.motif_length - 1:
        #         start_source = start_transition[:-1].view((self.num_motifs+1)-1,1).contiguous(); # [states - 1 x 1]
        #         emit_source  = emits[:-1].contiguous(); # [states - 1 x batch_size]
        #         forward_matrix[:-1,self.motif_length-1] = start_source * emit_source;
        #     elif i > self.motif_length - 1:
        #         forward_source        = forward_matrix[:,i-self.motif_length].clone().contiguous();
        #         transition_source     = torch.transpose(transition_matrix[:,:-2], 0, 1).contiguous();
        #         forward_prelim        = torch.matmul(transition_source, forward_source);
        #         forward_matrix[:-1,i] = forward_prelim * emits[:-1];
        # Recurrence
        for i, emits in enumerate(emissions):
            transition_matrix = transition_matrix_;

            """ notif state """
            if i == 0:
                forward_matrix[-1,0] = start_transition[-1] + emits[-1];
            else:
                if i <= self.motif_length-1:
                    # Note : Until all the motif stats can be used (the first motif_length positions), we need
                    # to carefully compute notif state-only forward values separately, because zero values in
                    # the forward matrix are not compatible with log domain computations
                    forward_matrix[-1,i] = forward_matrix[-1,i-1] + transition_matrix[:,-1,-2] + emits[-1];
                else:
                    transition_source    = transition_matrix[:,:,-2]; #.contiguous();                                         # [batch x states]
                    forward_source       = forward_matrix[:,i-1]; # .clone().contiguous();                                     # [states x batch]
                    forward_source       = torch.transpose(forward_source, 0, 1); # .contiguous();                             # [batch x states]
                    forward_value        = log_add(transition_source + forward_source, dim=1) + emits[-1]      

                    forward_matrix[-1,i] = forward_value;

            """ motif states """
            if i == self.motif_length - 1:
                start_source = start_transition[:-1].view((self.num_motifs+1)-1,1); #.contiguous(); # [states - 1 x 1]
                emit_source  = emits[:-1].contiguous(); # [states - 1 x batch_size]
                forward_matrix[:-1,self.motif_length-1] = start_source + emit_source;
            elif i > self.motif_length - 1:

                if i - self.motif_length >= self.motif_length-1:
                    transition_source     = torch.transpose(transition_matrix[:,:,:-2], 1, 2) #.contiguous();  # [batch x state-1 x state]
                    forward_source        = forward_matrix[:,i-self.motif_length];                           # [states x batch]
                    forward_source        = torch.transpose(forward_source, 0, 1);                           # [batch x states]
                    forward_source        = forward_source.view(batch_size,1,self.num_motifs+1);             # [batch x 1 x states]
                    forward_prelim        = log_add(transition_source + forward_source, dim=2);              # [batch x states-1]
                    forward_prelim        = torch.transpose(forward_prelim, 0, 1);                           # [states - 1 x batch]
                    emission_source       = emits[:-1];                                                      # [states - 1 x batch]

                    forward_matrix[:-1,i] = forward_prelim + emission_source;                                # [states - 1 x batch]
                else:
                    # Note : Motif states look at forward entries motif_length positions before the current position
                    # However until i = 2 * motif_length - 1, the motif state entries in the forwrard matrix will be zero
                    # The zero values are not compatible with log domain. So, until i reaches this value we have to separately
                    # compute motif state forward values from notif state-only forward values
                    forward_source = forward_matrix[-1,i-self.motif_length].view(1,batch_size);
                    transition_src = torch.transpose(transition_matrix[:,-1,:-2],0,1);
                    emission_src   = emits[:-1];

                    forward_matrix[:-1,i] = forward_source + transition_src + emission_src;

        # # Termination
        # forward_last_col   = forward_matrix[:,-1,:];
        # predecessor_values = forward_last_col;
        # transition_vector  = transition_matrix[:,-1].contiguous();
        # transition_vector  = transition_vector.view(transition_vector.size()[0],1);
        # termination        = torch.sum(predecessor_values * transition_vector, dim=0);

        # return forward_matrix, torch.log(termination + small_value);

        # Termination
        forward_last_col   = forward_matrix[:,-1,:];
        transition_vector  = transition_matrix_[:,:,-1];                                              # [batch x state]
        predecessor_values = forward_last_col;                                                            # [states x batch]
        predecessor_values = torch.transpose(predecessor_values, 0, 1); #.contiguous();                   # [batch x states]
        termination        = log_add(transition_vector + predecessor_values, dim=1);                      # [batch]

        return forward_matrix, termination;

    def __forward(self, items, emissions, index = None):
        """
        emissions_state?   := batch_size x sequence_length x distribution
        transitions_state? := batch_size x sequence_length x distribution
        """
        batch_size        = items[0].size()[0];
        sequence_length   = items[0].size()[1];

        # Initialization of matrices
        transition_matrix_= self.transition_matrix();
        start_transition  = self.start_transition();
        forward_matrix    = torch.autograd.Variable(torch.zeros((self.num_motifs+1),sequence_length,batch_size).double());
        is_cuda           = next(self.parameters()).is_cuda;

        """
        New : This  is a double precision implementation, convert all to double
        #######################################################################
        """
        start_transition   = start_transition.double();
        transition_matrix_ = transition_matrix_.double();
        emissions          = [e.double() for e in emissions];
        """
        #######################################################################
        """

        if is_cuda:
            forward_matrix = forward_matrix.cuda(self.device);

        # Parallelize the operation for all motif states
        # ==============================================
        # Parallelizing the recursion
        # ---------------------------
        # 1. Take forward[:,i-self.motif_length]. This will be [states x batch_size]. Call this forward_source.
        # 2. Now take transition probabilities to every state except the notif and end states. Since the excluded states are the last two, this is simply,
        #    transition_matrix[:,:-2]. Let this be transition_source. 
        # 3. Rearrange transition_source as follows such that row in transition_source is the destination state. But originally, the transition_source,
        #    being a slice of transition_matrix, has rows as the source or predecessor states, and the columns as the target or destination states. Simply
        #    transposing it will have the desired effect.
        # 4. Every row of transition_source (after step 4), should dot with every forward_source state. Note that transition_source is [state-1 x state] and
        #    forward_source is [states x batch_size]. Hence, we want torch.matmul(transition_source, forward_source). This is [state-1 x batch_size].
        # 5. Then we need to multiply this with emissions which are [states x batch_size]. Slice as appropriate.
        # 
        # Parallelizing the initialization
        # --------------------------------
        # 1. Take start_transition[:-1]. This excludes the notif state. Reshape this into [states-1 x 1].
        # 2. Take emits[:-1]. This excludes the notif state. Reshape this into [states-1 x batch_size].
        # 3. Multiply together and assign to forward_matrix[:-1,motif_length-1]

        # Recurrence
        for i, emits in enumerate(emissions):
            transition_matrix = transition_matrix_;

            """ notif state """
            if i == 0:
                forward_matrix[-1,0] = start_transition[-1] * emits[-1];
            else:
                forward_source       = forward_matrix[:,i-1].clone().contiguous();                              # [states x batch]
                transition_source    = transition_matrix[:,-2].contiguous().view((self.num_motifs+1),1);        # [states x 1]
                emission_vector      = emits[-1].contiguous().view(batch_size);                                 # [batch_size]
                forward_value        = torch.sum(forward_source * transition_source, dim=0).view(batch_size);   # [batch_size]
                forward_matrix[-1,i] = forward_value * emission_vector;

            if index == (self.num_motifs+1) - 1:
                continue;

            """ motif states """
            if index is not None:
                if i == self.motif_length - 1:
                    forward_matrix[index,0] = start_transition[index] * emits[index];
                elif i > self.motif_length - 1:
                    forward_source       = forward_matrix[:,i-self.motif_length].clone().contiguous();              # [states x batch]
                    transition_source    = transition_matrix[:,index].contiguous().view((self.num_motifs+1),1);     # [states x 1]
                    emission_vector      = emits[index].contiguous().view(batch_size,1);                            # [batch_size x 1]
                    forward_value        = torch.sum(forward_source * transition_source, dim=0).view(batch_size,1); # [batch_size x 1]

                    forward_matrix[index,i] = forward_value * emission_vector;

                continue;

            if i == self.motif_length - 1:
                start_source = start_transition[:-1].view((self.num_motifs+1)-1,1).contiguous(); # [states - 1 x 1]
                emit_source  = emits[:-1].contiguous(); # [states - 1 x batch_size]
                forward_matrix[:-1,self.motif_length-1] = start_source * emit_source;
            elif i > self.motif_length - 1:
                forward_source        = forward_matrix[:,i-self.motif_length].clone().contiguous();
                transition_source     = torch.transpose(transition_matrix[:,:-2], 0, 1).contiguous();
                forward_prelim        = torch.matmul(transition_source, forward_source);
                forward_matrix[:-1,i] = forward_prelim * emits[:-1];

        # Termination
        forward_last_col   = forward_matrix[:,-1,:];
        predecessor_values = forward_last_col;
        transition_vector  = transition_matrix[:,-1].contiguous();
        transition_vector  = transition_vector.view(transition_vector.size()[0],1);
        termination        = torch.sum(predecessor_values * transition_vector, dim=0);

        return forward_matrix, torch.log(termination + small_value);

    @profile
    def forward(self, items, bidirectional = True):
        """
        Input sequence batch is of shape [batch_size x sequence_length x character]
        """
        directions = ["forward"];

        if bidirectional: directions.append("backward");

        tns = [];

        for direction in directions:
            self.direction = direction;

            emissions = self.__emissions(items);
            emissions = [e for e in emissions];

            tn1 = tn2 = None;

            if self.transition_type == "Markov":
                # # Test code
                # transition_matrix_= list(self.transition_matrix(items[2], torch.stack(emissions, dim=2)));
                # fm11, tn11 = self.__forward_log(items, emissions);
                # fm21, tn22 = self.__forward_log(items, emissions, index = (self.num_motifs+1) - 1);
                # fm1, tn1 = self.__forward(items, emissions);
                # fm2, tn2 = self.__forward(items, emissions, index = (self.num_motifs+1) - 1);
                # print(tn1, tn11);
                # print(tn2, tn22);
                # assert(1==2);
                # # Test code end
                if self.use_log_forward:
                    fm1, tn1 = self.__forward_log(items, emissions);
                    fm2, tn2 = self.__forward_log(items, emissions, index = (self.num_motifs+1) - 1);
                else:
                    fm1, tn1 = self.__forward(items, emissions);
                    fm2, tn2 = self.__forward(items, emissions, index = (self.num_motifs+1) - 1);
            else:
                # # Test code
                # transition_matrix_= list(self.transition_matrix(items[2], torch.stack(emissions, dim=2)));
                # fm11, tn11 = self.__forward_lstm_log(items, emissions, transition_matrix_);
                # fm21, tn22 = self.__forward_lstm_log(items, emissions, transition_matrix_, index = (self.num_motifs+1) - 1);
                # fm1, tn1 = self.__forward_lstm(items, emissions);
                # fm2, tn2 = self.__forward_lstm(items, emissions, index = (self.num_motifs+1) - 1);
                # print(tn1[:,0,0], tn11);
                # print(tn2[:,0,0], tn22);
                # assert(1==2);
                # # Test code end

                if self.use_log_forward:
                    transition_matrix_= list(self.transition_matrix(items[2], torch.stack(emissions, dim=2)));
                    fm1, tn1 = self.__forward_lstm_log(items, emissions, transition_matrix_);
                    fm2, tn2 = self.__forward_lstm_log(items, emissions, transition_matrix_, index = (self.num_motifs+1) - 1);
                else:
                    fm1, tn1 = self.__forward_lstm(items, emissions);
                    fm2, tn2 = self.__forward_lstm(items, emissions, index = (self.num_motifs+1) - 1);

            tn1 = tn1.view(tn1.size(0), 1);
            tn2 = tn2.view(tn2.size(0), 1);

            tns.append(tn1);
            tns.append(tn2);

        if self.mode == 0:
            return torch.cat(tns, dim=1);

        raise ValueError("Not implemented!");

        tns = [];

        for i in range((self.num_motifs+1) - 1):
            fm_, tn_ = self.__forward(items, emissions, index=i);
            tns.append(tn_.view(tn_.size()[0],1));

        return torch.cat([tn1] + tns + [tn2], dim=1);
