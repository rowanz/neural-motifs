from typing import Tuple

from overrides import overrides
import torch
from torch.autograd import Function, Variable
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
import itertools
from ._ext import highway_lstm_layer


def block_orthogonal(tensor, split_sizes, gain=1.0):
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """

    if isinstance(tensor, Variable):
        block_orthogonal(tensor.data, split_sizes, gain)
        return tensor

    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError("tensor dimensions must be divisible by their respective "
                         "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])

        # let's not initialize empty things to 0s because THAT SOUNDS REALLY BAD
        assert len(block_slice) == 2
        sizes = [x.stop - x.start for x in block_slice]
        tensor_copy = tensor.new(max(sizes), max(sizes))
        torch.nn.init.orthogonal(tensor_copy, gain=gain)
        tensor[block_slice] = tensor_copy[0:sizes[0], 0:sizes[1]]


class _AlternatingHighwayLSTMFunction(Function):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, train: bool) -> None:
        super(_AlternatingHighwayLSTMFunction, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.train = train

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                state_accumulator: torch.Tensor,
                memory_accumulator: torch.Tensor,
                dropout_mask: torch.Tensor,
                lengths: torch.Tensor,
                gates: torch.Tensor) -> Tuple[torch.Tensor, None]:
        sequence_length, batch_size, input_size = inputs.size()
        tmp_i = inputs.new(batch_size, 6 * self.hidden_size)
        tmp_h = inputs.new(batch_size, 5 * self.hidden_size)
        is_training = 1 if self.train else 0
        highway_lstm_layer.highway_lstm_forward_cuda(input_size,  # type: ignore # pylint: disable=no-member
                                                     self.hidden_size,
                                                     batch_size,
                                                     self.num_layers,
                                                     sequence_length,
                                                     inputs,
                                                     lengths,
                                                     state_accumulator,
                                                     memory_accumulator,
                                                     tmp_i,
                                                     tmp_h,
                                                     weight,
                                                     bias,
                                                     dropout_mask,
                                                     gates,
                                                     is_training)

        self.save_for_backward(inputs, lengths, weight, bias, state_accumulator,
                               memory_accumulator, dropout_mask, gates)

        # The state_accumulator has shape: (num_layers, sequence_length + 1, batch_size, hidden_size)
        # so for the output, we want the last layer and all but the first timestep, which was the
        # initial state.
        output = state_accumulator[-1, 1:, :, :]
        return output, state_accumulator[:, 1:, :, :]

    @overrides
    def backward(self, grad_output, grad_hy):  # pylint: disable=arguments-differ

        (inputs, lengths, weight, bias, state_accumulator,  # pylint: disable=unpacking-non-sequence
         memory_accumulator, dropout_mask, gates) = self.saved_tensors

        inputs = inputs.contiguous()
        sequence_length, batch_size, input_size = inputs.size()
        parameters_need_grad = 1 if self.needs_input_grad[1] else 0  # pylint: disable=unsubscriptable-object

        grad_input = inputs.new().resize_as_(inputs).zero_()
        grad_state_accumulator = inputs.new().resize_as_(state_accumulator).zero_()
        grad_memory_accumulator = inputs.new().resize_as_(memory_accumulator).zero_()
        grad_weight = inputs.new()
        grad_bias = inputs.new()
        grad_dropout = None
        grad_lengths = None
        grad_gates = None

        if parameters_need_grad:
            grad_weight.resize_as_(weight).zero_()
            grad_bias.resize_as_(bias).zero_()

        tmp_i_gates_grad = inputs.new().resize_(batch_size, 6 * self.hidden_size).zero_()
        tmp_h_gates_grad = inputs.new().resize_(batch_size, 5 * self.hidden_size).zero_()

        is_training = 1 if self.train else 0
        highway_lstm_layer.highway_lstm_backward_cuda(input_size,  # pylint: disable=no-member
                                                      self.hidden_size,
                                                      batch_size,
                                                      self.num_layers,
                                                      sequence_length,
                                                      grad_output,
                                                      lengths,
                                                      grad_state_accumulator,
                                                      grad_memory_accumulator,
                                                      inputs,
                                                      state_accumulator,
                                                      memory_accumulator,
                                                      weight,
                                                      gates,
                                                      dropout_mask,
                                                      tmp_h_gates_grad,
                                                      tmp_i_gates_grad,
                                                      grad_hy,
                                                      grad_input,
                                                      grad_weight,
                                                      grad_bias,
                                                      is_training,
                                                      parameters_need_grad)

        return (grad_input, grad_weight, grad_bias, grad_state_accumulator,
                grad_memory_accumulator, grad_dropout, grad_lengths, grad_gates)


class AlternatingHighwayLSTM(torch.nn.Module):
    """
    A stacked LSTM with LSTM layers which alternate between going forwards over
    the sequence and going backwards, with highway connections between each of
    the alternating layers. This implementation is based on the description in
    `Deep Semantic Role Labelling - What works and what's next
    <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    num_layers : int, required
        The number of stacked LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .

    Returns
    -------
    output : PackedSequence
        The outputs of the interleaved LSTMs per timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 recurrent_dropout_probability: float = 0) -> None:
        super(AlternatingHighwayLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.training = True

        # Input dimensions consider the fact that we do
        # all of the LSTM projections (and highway parts)
        # in a single matrix multiplication.
        input_projection_size = 6 * hidden_size
        state_projection_size = 5 * hidden_size
        bias_size = 5 * hidden_size

        # Here we are creating a single weight and bias with the
        # parameters for all layers unfolded into it. This is necessary
        # because unpacking and re-packing the weights inside the
        # kernel would be slow, as it would happen every time it is called.
        total_weight_size = 0
        total_bias_size = 0
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size

            input_weights = input_projection_size * layer_input_size
            state_weights = state_projection_size * hidden_size
            total_weight_size += input_weights + state_weights

            total_bias_size += bias_size

        self.weight = Parameter(torch.FloatTensor(total_weight_size))
        self.bias = Parameter(torch.FloatTensor(total_bias_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bias.data.zero_()
        weight_index = 0
        bias_index = 0
        for i in range(self.num_layers):
            input_size = self.input_size if i == 0 else self.hidden_size

            # Create a tensor of the right size and initialize it.
            init_tensor = self.weight.data.new(input_size, self.hidden_size * 6).zero_()
            block_orthogonal(init_tensor, [input_size, self.hidden_size])
            # Copy it into the flat weight.
            self.weight.data[weight_index: weight_index + init_tensor.nelement()] \
                .view_as(init_tensor).copy_(init_tensor)
            weight_index += init_tensor.nelement()

            # Same for the recurrent connection weight.
            init_tensor = self.weight.data.new(self.hidden_size, self.hidden_size * 5).zero_()
            block_orthogonal(init_tensor, [self.hidden_size, self.hidden_size])
            self.weight.data[weight_index: weight_index + init_tensor.nelement()] \
                .view_as(init_tensor).copy_(init_tensor)
            weight_index += init_tensor.nelement()

            # Set the forget bias to 1.
            self.bias.data[bias_index + self.hidden_size:bias_index + 2 * self.hidden_size].fill_(1)
            bias_index += 5 * self.hidden_size

    def forward(self, inputs, initial_state=None) -> Tuple[PackedSequence, torch.Tensor]:
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            Currently, this is ignored.

        Returns
        -------
        output_sequence : ``PackedSequence``
            The encoded sequence of shape (batch_size, sequence_length, hidden_size)
        final_states: ``torch.Tensor``
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size).
        """
        inputs, lengths = pad_packed_sequence(inputs, batch_first=False)

        sequence_length, batch_size, _ = inputs.size()
        accumulator_shape = [self.num_layers, sequence_length + 1, batch_size, self.hidden_size]
        state_accumulator = Variable(inputs.data.new(*accumulator_shape).zero_(), requires_grad=False)
        memory_accumulator = Variable(inputs.data.new(*accumulator_shape).zero_(), requires_grad=False)

        dropout_weights = inputs.data.new().resize_(self.num_layers, batch_size, self.hidden_size).fill_(1.0)
        if self.training:
            # Normalize by 1 - dropout_prob to preserve the output statistics of the layer.
            dropout_weights.bernoulli_(1 - self.recurrent_dropout_probability) \
                .div_((1 - self.recurrent_dropout_probability))

        dropout_weights = Variable(dropout_weights, requires_grad=False)
        gates = Variable(inputs.data.new().resize_(self.num_layers,
                                                   sequence_length,
                                                   batch_size, 6 * self.hidden_size))

        lengths_variable = Variable(torch.IntTensor(lengths))
        implementation = _AlternatingHighwayLSTMFunction(self.input_size,
                                                         self.hidden_size,
                                                         num_layers=self.num_layers,
                                                         train=self.training)
        output, _ = implementation(inputs, self.weight, self.bias, state_accumulator,
                                   memory_accumulator, dropout_weights, lengths_variable, gates)

        output = pack_padded_sequence(output, lengths, batch_first=False)
        return output, None
