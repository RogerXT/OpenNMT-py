"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
import onmt.inputters as inputters
from onmt.modules.sparse_losses import SparsemaxLoss


def build_loss_compute(model, tgt_vocab, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength)
    else:
        compute = NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing if train else 0.0)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        # print output.shape
        batch_stats = onmt.utils.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            #print loss
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, size_average=False)


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, len(tgt_vocab), ignore_index=self.padding_idx
            )
        elif self.sparse:
            self.criterion = SparsemaxLoss(
                ignore_index=self.padding_idx, size_average=False
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, size_average=False, reduce=False
            )
        self._loss = nn.NLLLoss(ignore_index=self.padding_idx, size_average=False)
        self._device = torch.device("cuda")
        self._ignore = {0, 43, 5, 6, 265, 13, 14, 19, 20, 21, 22, 23, 24, 29, 30, 31, 32, 34, 36, 37, 39, 555, 46,
                        47, 48, 50, 52, 55, 56, 59, 63, 288, 74, 79, 2303}
        # 0 means <unk>
        self._content = {4, 2054, 7, 8, 9, 10, 16396, 2064, 17, 49170, 4119, 8217, 26, 28, 8223, 35, 4137, 4142,
                             2739, 60, 8259, 72, 12, 2122, 6839, 2130, 697, 2136, 89, 698, 97, 6162, 22642, 117, 2167,
                             127, 133, 13335, 8898, 143, 146, 149, 4254, 159, 41120, 162, 163, 6308, 2417, 169, 176,
                             4274, 179, 182, 2234, 187, 188, 189, 191, 200, 209, 216, 6361, 219, 220, 223, 225, 226,
                             45284, 230, 231, 233, 236, 237, 2287, 243, 244, 2294, 1748, 254, 257, 262, 20746, 1411,
                             2325, 2326, 283, 296, 2348, 6446, 6877, 6449, 1418, 6463, 326, 6471, 328, 6473, 331, 338,
                             12686, 342, 557, 4447, 2400, 360, 2414, 368, 369, 372, 375, 4472, 379, 382, 4480, 6530,
                             6531, 4484, 389, 394, 695, 398, 400, 2450, 408, 47518, 2468, 425, 12719, 2487, 46280, 444,
                             2496, 75, 455, 459, 466, 39382, 479, 488, 4586, 496, 2546, 2548, 11337, 512, 2561, 45570,
                             515, 516, 518, 521, 522, 16908, 4791, 4628, 39446, 536, 4633, 539, 540, 12829, 543, 2592,
                             545, 2600, 2602, 2605, 10798, 6706, 570, 19005, 578, 2628, 4680, 2633, 22967, 21072, 597,
                             39511, 12888, 783, 605, 6750, 12899, 6760, 617, 21025, 2678, 45690, 2684, 2691, 6791, 6796,
                             3864, 659, 662, 8855, 2720, 8872, 8878, 687, 1139, 1455, 2742, 2743, 4793, 2746, 699, 6844,
                             4800, 2753, 5138, 710, 21192, 4809, 21203, 8917, 10970, 731, 39645, 2782, 6884, 4852, 8949,
                             761, 39039, 766, 4863, 45867, 772, 12758, 8966, 11016, 782, 4879, 6930, 792, 23321, 2845,
                             21280, 802, 2851, 2853, 47915, 4908, 9010, 4910, 13103, 2866, 4915, 821, 2871, 6968, 825,
                             828, 831, 832, 31554, 835, 846, 6994, 851, 855, 37722, 860, 863, 6098, 865, 878, 886, 490,
                             4991, 901, 2955, 11148, 2961, 11154, 2968, 921, 922, 23453, 2975, 23456, 9125, 939, 941,
                             10056, 3912, 952, 5054, 3008, 3014, 23495, 972, 11216, 3025, 978, 5076, 3030, 11227, 2215,
                             3039, 992, 13479, 1006, 21489, 5106, 48116, 1013, 1015, 5112, 3067, 21502, 13312, 1025,
                             513, 5134, 1040, 39954, 11285, 5143, 3103, 1057, 1058, 3112, 3119, 3121, 1074, 3650, 3125,
                             5178, 7007, 21572, 21177, 8717, 13497, 13402, 1551, 5215, 1120, 3183, 7283, 1140, 11381,
                             1143, 6335, 7300, 7302, 7304, 11405, 1167, 3216, 1169, 3221, 3268, 3235, 1188, 23719,
                             42153, 1199, 1201, 1209, 2591, 15548, 44225, 1220, 1223, 44232, 23756, 1229, 7378, 1238,
                             11486, 7395, 1252, 3303, 1256, 2224, 451, 3313, 1266, 7415, 3322, 895, 5372, 554, 3326,
                             5376, 3339, 5389, 16941, 3344, 7443, 1301, 5398, 1304, 5401, 5404, 5410, 11555, 3364,
                             19753, 4272, 1329, 8705, 7488, 1346, 3404, 1357, 23887, 3410, 1364, 3414, 228, 3431, 15721,
                             3436, 3444, 3446, 3451, 1408, 1259, 15751, 7562, 1419, 3469, 3484, 42399, 1442, 21925,
                             1448, 1449, 23979, 925, 1469, 3518, 1474, 42441, 3538, 21979, 3548, 1502, 2640, 3554, 3560,
                             15851, 40429, 40432, 1523, 1524, 1525, 5633, 2987, 1543, 15882, 9743, 1552, 46619, 1568,
                             22054, 9771, 42543, 22065, 1590, 40507, 9789, 3648, 15938, 3652, 22088, 3668, 1622, 42585,
                             7439, 5728, 1635, 3692, 15983, 15986, 12905, 3708, 1664, 3714, 5764, 9863, 1681, 1693,
                             3698, 16052, 1726, 1732, 3162, 7892, 1316, 976, 3810, 1763, 1660, 44781, 1775, 3829, 16119,
                             16125, 40712, 9993, 7950, 7951, 1816, 22300, 10016, 1825, 5924, 7976, 7980, 16176, 1841,
                             3897, 5947, 7996, 10054, 1864, 1867, 26444, 1871, 3921, 5982, 1888, 8033, 8038, 3943, 1901,
                             22397, 6017, 11586, 10131, 40855, 1224, 1953, 2861, 10150, 16295, 16296, 3058, 1968, 1970,
                             8115, 20404, 8117, 8124, 2379, 1989, 7500, 45004, 4046, 16338, 2007, 6117, 12968, 16375,
                             8185, 2043, 2046}
        # print tgt_vocab.stoi['the']
        # print tgt_vocab.itos[the_index]

    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, target):
        bottled_output = self._bottle(output)
        if self.sparse:
            # for sparsemax loss, the loss function operates on the raw output
            # vector, not a probability vector. Hence it's only necessary to
            # apply the first part of the generator here.
            scores = self.generator[0](bottled_output)
        else:
            scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        loss = self.criterion(scores, gtruth)
        # loss.shape == gtruth.shape == pred.shape

        pred = scores.max(1)[1]
        wrong = pred != gtruth

        weights = []
        for i in range(len(wrong)):
            if wrong[i]:
                id = int(gtruth[i])
                if id in self._ignore:
                    weights.append(0.5)
                elif id in self._content:
                    weights.append(2.0)
                else:
                    weights.append(1.0)
                #if not loss[i]:
                #    loss[i] += 1.0
            else:
                weights.append(0.2) # loss is 0

        loss = (loss * torch.tensor(weights).to(self._device)).sum()

        stats = self._stats(loss.clone(), scores, gtruth)
        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
