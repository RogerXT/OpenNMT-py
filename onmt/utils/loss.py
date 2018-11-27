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
        # print tgt_vocab.stoi['the']
        # print tgt_vocab.itos[the_index]

        """
        content_idx = set()
        for w in content_5:
            w1 = tgt_vocab.stoi[w[0]]
            w2 = tgt_vocab.stoi[w[1]]
            w3 = tgt_vocab.stoi[w[2]]
            w4 = tgt_vocab.stoi[w[3]]
            w5 = tgt_vocab.stoi[w[3]]
            if w1 != 0 and w2 != 0 and w3 != 0 and w5 != 0:
                content_idx.add((w1, w2, w3, w4, w5))

        print content_idx
        """

        self._content = {
             5633, 939, 516, 2054, 8705, 15882, 3410, 8717, 5134, 9743, 1040, 188, 11285, 512, 13335, 536, 4633, 444,
             8223, 1568, 4272, 3121, 22054, 2600, 554, 1543, 10798, 3119, 22065, 6706, 179, 3125, 1590, 40507, 19005,
             5728, 3650, 8259, 2628, 182, 865, 22088, 2633, 6839, 13312, 2640, 12899, 3668, 1622, 39511, 42585, 13402,
             7951, 6750, 5215, 1120, 1552, 7300, 699, 12905, 2987, 3183, 3652, 15986, 7283, 1140, 21025, 42543, 2167,
             45690, 1660, 23453, 1664, 328, 3714, 21489, 5398, 7302, 2753, 4908, 6796, 1167, 1681, 39645, 3221, 662,
             5401, 3268, 15983, 4254, 2720, 46619, 1188, 13479, 22300, 455, 6791, 8878, 2224, 46280, 4274, 1139, 2742,
             4791, 1209, 698, 187, 6844, 189, 6335, 4800, 44225, 8898, 1732, 1223, 44232, 4809, 1524, 209, 21203, 2496,
             8917, 31554, 6361, 3451, 220, 6877, 11486, 3810, 1763, 228, 1252, 1256, 233, 44781, 1775, 243, 244, 8949,
             2294, 16119, 3469, 761, 3322, 2968, 4863, 4480, 772, 12758, 262, 2348, 5106, 5389, 11227, 2861, 3344, 1748,
             1301, 2326, 792, 23321, 283, 5404, 2845, 21280, 2851, 5924, 163, 6449, 19753, 47915, 7980, 2866, 6446,
             13103, 16176, 1329, 9010, 2871, 6968, 3039, 7996, 6463, 832, 1346, 2955, 10054, 1864, 1867, 7500, 1871,
             3921, 6994, 7395, 2215, 3414, 13497, 37722, 5982, 6098, 8033, 15721, 3436, 1901, 16396, 368, 2417, 4793,
             3444, 4472, 379, 16908, 4991, 1408, 6530, 4484, 901, 45284, 8855, 11148, 11586, 398, 400, 40432, 11154,
             6884, 408, 921, 3484, 925, 42399, 23456, 1953, 10150, 16295, 1448, 1449, 23979, 941, 12719, 10056, 8115,
             20404, 8117, 22967, 4852, 2975, 8124, 5054, 3008, 41120, 75, 1989, 3014, 23495, 3058, 26444, 7415, 972,
             4046, 16338, 3235, 5076, 3030, 2007, 6308, 851, 21979, 3548, 7892, 1502, 479, 488, 4586, 15851, 40429, 496,
             16296, 2546, 15548, 2548, 1551, 8185, 2043}

        self._content_2 = {
             (1238, 835), (1304, 7304), (4910, 3216), (2561, 855), (200, 3648), (216, 143), (4915, 133), (1816, 545),
             (3404, 863), (1523, 886), (782, 338), (219, 12968), (15938, 7950), (127, 7007), (23719, 226),
             (6017, 11381), (223, 149), (466, 1199), (578, 296), (3554, 1525), (543, 11216), (846, 257), (200, 1726),
             (5143, 2325), (2130, 5372), (3560, 860), (2684, 2605), (39382, 21192), (389, 254), (2046, 3692),
             (7443, 3025), (159, 360), (1025, 1418), (539, 5764), (7488, 7562), (3303, 17), (176, 1364), (2379, 3864),
             (2122, 992), (1006, 230), (372, 5178), (2743, 570), (9771, 1841), (372, 515), (1169, 1013), (4, 2400),
             (1266, 6117), (1357, 12829), (1970, 5947), (3162, 169), (605, 39446), (1199, 4628), (522, 3829),
             (1143, 687), (20746, 1825), (146, 5138), (459, 6162), (3912, 2064), (5112, 821), (2746, 39954)}

        self._content_3 = {
             (451, 17, 831), (8217, 8, 8217), (231, 4, 2592), (2961, 7, 2136), (4, 375, 3538), (11016, 7, 72),
             (522, 26, 7976), (1259, 7, 6471), (1442, 2487, 1888)}

        self._content_4 = (4142, 7, 12, 338)

        self._content_5 = {(331, 7, 4, 117, 117), (6531, 97, 10, 4, 4)}


        """
        self._content = torch.tensor(
            [2054, 16396, 2064, 6162, 8217, 8223, 4142, 8259, 72, 2122, 6839, 2130, 2136, 698, 97, 5138, 117, 2167, 127,
             133, 13335, 143, 146, 149, 4254, 159, 41120, 163, 6308, 2215, 169, 4272, 4274, 179, 182, 187, 188, 189,
             6335, 200, 209, 216, 6361, 219, 220, 223, 226, 45284, 230, 231, 233, 243, 244, 2294, 1748, 254, 257, 262,
             20746, 2325, 2326, 283, 296, 2348, 6446, 6877, 6449, 1418, 6463, 6471, 328, 331, 338, 342, 2400, 360, 368,
             2417, 372, 375, 4472, 379, 4480, 6530, 6531, 4484, 389, 398, 400, 408, 47518, 12719, 2487, 444, 2496, 75,
             455, 459, 466, 12758, 479, 488, 4586, 496, 2546, 2548, 512, 2561, 515, 516, 522, 16908, 8717, 4628, 39446,
             536, 4633, 539, 12829, 543, 2592, 545, 2600, 554, 2605, 10798, 6706, 570, 19005, 578, 2628, 4791, 2633,
             22967, 2640, 39511, 605, 6750, 12899, 12905, 21025, 45690, 2684, 6791, 6796, 3864, 451, 662, 8855, 2720,
             12968, 8878, 687, 1139, 2742, 2743, 4793, 2746, 699, 6844, 4800, 2753, 8898, 710, 21192, 4809, 21203, 8917,
             39645, 6884, 4852, 8949, 761, 4863, 772, 39382, 11016, 782, 4879, 792, 23321, 2845, 21280, 2851, 47915,
             4908, 9010, 4910, 13103, 2866, 4915, 821, 2871, 6968, 825, 831, 832, 31554, 835, 846, 6994, 851, 855,
             37722, 860, 863, 865, 886, 4991, 901, 2955, 11148, 2961, 11154, 2968, 921, 925, 2975, 23456, 939, 941,
             10056, 3912, 5054, 3008, 3014, 23495, 972, 11216, 3025, 5076, 3030, 11227, 3039, 992, 13479, 1006, 21489,
             5106, 1013, 5112, 13312, 1025, 8705, 5134, 1040, 39954, 11285, 5143, 2224, 3112, 3119, 3121, 3650, 3125,
             5178, 7007, 13497, 13402, 1551, 5215, 1120, 3183, 7283, 1140, 11381, 1143, 7300, 7302, 7304, 1167, 3216,
             1169, 3221, 16338, 3235, 1188, 23719, 176, 1199, 46280, 1209, 15548, 44225, 3268, 1223, 44232, 1238, 11486,
             7395, 1252, 3303, 1256, 1259, 1266, 7415, 3322, 5372, 3326, 5389, 557, 3344, 7443, 1301, 5398, 1304, 5401,
             5404, 19753, 1329, 7488, 1346, 3404, 1357, 3410, 1364, 3414, 228, 3431, 15721, 3436, 3444, 3451, 1408,
             7562, 3469, 3484, 42399, 1442, 1448, 1449, 23979, 23453, 3538, 21979, 3548, 1502, 3554, 3560, 15851, 40429,
             40432, 1523, 1524, 1525, 5633, 2987, 1543, 15882, 9743, 1552, 46619, 1568, 22054, 9771, 42543, 22065, 1590,
             40507, 3648, 15938, 3652, 22088, 3668, 1622, 42585, 5728, 3692, 15983, 15986, 3708, 1664, 3714, 5764, 1681,
             1693, 3698, 1726, 1732, 3162, 7892, 3810, 1763, 1660, 44781, 1775, 3829, 16119, 7950, 7951, 1816, 22300,
             1825, 5924, 7976, 7980, 16176, 1841, 5947, 7996, 10054, 1864, 1867, 26444, 1871, 3921, 5982, 1888, 8033,
             3943, 1901, 6017, 11586, 1953, 2861, 10150, 16295, 16296, 3058, 1970, 8115, 20404, 8117, 8124, 2379, 1989,
             7500, 4046, 6098, 2007, 6117, 8185, 2043, 2046]).to(self._device)
        """

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

        #pred = scores.max(1)[1]

        # x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).to(device)
        # y = torch.tensor([2, 3, 5]).to(device)
        # check whether element in x is in y
        # z = x.view(1, -1).eq(y.view(-1, 1)).sum(0).float()

        w = 2.0
        b = 1.0

        #weights = gtruth.view(1, -1).eq(self._content.view(-1, 1)).sum(0).float() * w + b

        weights = []
        i = 0
        n = len(gtruth)

        while i < n:
            if i < n - 4:
                if (gtruth[i].item(), gtruth[i+1].item(), gtruth[i+2].item(), gtruth[i+3].item(), gtruth[i+4].item()) in self._content_5:
                    weights += [1.0, 1.0, 1.0, 1.0, 1.0]
                    i += 5
                    continue

            if i < n - 3:
                if (gtruth[i].item(), gtruth[i+1].item(), gtruth[i+2].item(), gtruth[i+3].item()) == self._content_4:
                    weights += [1.0, 1.0, 1.0, 1.0]
                    i += 4
                    continue

            if i < n - 2:
                if (gtruth[i].item(), gtruth[i+1].item(), gtruth[i+2].item()) in self._content_3:
                    weights += [1.0, 1.0, 1.0]
                    i += 3
                    continue

            if i < n - 1:
                if (gtruth[i].item(), gtruth[i+1].item()) in self._content_2:
                    weights += [1.0, 1.0]
                    i += 2
                    continue

            if gtruth[i].item() in self._content:
                weights.append(1.0)
            else:
                weights.append(0.0)
            i += 1

        weights = torch.tensor(weights).to(self._device) * w + b

        loss = (loss * weights).sum()

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
