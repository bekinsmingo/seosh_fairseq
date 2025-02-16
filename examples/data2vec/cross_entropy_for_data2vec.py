# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from fairseq.logging.meters import safe_round
from fairseq.dataclass.constants import DDP_BACKEND_CHOICES
import logging
logger = logging.getLogger(__name__)

from itertools import groupby
import editdistance
import numpy as np

@dataclass
class CrossEntropyforData2vecCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    ddp_backend: DDP_BACKEND_CHOICES = II("distributed_training.ddp_backend")
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    mwer: bool = field(
        default=False,
        metadata={"help": ""},
    )
    ctc: bool = field(
        default=False,
        metadata={"help": ""},
    )
    zero_infinity: bool = field(
        default=True,
        metadata={"help": "zero inf loss when source length <= target length"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("cross_entropy_for_data2vec", dataclass=CrossEntropyforData2vecCriterionConfig)
class CrossEntropyCriterionforData2Vec(FairseqCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False, mwer=False, ctc=False, zero_infinity=False):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.mwer = mwer
        self.ctc = ctc

        self.blank_idx = 0
        self.pad_idx = 0
        self.eos_idx = 0
        self.zero_infinity = zero_infinity

        self.eps_for_reinforce = np.finfo(np.float32).eps.item()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # net_output = model(**sample["net_input"])
        net_output, ctc_transcripts, target_padding_mask, target_for_mlm, net_output_from_nbest_list, nbest_target_padding_mask, upsampled_decoder_out, upsampled_padding_mask = model(sample)
        
        ## Total loss
        loss = torch.tensor(0).to(model.device).float()

        ## for CMLM (masked-ctc) loss
        ce_loss = torch.tensor(0).to(model.device).float()
        if model.decoder.adaptive_softmax is not None:
            ce_loss, lprobs = self.compute_adaptive_loss(model, net_output, sample, reduce=reduce)
            # TODO: adaptive softmax for evaluation mode
        elif (self.eps > 0.0) and (model.training):
            ce_loss, lprobs = self.compute_label_smoothed_loss(model, net_output, sample, target_for_mlm, reduce=reduce)
        else:
            ce_loss, lprobs = self.compute_loss(model, net_output, sample, target_for_mlm, reduce=reduce)
        loss += ce_loss

        # import pdb; pdb.set_trace()

        ## for ctc loss
        ctc_loss = torch.tensor(0)
        ctc_lprobs = None
        if self.ctc:
            self.blank_idx = model.roberta_mask_idx
            self.pad_idx = model.roberta_src_dict.pad()
            self.eos_idx = model.roberta_src_dict.eos()
            if upsampled_decoder_out.size(0) == sample["target"].size(0):
                ctc_loss, ctc_lprobs = self.compute_ctc_loss(model, upsampled_decoder_out, sample, target_padding_mask, upsampled_padding_mask)
            else:
                ctc_loss, ctc_lprobs = self.compute_ctc_loss(model, upsampled_decoder_out, sample, nbest_target_padding_mask, upsampled_padding_mask)
        loss += ctc_loss


        ## for MWER loss
        mwer_loss = torch.tensor(0)
        # self.mwer = True
        if self.mwer:
            self.pad_idx = model.roberta_src_dict.pad()
            self.eos_idx = model.roberta_src_dict.eos()
            if net_output_from_nbest_list is None:
                mwer_loss, _ = self.compute_mwer_loss(model, net_output, sample, target_padding_mask)
            else:
                mwer_loss, _ = self.compute_mwer_loss(model, net_output_from_nbest_list, sample, nbest_target_padding_mask)
        loss += mwer_loss

        # lprobs.masked_fill(target_padding_mask,self.padding_idx)
        # model.get_normalized_probs(net_output, log_probs=True).max(-1)[1].masked_fill(target_padding_mask,self.padding_idx)[:4]
        # _.max(-1)[1].masked_fill(nbest_target_padding_mask,self.padding_idx)[:4]

        '''
        (Pdb) ce_loss; ctc_loss; mwer_loss; loss
        tensor(532.1523, device='cuda:0', grad_fn=<AddBackward0>)
        tensor(99.8450, device='cuda:0', grad_fn=<SumBackward0>)
        tensor(-5.2061, device='cuda:0', grad_fn=<MeanBackward0>)
        tensor(626.7913, device='cuda:0', grad_fn=<AddBackward0>)
        '''


        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        ctc_sample_size = (upsampled_decoder_out.size(0) if self.sentence_avg else (~upsampled_padding_mask).long().sum(-1))
        if net_output_from_nbest_list is None:
            mwer_sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        else:
            mwer_sample_size = (net_output_from_nbest_list.size(0) if self.sentence_avg else sample["ntokens"]*(net_output_from_nbest_list.size(0)//sample["target"].size(0)))

        if not model.training:
            loss = torch.tensor(0).float()

        logging_output = {
            "loss": loss.data,
            "ce_loss" : ce_loss.data,
            "final_ctc_loss": utils.item(ctc_loss.data) if self.ctc else 0,
            "mwer_loss": utils.item(mwer_loss.data) if self.mwer else 0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "ctc_sample_size": ctc_sample_size,
            "mwer_sample_size": mwer_sample_size,
        }

        if not model.training:

            with torch.no_grad():
                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0

                ctc_c_err = 0
                ctc_c_len = 0
                ctc_w_errs = 0
                ctc_w_len = 0

                final_ctc_w_errs = 0
                final_ctc_w_len = 0
                final_ctc_c_err = 0
                final_ctc_c_len = 0 

                rp_ctc_wer = []
                rp_wer = []
                rp_final_ctc_wer = []

                lprobs = lprobs.argmax(dim=-1)
                lprobs = lprobs.masked_fill(target_padding_mask,self.padding_idx)

                for i, (lp, t, ctc_transcript) in enumerate(zip(lprobs, sample["target_label"] if "target_label" in sample else sample["target"], ctc_transcripts)):

                    # Processing prediction and target tensors to string
                    p = (t != model.roberta_src_dict.pad()) & (
                        t != model.roberta_src_dict.eos()
                    )
                    # for roberta output
                    targ = t[p]
                    targ_units =  model.roberta_src_dict.string(targ).replace('<pad>','').replace('<mask>','').replace('<s>','').replace('</s>','').replace('<unk>','')
                    targ_units_arr = model.bpe.decode(targ_units)

                    toks = lp
                    # toks = lp.argmax(dim=-1)
                    bpe_sentence = model.roberta_src_dict.string(toks).replace('<pad>','').replace('<mask>','').replace('<s>','').replace('</s>','').replace('<unk>','')
                    pred_units_arr = model.bpe.decode(bpe_sentence)

                    # 1. Mask-ctc CER, WER
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words=targ_units_arr.split(' ')
                    pred_words=pred_units_arr.split(' ')
                    w_errs += editdistance.eval(pred_words, targ_words)
                    w_len += len(targ_words)

                    # 2. Vanilla w2v2-ctc CER, WER
                    ctc_targ_units_arr = targ_units_arr.upper().replace('.','')
                    ctc_pred_units_arr = ctc_transcript

                    ctc_c_err += editdistance.eval(ctc_pred_units_arr, ctc_targ_units_arr)
                    ctc_c_len += len(ctc_targ_units_arr)

                    ctc_targ_words=ctc_targ_units_arr.split(' ')
                    ctc_pred_words=ctc_pred_units_arr.split(' ')
                    ctc_w_errs += editdistance.eval(ctc_pred_words, ctc_targ_words)
                    ctc_w_len += len(targ_words)

                    # for report
                    rp_ctc_wer.append(safe_round(editdistance.eval(ctc_pred_words, ctc_targ_words)*100/len(targ_words),3))
                    rp_wer.append(safe_round(editdistance.eval(pred_words, targ_words)*100/len(targ_words),3))

                    if self.ctc:
                        ctc_lprob = ctc_lprobs[i]
                        _, ctc_ids = torch.exp(ctc_lprob).max(dim=-1)
                        ctc_pred = torch.stack([x[0] for x in groupby(ctc_ids)])

                        bpe_sentence = model.roberta_src_dict.string(ctc_pred).replace('<pad>','').replace('<mask>','').replace('<s>','').replace('</s>','').replace('<unk>','')
                        
                        # 3. Final CTC CER, WER
                        final_ctc_pred_units_arr = model.bpe.decode(bpe_sentence)
                        final_ctc_pred_words=final_ctc_pred_units_arr.split(' ')

                        final_ctc_c_err += editdistance.eval(final_ctc_pred_units_arr, targ_units_arr)
                        final_ctc_c_len += len(targ_units_arr)
                        final_ctc_w_errs += editdistance.eval(final_ctc_pred_words, targ_words)
                        final_ctc_w_len += len(targ_words)

                        rp_final_ctc_wer.append(safe_round(editdistance.eval(final_ctc_pred_words, targ_words)*100/len(targ_words),3))

                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

                logging_output["ctc_w_errors"] = ctc_w_errs
                logging_output["ctc_w_total"] = ctc_w_len
                logging_output["ctc_c_errors"] = ctc_c_err
                logging_output["ctc_c_total"] = ctc_c_len

                logging_output["final_ctc_w_errors"] = final_ctc_w_errs
                logging_output["final_ctc_w_total"] = final_ctc_w_len
                logging_output["final_ctc_c_errors"] = final_ctc_c_err
                logging_output["final_ctc_c_total"] = final_ctc_c_len

                logger.info("--------------------- ctc ---------------------")
                logger.info("TARG : " + ctc_targ_units_arr)
                logger.info("HYPO : " + ctc_pred_units_arr)
                logger.info("BATCH WER : {}".format(rp_ctc_wer))
                logger.info("BATCH WER : {}".format(safe_round(ctc_w_errs*100/ctc_w_len,3)))
                logger.info("-------------------- Roberta ------------------")
                logger.info("TARG : " + targ_units_arr)
                logger.info("HYPO : " + pred_units_arr)
                logger.info("BATCH WER : {}".format(rp_wer))
                logger.info("BATCH WER : {}".format(safe_round(w_errs*100/w_len,3)))
                if self.ctc:
                    logger.info("-------------------- final ctc ------------------")
                    logger.info("TARG : " + targ_units_arr)
                    logger.info("HYPO : " + final_ctc_pred_units_arr)
                    logger.info("BATCH WER : {}".format(rp_final_ctc_wer))
                    logger.info("BATCH WER : {}".format(safe_round(final_ctc_w_errs*100/w_len,3)))
                logger.info("-----------------------------------------------")

        return loss, sample_size, logging_output


    def compute_loss(self, model, net_output, sample, target_for_mlm, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        # target = target_for_mlm
        if target is not None:
            if target.size(1) > lprobs.size(1):
                target = target[:lprobs.size(0)]
            elif target.size(1) < lprobs.size(1):
                tmp = torch.ones(target.size(0),lprobs.size(1) - target.size(1)).long()
                target = torch.cat((target,tmp.to(model.device)),-1) 
        # loss = torch.tensor(0)
        if model.training:
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = target.view(-1)
            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )
            # import pdb; pdb.set_trace()
        else:
            loss = torch.tensor(0).float()
        return loss, lprobs


    def compute_label_smoothed_loss(self, model, net_output, sample, target_for_mlm, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True) # torch.Size([8, 48, 50264])
        target = model.get_targets(sample, net_output)
        # target = target_for_mlm
        if target is not None:
            if target.size(1) > lprobs.size(1):
                target = target[:lprobs.size(0)]
            elif target.size(1) < lprobs.size(1):
                tmp = torch.ones(target.size(0),lprobs.size(1) - target.size(1)).long()
                target = torch.cat((target,tmp.to(model.device)),-1) 
        # import pdb; pdb.set_trace()
        # model.get_normalized_probs(net_output, log_probs=True).size() ; model.get_targets(sample, net_output).size(); target.size();lprobs.size()
        if model.training:
            lprobs = lprobs.view(-1, lprobs.size(-1)) # torch.Size([384, 50264])
            target = target.view(-1)
            if self.ignore_prefix_size > 0:
                # lprobs: B x T x C
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            # return loss, nll_loss
        else:
            loss = torch.tensor(0).float()
        # import pdb; pdb.set_trace()
        return loss, lprobs


    def compute_mwer_loss(self, model, net_output, sample, target_padding_mask):
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous() # B, T, C

        # lprobs_ = lprobs.transpose(0, 1) # T, B, C

        target = sample["target"]
        if target.size(0) != lprobs.size(0):
            repeated_target = torch.zeros(lprobs.size(0),target.size(1))
            for i in range(lprobs.size(0)//target.size(0)):
                repeated_target[target.size(0)*i:target.size(0)*(i+1)] = target.clone()
            target = repeated_target.long()

        argmax_preds = lprobs.argmax(dim=-1)
        argmax_preds = argmax_preds.masked_fill(target_padding_mask,self.padding_idx)

        ## get WERs for MWER (REINFORCE) loss
        wers = []
        for i, (lp, t) in enumerate(zip(argmax_preds, target)):
            # Processing prediction and target tensors to string
            p = (t != model.roberta_src_dict.pad()) & (t != model.roberta_src_dict.eos())
            targ = t[p]
            targ_units =  model.roberta_src_dict.string(targ).replace('<pad>','').replace('<mask>','').replace('<s>','').replace('</s>','').replace('<unk>','')
            targ_units_arr = model.bpe.decode(targ_units)


            toks = lp
            # toks = lp.argmax(dim=-1)
            bpe_sentence = model.roberta_src_dict.string(toks).replace('<pad>','').replace('<mask>','').replace('<s>','').replace('</s>','').replace('<unk>','')
            bpe_sentence = bpe_sentence.replace('madeupword0000','').replace('madeupword0001','').replace('madeupword0002','').replace('madeupword0003','').replace('madeupword0004','').replace('madeupword0005','').replace('madeupword0006','')
            # import pdb; pdb.set_trace()
            pred_units_arr = model.bpe.decode(bpe_sentence)

            # calculate WER
            targ_words=targ_units_arr.split(' ')
            pred_words=pred_units_arr.split(' ')
            w_errs = editdistance.eval(pred_words, targ_words)
            w_len = len(targ_words)
            wers.append(safe_round(w_errs*100/w_len,3))

        wers_ = torch.FloatTensor(wers)
        wers = (wers_ - torch.mean(wers_)) / (wers_.std() + self.eps_for_reinforce)
        wers = -wers # because WERs are not good Reward, the lower is the better
        # import pdb; pdb.set_trace()
        '''
        (Pdb) wers_; wers
        tensor([ 3.0300,  2.3810,  4.7620,  0.0000,  0.0000, 15.7890,  0.0000,  0.0000,
                3.0300, 11.9050,  0.0000,  4.1670,  0.0000, 10.5260, 13.8890,  6.4520])
        tensor([ 0.3168,  0.4366, -0.0030,  0.8762,  0.8762, -2.0388,  0.8762,  0.8762,
                0.3168, -1.3218,  0.8762,  0.1068,  0.8762, -1.0672, -1.6880, -0.3150])
        '''
        
        selected_lprobs = torch.gather(lprobs,-1,lprobs.max(-1)[1].unsqueeze(-1)).squeeze(-1)
        # selected_lprobs = selected_lprobs.masked_fill(target_padding_mask,0)
        # selected_lprobs = selected_lprobs.masked_fill(target_padding_mask,-torch.tensor(float('inf')))
        mwer_loss = [ -lprob[:(~mask).sum()].sum() * wer for lprob, mask, wer in zip(selected_lprobs, target_padding_mask, wers)] # original implementation : (NLL * Reward)
        loss = torch.stack(mwer_loss).mean()
        '''
        (Pdb) torch.stack(mwer_loss); loss
        tensor([  2.7242,   6.6850,  -0.0309,   6.2604,   7.2141, -22.5037,   9.5414, 7.2978,   
                3.5687, -23.5570,   8.0708,   0.9697,   6.9169,  -8.5435, -29.0875,  -3.6746], device='cuda:0', grad_fn=<StackBackward>)
        tensor(-1.7593, device='cuda:0', grad_fn=<MeanBackward0>)
        '''
        # import pdb; pdb.set_trace()

        return loss, lprobs


    def compute_ctc_loss(self, model, net_output, sample, target_padding_mask, upsampled_padding_mask):
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (B, T, C) from the encoder # e.g. torch.Size([16, 96, 50265])

        lprobs_ = lprobs.transpose(0, 1) # T, B, C, torch.Size([96, 16, 50265])

        non_padding_mask = ~upsampled_padding_mask
        input_lengths = non_padding_mask.long().sum(-1)

        target = sample["target"]
        if target.size(0) != lprobs.size(0):
            repeated_target = torch.zeros(lprobs.size(0),target.size(1))
            for i in range(lprobs.size(0)//target.size(0)):
                repeated_target[target.size(0)*i:target.size(0)*(i+1)] = target.clone()
            target = repeated_target.long()

        # pad_mask = ~target_padding_mask
        pad_mask = (target != self.pad_idx)
        targets_flat = target.masked_select(pad_mask)
        target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs_,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        return loss, lprobs


    def compute_adaptive_loss(self, model, net_output, sample, reduce=True):
        adaptive_softmax = model.decoder.adaptive_softmax
        orig_target = model.get_targets(sample, net_output)
        # nsentences = orig_target.size(0)
        orig_target = orig_target.view(-1)

        bsz = orig_target.size(0)

        logits, target = adaptive_softmax(net_output, orig_target)
        assert len(target) == len(logits)

        loss = net_output.new(1 if reduce else bsz).zero_()

        for i in range(len(target)):
            if target[i] is not None:
                assert target[i].min() >= 0 and target[i].max() <= logits[i].size(1)
                loss += F.cross_entropy(
                    logits[i],
                    target[i],
                    ignore_index=self.padding_idx,
                    reduction="sum" if reduce else "none",
                )

        # orig = utils.strip_pad(orig_target, self.padding_idx)
        # ntokens = orig.numel()
        # sample_size = sample["target"].size(0) if self.sentence_avg else ntokens

        return loss, logits

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        final_ctc_loss_sum = utils.item(sum(log.get("final_ctc_loss", 0) for log in logging_outputs))
        mwer_loss_sum = utils.item(sum(log.get("mwer_loss", 0) for log in logging_outputs))
        
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        ctc_sample_size = sum(log.get("ctc_sample_size", 0) for log in logging_outputs)
        mwer_sample_size = sum(log.get("mwer_sample_size", 0) for log in logging_outputs)
        nsentences = utils.item(sum(log.get("nsentences", 0) for log in logging_outputs))

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("ce_loss", ce_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("final_ctc_loss", final_ctc_loss_sum / ctc_sample_size / math.log(2), ctc_sample_size, round=3)
        metrics.log_scalar("mwer_loss", mwer_loss_sum / mwer_sample_size / math.log(2), mwer_sample_size, round=3)
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        if sample_size != ntokens:
            metrics.log_scalar("nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))
        else:
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)


        ctc_c_errors = sum(log.get("ctc_c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_c_errors", ctc_c_errors)
        ctc_c_total = sum(log.get("ctc_c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_c_total", ctc_c_total)
        ctc_w_errors = sum(log.get("ctc_w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_w_errors", ctc_w_errors)
        ctc_w_total = sum(log.get("ctc_w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_w_total", ctc_w_total)


        final_ctc_c_errors = sum(log.get("final_ctc_c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_final_ctc_c_errors", final_ctc_c_errors)
        final_ctc_c_total = sum(log.get("final_ctc_c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_final_ctc_c_total", final_ctc_c_total)
        final_ctc_w_errors = sum(log.get("final_ctc_w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_final_ctc_w_errors", final_ctc_w_errors)
        final_ctc_w_total = sum(log.get("final_ctc_w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_final_ctc_w_total", final_ctc_w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

        if ctc_c_total > 0:
            metrics.log_derived(
                "ctc_uer",
                lambda meters: safe_round(
                    meters["_ctc_c_errors"].sum * 100.0 / meters["_ctc_c_total"].sum, 3
                )
                if meters["_ctc_c_total"].sum > 0
                else float("nan"),
            )
        if ctc_w_total > 0:
            metrics.log_derived(
                "ctc_wer",
                lambda meters: safe_round(
                    meters["_ctc_w_errors"].sum * 100.0 / meters["_ctc_w_total"].sum, 3
                )
                if meters["_ctc_w_total"].sum > 0
                else float("nan"),
            )


        if final_ctc_c_total > 0:
            metrics.log_derived(
                "final_ctc_uer",
                lambda meters: safe_round(
                    meters["_final_ctc_c_errors"].sum * 100.0 / meters["_final_ctc_c_total"].sum, 3
                )
                if meters["_final_ctc_c_total"].sum > 0
                else float("nan"),
            )
        if final_ctc_w_total > 0:
            metrics.log_derived(
                "final_ctc_wer",
                lambda meters: safe_round(
                    meters["_final_ctc_w_errors"].sum * 100.0 / meters["_final_ctc_w_total"].sum, 3
                )
                if meters["_final_ctc_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
