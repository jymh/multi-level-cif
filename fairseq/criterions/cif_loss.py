#Implemented by RYM

import logging
from omegaconf import II
import math
from dataclasses import dataclass, field
import editdistance

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import safe_round

logger = logging.getLogger(__name__)

@dataclass
class CifLossCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )

    sentence_avg: bool = II("optimization.sentence_avg")

    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
                    "wordpiece, BPE symbols, etc. "
                    "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )

    ctc_weight: float = field(
        default = 0.3,
        metadata = {"help": "weight of ctc loss in cif loss"},
    )

    qua_weight: float = field(
        default = 1.0,
        metadata = {"help": "weight of quantity loss in cif loss"},
    )

    apply_quantity_loss: bool = field(
        default = True,
        metadata = {"help": "whether to apply quantity loss"},
    )

    apply_ctc_loss: bool = field(
        default = True,
        metadata = {"help": "whether to apply ctc loss"},
    )


@register_criterion("cif_loss", dataclass=CifLossCriterionConfig)
class CifLossCriterion(FairseqCriterion):
    def __init__(self, cfg: CifLossCriterionConfig, task):
        super().__init__(task)

        self.sentence_avg = cfg.sentence_avg
        self.post_process = cfg.post_process
        self.ctc_weight = cfg.ctc_weight
        self.qua_weight = cfg.qua_weight

        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )

        tgt_dict = task.target_dictionary
        self.pad_idx = tgt_dict.pad() if tgt_dict is not None else -100

        self.eos_idx = task.target_dictionary.eos()
        self.zero_infinity = cfg.zero_infinity
        self.apply_ctc_loss = cfg.apply_ctc_loss
        self.apply_qua_loss = cfg.apply_quantity_loss

    def forward(self, model, sample, reduce=True):
        """ L = L_ce + L_qua + L_ctc"""

        # logger.info("sample {}".format(sample))
        net_output = model(target_lengths=sample["target_lengths"]+1, **sample["net_input"])
        encoder_outputs = net_output["encoder_outputs"] # for ctc loss
        joint_outputs = net_output["joint_outputs"] # for ce loss
        qua_out = joint_outputs["qua_out"] # for quantity loss

        """move eos token from the last location to the end of the real sentence"""
        target = sample["target"]
        batch_size = target.size(0)
        pad_mask = ((target != self.pad_idx) & (target != self.eos_idx)).int()
        add_eos_idx = ((target * pad_mask) != 0).int().sum(-1).unsqueeze(dim=-1)  # B x 1
        add_one_hot_tensor = torch.zeros(
            batch_size, pad_mask.size(1)
        ).int().cuda().scatter_(1, add_eos_idx, 1) * self.eos_idx
        adjusted_target = torch.where(
            ((target.int() * pad_mask) + add_one_hot_tensor) == 0,
            torch.ones_like(target).int().cuda() * self.pad_idx,
            (target.int() * pad_mask) + add_one_hot_tensor,
        )

        qua_loss = torch.tensor(0.0)
        if self.apply_qua_loss:
            qua_loss = self.computeQualoss(qua_out, sample)

        ctc_loss = torch.tensor(0.0)
        if self.apply_ctc_loss:
            ctc_loss = self.computeCtcloss(model, encoder_outputs, sample, adjusted_target)

        ce_loss = self.computeCEloss(model, joint_outputs, sample, adjusted_target)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        loss = ce_loss + self.qua_weight * qua_loss + self.ctc_weight * ctc_loss

        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "qua_loss": utils.item(qua_loss.data),
            "ce_loss": utils.item(ce_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
        }

        # Evaluate on valid sets
        if not model.training:
            with torch.no_grad():
                ce_lprobs = model.get_probs_from_logits(joint_outputs["joint_out"], log_probs=True)
                lprobs = ce_lprobs.float().contiguous().cpu()
                cif_lengths = joint_outputs["padding_mask"].int().sum(dim=-1) # B x T

                c_err = 0
                c_len = 0
                w_err = 0
                w_len = 0
                wv_err = 0
                for lp, t, inp_l in zip(
                    lprobs,
                    adjusted_target,
                    cif_lengths
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    p = (t != self.pad_idx) & (t != self.eos_idx)
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    if min(lp.shape) == 0:
                        toks = targ
                    else:
                        toks = lp.argmax(dim=-1)

                    pred_units_arr = \
                        toks[(toks != self.blank_idx) & (toks != self.pad_idx) & (toks != self.eos_idx)].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    # Calculate word error
                    dist = editdistance.eval(pred_words_raw, targ_words)
                    w_err += dist
                    wv_err += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_err
                logging_output["w_errors"] = w_err
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    def computeQualoss(self, qua_out, sample):
        target_lengths = sample["target_lengths"]
        target_with_eos_lengths = target_lengths + 1
        qua_loss = torch.abs(qua_out - target_with_eos_lengths).sum()

        return qua_loss


    def computeCEloss(self, model, joint_outputs, sample, adjusted_target, reduce=True):
        joint_out = joint_outputs["joint_out"]
        lprobs = model.get_probs_from_logits(joint_out, log_probs=True)

        cif_max_len = joint_outputs["padding_mask"].size(1)

        assert cif_max_len == lprobs.size(1), "cif output padding mask does not match lprob"

        target_with_eos_lengths = sample["target_lengths"] + 1
        target_max_len = target_with_eos_lengths.max()
        min_len = min(cif_max_len, target_max_len)
        if target_max_len != cif_max_len:
            lprobs = lprobs[:, :min_len, :]
            adjusted_target = adjusted_target[:, :min_len]

        lprobs = lprobs.contiguous().view(-1, lprobs.size(-1))
        target_probs = adjusted_target.contiguous().view(-1)

        # logger.info("CE loss lprobs size {}".format(lprobs.shape))
        # logger.info("CE loss target size {}".format(target_probs.shape))

        ce_loss = F.nll_loss(
            lprobs,
            target_probs.long(),
            ignore_index=self.pad_idx,
            reduction="sum" if reduce else "none",
        )

        return ce_loss

    def computeCtcloss(self, model, encoder_outputs, sample, adjusted_target, reduce=True):
        encoder_out = encoder_outputs["encoder_out"]
        lprobs = model.get_probs_from_logits(
            encoder_out, log_probs=True
        ).contiguous()

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if encoder_outputs["padding_mask"] is not None:
                non_padding_mask = ~encoder_outputs["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )


        adjusted_pad_mask = (adjusted_target != self.pad_idx)
        targets_flat = adjusted_target.masked_select(adjusted_pad_mask)

        target_lengths = sample["target_lengths"] + 1 # with eos

        #logger.info("encoder out {}".format(encoder_out))
        #logger.info("lprobs {}".format(lprobs))

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        return ctc_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data paralell training"""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        qua_loss_sum = sum(log.get("qua_loss", 0) for log in logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "quantity loss", qua_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ce loss", ce_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctc loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

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
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )


@register_criterion("cifencoder_loss", dataclass=CifLossCriterionConfig)
class CifEncoderLossCriterion(FairseqCriterion):
    def __init__(self, cfg: CifLossCriterionConfig, task):
        super().__init__(task)

        self.sentence_avg = cfg.sentence_avg
        self.post_process = cfg.post_process
        self.ctc_weight = cfg.ctc_weight
        self.qua_weight = cfg.qua_weight

        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )

        tgt_dict = task.target_dictionary
        self.pad_idx = tgt_dict.pad() if tgt_dict is not None else -100

        self.eos_idx = task.target_dictionary.eos()
        self.zero_infinity = cfg.zero_infinity
        self.apply_ctc_loss = cfg.apply_ctc_loss
        self.apply_qua_loss = cfg.apply_quantity_loss

    def forward(self, model, sample, reduce=True):
        """ L = L_ce + L_qua + L_ctc"""

        # logger.info("sample {}".format(sample))
        net_output = model(target_lengths=sample["target_lengths"]+1, **sample["net_input"])
        encoder_outputs = net_output["encoder_outputs"] # for ctc loss
        cif_outputs = net_output["cif_outputs"] # for ce loss
        qua_out = cif_outputs["loss_pen"] # for quantity loss

        """move eos token from the last location to the end of the real sentence"""
        target = sample["target"]
        batch_size = target.size(0)
        pad_mask = ((target != self.pad_idx) & (target != self.eos_idx)).int()
        add_eos_idx = ((target * pad_mask) != 0).int().sum(-1).unsqueeze(dim=-1)  # B x 1
        add_one_hot_tensor = torch.zeros(
            batch_size, pad_mask.size(1)
        ).int().cuda().scatter_(1, add_eos_idx, 1) * self.eos_idx
        adjusted_target = torch.where(
            ((target.int() * pad_mask) + add_one_hot_tensor) == 0,
            torch.ones_like(target).int().cuda() * self.pad_idx,
            (target.int() * pad_mask) + add_one_hot_tensor,
        )

        qua_loss = torch.tensor(0.0)
        if self.apply_qua_loss:
            qua_loss = self.computeQualoss(qua_out, sample)

        ctc_loss = torch.tensor(0.0)
        if self.apply_ctc_loss:
            ctc_loss = self.computeCtcloss(model, encoder_outputs, sample, adjusted_target)

        ce_loss = self.computeCEloss(model, cif_outputs, sample, adjusted_target)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        loss = ce_loss + self.qua_weight * qua_loss + self.ctc_weight * ctc_loss

        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "qua_loss": utils.item(qua_loss.data),
            "ce_loss": utils.item(ce_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
        }

        # Evaluate on valid sets
        if not model.training:
            with torch.no_grad():
                ce_lprobs = model.get_normalized_probs(cif_outputs, log_probs=True)
                lprobs = ce_lprobs.float().contiguous().cpu()
                cif_lengths = cif_outputs["padding_mask"].int().sum(dim=-1) # B x T

                c_err = 0
                c_len = 0
                w_err = 0
                w_len = 0
                wv_err = 0
                for lp, t, inp_l in zip(
                    lprobs,
                    adjusted_target,
                    cif_lengths
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    p = (t != self.pad_idx) & (t != self.eos_idx)
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    if min(lp.shape) == 0:
                        toks = targ
                    else:
                        toks = lp.argmax(dim=-1)

                    pred_units_arr = toks[(toks != self.blank_idx) & (toks != self.pad_idx) & (toks != self.eos_idx)].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    # Calculate word error
                    dist = editdistance.eval(pred_words_raw, targ_words)
                    w_err += dist
                    wv_err += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_err
                logging_output["w_errors"] = w_err
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    def computeQualoss(self, qua_out, sample):
        target_lengths = sample["target_lengths"]
        target_with_eos_lengths = target_lengths + 1
        qua_loss = torch.abs(qua_out - target_with_eos_lengths).sum()

        return qua_loss


    def computeCEloss(self, model, cif_outputs, sample, adjusted_target, reduce=True):
        lprobs = model.get_normalized_probs(cif_outputs, log_probs=True)
        # logger.info("CE loss lprobs before view size {}".format(lprobs.shape))

        cif_max_len = cif_outputs["padding_mask"].size(1)

        assert cif_max_len == lprobs.size(1), "cif output padding mask does not match lprob"

        target_with_eos_lengths = sample["target_lengths"] + 1
        target_max_len = target_with_eos_lengths.max()
        min_len = min(cif_max_len, target_max_len)
        if target_max_len != cif_max_len:
            lprobs = lprobs[:, :min_len, :]
            adjusted_target = adjusted_target[:, :min_len]

        lprobs = lprobs.contiguous().view(-1, lprobs.size(-1))
        target_probs = adjusted_target.contiguous().view(-1)

        # logger.info("CE loss lprobs size {}".format(lprobs.shape))
        # logger.info("CE loss target size {}".format(target_probs.shape))

        ce_loss = F.nll_loss(
            lprobs,
            target_probs.long(),
            ignore_index=self.pad_idx,
            reduction="sum" if reduce else "none",
        )

        return ce_loss

    def computeCtcloss(self, model, encoder_outputs, sample, adjusted_target, reduce=True):
        encoder_out = encoder_outputs["encoder_out"]
        lprobs = model.get_encoder_normalized_probs(
            encoder_out, log_probs=True
        ).contiguous()

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if encoder_outputs["padding_mask"] is not None:
                non_padding_mask = ~encoder_outputs["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )


        adjusted_pad_mask = (adjusted_target != self.pad_idx)
        targets_flat = adjusted_target.masked_select(adjusted_pad_mask)

        target_lengths = sample["target_lengths"] + 1 # with eos

        #logger.info("encoder out {}".format(encoder_out))
        #logger.info("lprobs {}".format(lprobs))

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        return ctc_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data paralell training"""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        qua_loss_sum = sum(log.get("qua_loss", 0) for log in logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "quantity loss", qua_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ce loss", ce_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctc loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

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
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
