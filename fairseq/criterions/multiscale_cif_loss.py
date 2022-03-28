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
from fairseq.tasks import FairseqTask
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import safe_round

logger = logging.getLogger(__name__)

@dataclass
class MultiScaleCifLossCriterionConfig(FairseqDataclass):
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


@register_criterion("ms_cifloss", dataclass=MultiScaleCifLossCriterionConfig)
class MultiScaleCifLossCriterion(FairseqCriterion):
    def __init__(self, cfg: MultiScaleCifLossCriterionConfig, task: FairseqTask):
        super().__init__(task)

        self.sentence_avg = cfg.sentence_avg
        self.post_process = cfg.post_process
        self.ctc_weight = cfg.ctc_weight
        self.qua_weight = cfg.qua_weight

        self.blank_idx = (
            task.target_char_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )

        p_tgt_dict = task.target_phone_dictionary
        c_tgt_dict = task.target_char_dictionary
        self.pad_idx = p_tgt_dict.pad() if p_tgt_dict is not None else -100

        self.eos_idx = task.target_phone_dictionary.eos()
        self.zero_infinity = cfg.zero_infinity
        self.apply_ctc_loss = cfg.apply_ctc_loss
        self.apply_qua_loss = cfg.apply_quantity_loss

    def move_eos(self, target):
        """move eos token from the last location to the end of the real sentence"""
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
        return adjusted_target

    def forward(self, model, sample, reduce=True):
        """ L = L_ce + L_qua + L_ctc"""

        # logger.info("sample {}".format(sample))

        p_sample = sample["p_sample"]
        c_sample = sample["c_sample"]

        p_target = p_sample["target"]
        c_target = c_sample["target"]

        net_output = model(
            phone_target_lengths=p_sample["target_lengths"]+1,
            char_target_lengths=c_sample["target_lengths"]+1,
            **sample["net_input"]
        )

        '''
        net_output: {
            "joint_outputs": {
                "joint_out": joint_out,  # B x Tmin x C
                "padding_mask": c_cif_outputs["padding_mask"],
                "qua_out": c_cif_outputs["loss_pen"],
            },

            "phone_cif_outputs": {
                "phone_cif_out": p_cif_out,
                "padding_mask": p_cif_outputs["padding_mask"],
            },

            # for ctc loss
            "phone_encoder_outputs": {
                "encoder_out": p_encoder_outputs["encoder_out"],
                "padding_mask": p_encoder_outputs["padding_mask"],
            },
        }
        '''

        phone_encoder_outputs = net_output["phone_encoder_outputs"] # for ctc loss
        joint_outputs = net_output["joint_outputs"] # for ce loss
        phone_cif_outputs = net_output["phone_cif_outputs"]

        adjusted_p_target = self.move_eos(p_target)
        adjusted_c_target = self.move_eos(c_target)

        p_qua_loss = torch.tensor(0.0).type_as(p_target)
        c_qua_loss = torch.tensor(0.0).type_as(c_target)
        if self.apply_qua_loss:
            p_qua_loss = self.computeQualoss(phone_cif_outputs["qua_out"], p_sample)
            c_qua_loss = self.computeQualoss(joint_outputs["qua_out"], c_sample)

        ctc_loss = torch.tensor(0.0).type_as(p_target)
        if self.apply_ctc_loss:
            ctc_loss = self.computeCtcloss(model, phone_encoder_outputs, sample, adjusted_p_target)

        p_ce_loss = self.computeCEloss(model, phone_cif_outputs, p_sample, adjusted_p_target)
        c_ce_loss = self.computeCEloss(model, joint_outputs, c_sample, adjusted_c_target)

        p_sample_size = (
            p_sample["target"].size(0) if self.sentence_avg else p_sample["ntokens"]
        )
        c_sample_size = (
            c_sample["target"].size(0) if self.sentence_avg else c_sample["ntokens"]
        )

        phone_loss = p_ce_loss + self.qua_weight * p_qua_loss + self.ctc_weight * ctc_loss
        char_loss = c_ce_loss + self.qua_weight * c_qua_loss
        loss = phone_loss + char_loss

        p_sample_size = p_sample["target"].size(0) if self.sentence_avg else p_sample["ntokens"]
        c_sample_size = c_sample["target"].size(0) if self.sentence_avg else c_sample["ntokens"]

        sample_size = p_sample_size + c_sample_size
        n_tokens = p_sample["ntokens"] + c_sample["ntokens"]
        n_sentences = p_sample["target"].size(0) + c_sample["target"].size(0)

        logging_output = {
            "loss": utils.item(loss.data),
            "phone_loss": utils.item(phone_loss.data),
            "char_loss": utils.item(char_loss.data),
            "ntokens": n_tokens,
            "p_ntokens": p_sample["ntokens"],
            "c_ntokens": c_sample["ntokens"],
            "nsentences": n_sentences,
            "p_nsentences": p_sample["target"].size(0),
            "c_nsentences": c_sample["target"].size(0),
            "sample_size": sample_size,
            "p_sample_size": p_sample_size,
            "c_sample_size": c_sample_size,
            "p_qua_loss": utils.item(p_qua_loss.data),
            "c_qua_loss": utils.item(c_qua_loss.data),
            "p_ce_loss": utils.item(p_ce_loss.data),
            "c_ce_loss": utils.item(c_ce_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
        }

        # Evaluate on valid sets
        if not model.training:
            with torch.no_grad():
                char_lprobs = model.get_probs_from_logits(joint_outputs["joint_out"], log_probs=True)
                char_lprobs = char_lprobs.float().contiguous().cpu()
                c_cif_lengths = joint_outputs["padding_mask"].int().sum(dim=-1) # B x T

                phone_lprobs = model.get_probs_from_logits(phone_cif_outputs["phone_cif_out"], log_probs=True)
                phone_lprobs = phone_lprobs.float().contiguous().cpu()
                p_cif_lengths = phone_cif_outputs["padding_mask"].int().sum(dim=-1)

                fired_marks = joint_outputs["fired_marks"]

                c_err = 0
                c_len = 0
                w_err = 0
                wv_err = 0
                w_len = 0
                p_u_err = 0
                p_u_len = 0
                p_err = 0
                p_len = 0

                for c_lp, t, inp_l in zip(
                    char_lprobs,
                    adjusted_c_target,
                    c_cif_lengths
                ):
                    c_lp = c_lp[:inp_l].unsqueeze(0)

                    p = (t != self.pad_idx) & (t != self.eos_idx)
                    targ = t[p]
                    targ_units = self.task.target_char_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    if min(c_lp.shape) == 0:
                        toks = targ
                    else:
                        toks = c_lp.argmax(dim=-1)

                    pred_units_arr = toks[(toks != self.blank_idx) & (toks != self.pad_idx) & (toks != self.eos_idx)].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_char_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    # Calculate word error
                    dist = editdistance.eval(pred_words_raw, targ_words)
                    w_err += dist
                    wv_err += dist

                    w_len += len(targ_words)

                i = 0
                for p_lp, t, inp_l in zip(
                    phone_lprobs,
                    adjusted_p_target,
                    p_cif_lengths
                ):
                    p_lp = p_lp[:inp_l].unsqueeze(0)

                    p = (t != self.pad_idx) & (t != self.eos_idx)
                    targ = t[p]
                    targ_units = self.task.target_phone_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    if min(p_lp.shape) == 0:
                        toks = targ
                    else:
                        toks = p_lp.argmax(dim=-1)

                    pred_units_arr = toks[(toks != self.blank_idx) & (toks != self.pad_idx) & (toks != self.eos_idx)].tolist()

                    p_u_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    p_u_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_phone_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    sample_fired_marks = fired_marks[i, :]
                    i += 1

                    # Calculate word error
                    dist = editdistance.eval(pred_words_raw, targ_words)
                    p_err += dist

                    p_len += len(targ_words)

                logging_output["wv_errors"] = wv_err
                logging_output["w_errors"] = w_err
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len
                logging_output["pu_errors"] = p_u_err
                logging_output["p_errors"] = p_err
                logging_output["pu_total"] = p_u_len
                logging_output["p_total"] = p_len

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

        target_lengths = sample["p_sample"]["target_lengths"] + 1 # with eos

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
        phone_loss_sum = sum(log.get("phone_loss", 0) for log in logging_outputs)
        char_loss_sum = sum(log.get("char_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        p_ntokens = sum(log.get("p_ntokens", 0) for log in logging_outputs)
        c_ntokens = sum(log.get("c_ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        p_sample_size = sum(log.get("p_sample_size", 0) for log in logging_outputs)
        c_sample_size = sum(log.get("c_sample_size", 0) for log in logging_outputs)
        p_qua_loss_sum = sum(log.get("p_qua_loss", 0) for log in logging_outputs)
        c_qua_loss_sum = sum(log.get("c_qua_loss", 0) for log in logging_outputs)
        p_ce_loss_sum = sum(log.get("p_ce_loss", 0) for log in logging_outputs)
        c_ce_loss_sum = sum(log.get("c_ce_loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "phone_loss", phone_loss_sum / p_sample_size / math.log(2), p_sample_size, round=3
        )
        metrics.log_scalar(
            "char_loss", char_loss_sum / c_sample_size / math.log(2), c_sample_size, round=3
        )
        metrics.log_scalar(
            "phone_quantity_loss", p_qua_loss_sum / p_sample_size / math.log(2), p_sample_size, round=3
        )
        metrics.log_scalar(
            "char_quantity_loss", c_qua_loss_sum / c_sample_size / math.log(2), c_sample_size, round=3
        )
        metrics.log_scalar(
            "phone_ce_loss", p_ce_loss_sum / p_sample_size / math.log(2), p_sample_size, round=3
        )
        metrics.log_scalar(
            "char_ce_loss", c_ce_loss_sum / c_sample_size / math.log(2), c_sample_size, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / p_sample_size / math.log(2), p_sample_size, round=3
        )
        metrics.log_scalar("p_ntokens", p_ntokens)
        metrics.log_scalar("c_ntokens", c_ntokens)
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

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
        pu_errors = sum(log.get("pu_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_pu_errors", pu_errors)
        pu_total = sum(log.get("pu_total", 0) for log in logging_outputs)
        metrics.log_scalar("_pu_total", pu_total)
        p_errors = sum(log.get("p_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_p_errors", p_errors)
        p_total = sum(log.get("p_total", 0) for log in logging_outputs)
        metrics.log_scalar("_p_total", p_total)

        if c_total > 0:
            metrics.log_derived(
                "cer",
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
        if pu_total > 0:
            metrics.log_derived(
                "phone_er",
                lambda meters: safe_round(
                    meters["_pu_errors"].sum * 100.0 / meters["_pu_total"].sum, 3
                )
                if meters["_pu_total"].sum > 0
                else float("nan"),
            )
        if p_total > 0:
            metrics.log_derived(
                "per",
                lambda meters: safe_round(
                    meters["_p_errors"].sum * 100.0 /meters["_p_total"].sum, 3
                )
                if meters["_p_total"].sum > 0
                else float("nan"),
            )