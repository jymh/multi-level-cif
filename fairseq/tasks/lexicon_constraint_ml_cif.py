# 2022
# Implemented by Rong Yiming for multi-level lexicon constraint cif

import logging
import os
import torch
import json

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any

from fairseq.data import AddMultiTargetsDataset, AddLexiconConstraintMlDataset, Dictionary, encoders
from fairseq.tasks.audio_pretraining import AudioPretrainingTask, AudioPretrainingConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from . import register_task
from .. import utils
from ..logging import metrics

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


def label_len_fn(label):
    return len(label.split(" "))


@dataclass
class MultiscaleCifConfig(AudioPretrainingConfig):
    # Options for reporting WER metrics during validation. Only applicable to
    # Seq2Seq models during fine-tuning
    phone_label: str = field(
        default = "ltr",
        metadata = {"help": "extension of phone level label file to load"}
    )
    char_label: str = field(
        default = "ltr.char",
        metadata = {"help": "extension of character level label file to load"}
    )

    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_detok: Optional[str] = field(
        default=None, metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); "
                    "required if using --eval-bleu; use 'space' to disable "
                    "detokenization; see fairseq.data.encoders for other options"
        }
    )
    eval_bleu_detok_args: str = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed"}
    )
    eval_tokenized_bleu: bool = field(
        default=False,
        metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None, metadata={"help": "remove BPE before computing BLEU"}
    )
    eval_bleu_args: str = field(
        default="{}",
        metadata={"help": "generation args for BLUE scoring, e.g., "
                          "'{\"beam\": 4, \"lenpen\": 0.6}'"}
    )
    eval_bleu_print_samples: bool = field(
        default=False,
        metadata={"help": "print sample generations during validation"}
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq models); "
            "adds 'prev_output_tokens' to input and appends eos to target"
        },
    )


@register_task("lexicon_constraint_mlcif", dataclass=MultiscaleCifConfig)
class MultiLevelLexiconConstraintCifTask(AudioPretrainingTask):
    """ """

    cfg: MultiscaleCifConfig

    def __init__(
        self,
        cfg: MultiscaleCifConfig,
    ):
        super().__init__(cfg)
        self.blank_symbol = "<s>"

        self.state.add_factory("target_phone_dictionary", self.load_phone_target_dictionary)
        self.state.add_factory("target_char_dictionary", self.load_char_target_dictionary)

    def load_phone_target_dictionary(self):
        if self.cfg.phone_label:
            dict_path = os.path.join(self.cfg.data, f"dict.{self.cfg.phone_label}.txt")
            return Dictionary.load(dict_path)
        return None

    def load_char_target_dictionary(self):
        if self.cfg.char_label:
            dict_path = os.path.join(self.cfg.data, f"dict.{self.cfg.char_label}.txt")
            return Dictionary.load(dict_path)
        return None

    def load_dataset(self, split: str, task_cfg: MultiscaleCifConfig = None, **kwargs):
        super().load_dataset(split, task_cfg, **kwargs)

        task_cfg = task_cfg or self.cfg
        assert task_cfg.phone_label and task_cfg.char_label is not None
        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )
        data_path = self.cfg.data

        phone_label_path = os.path.join(data_path, f"{split}.{task_cfg.phone_label}")
        char_label_path = os.path.join(data_path, f"{split}.{task_cfg.char_label}")

        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        text_compressor = TextCompressor(level=text_compression_level)
        with open(phone_label_path, "r") as f:
            phone_labels = [
                text_compressor.compress(l)
                for i, l in enumerate(f) if i not in skipped_indices
            ]

        assert len(phone_labels) == len(self.datasets[split]), (
            f"phone labels length ({len(phone_labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        with open(char_label_path, "r") as f:
            char_labels = [
                text_compressor.compress(l)
                for i, l in enumerate(f) if i not in skipped_indices
            ]

        assert len(char_labels) == len(self.datasets[split]), (
            f"character labels length ({len(char_labels)}) and dataset length "
            f"({len(self.datasets[split])}) do no match"
        )

        phone_process_label = LabelEncoder(self.target_phone_dictionary)
        char_process_label = LabelEncoder(self.target_char_dictionary)

        self.datasets[split] = AddMultiTargetsDataset(
            self.datasets[split],
            phone_labels,
            char_labels,
            pad=self.target_char_dictionary.pad(),
            eos=self.target_char_dictionary.eos(),
            batch_targets=True,
            phone_process_label=phone_process_label,
            char_process_label=char_process_label,
            label_len_fn=label_len_fn,
            add_to_input=task_cfg.get("autoregressive", False),
            text_compression_level=text_compression_level
        )

        if split == 'train':
            with open("/data1/ymrong/share_dir/data/hkust/train/lexicon_constraint.txt") as f:
                phone_segs = []
                for line in f.readlines():
                    line = line.strip()
                    one_phone_seg = [int(i) for i in line.split()]
                    phone_segs.append(one_phone_seg)
            with open("/data1/ymrong/share_dir/data/hkust/train/word_segs.txt") as f:
                integrate_weights = []
                for line in f.readlines():
                    line = line.strip()
                    one_integrate_weight = [int(i) for i in line.split()]
                    integrate_weights.append(one_integrate_weight)
            self.datasets[split] = AddLexiconConstraintMlDataset(
                self.datasets[split],
                phone_segs=phone_segs,
                integrate_weights=integrate_weights,
            )
        elif split == 'valid':
            with open("/data1/ymrong/share_dir/data/hkust/train/valid.lexicon_constraint.txt") as f:
                phone_segs = []
                for line in f.readlines():
                    line = line.strip()
                    one_phone_seg = [int(i) for i in line.split()]
                    phone_segs.append(one_phone_seg)
            with open("/data1/ymrong/share_dir/data/hkust/train/valid.word_segs.txt") as f:
                integrate_weights = []
                for line in f.readlines():
                    line = line.strip()
                    one_integrate_weight = [int(i) for i in line.split()]
                    integrate_weights.append(one_integrate_weight)
            self.datasets[split] = AddLexiconConstraintMlDataset(
                self.datasets[split],
                phone_segs=phone_segs,
                integrate_weights=integrate_weights,
            )


    @property
    def target_phone_dictionary(self):
        return self.state.target_phone_dictionary

    @property
    def target_char_dictionary(self):
        return self.state.target_char_dictionary

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_wer and self.cfg.autoregressive:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model)
            logging_output["_num_char_errors"] = metrics["num_char_errors"]
            logging_output["_num_chars"] = metrics["num_chars"]
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        if self.cfg.eval_bleu and self.cfg.autoregressive:
            metrics = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = metrics.sys_len
            logging_output['_bleu_ref_len'] = metrics.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(metrics.counts) == 4
            for i in range(4):
                logging_output[f"_bleu_counts_{i}"] = metrics.counts[i]
                logging_output[f"_bleu_totals_{i}"] = metrics.totals[i]
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)

        if self.cfg.eval_wer and self.cfg.autoregressive:
            self.sequence_generator = self.build_generator(
                [model],
                self.cfg.eval_wer_config,
            )
            if self.cfg.eval_wer_tokenizer:
                self.tokenizer = encoders.build_tokenizer(self.cfg.eval_wer_tokenizer)
            else:
                self.tokenizer = None
        if self.cfg.eval_bleu and self.cfg.autoregressive:
            assert self.cfg.eval_bleu_detok is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )
            gen_args = json.loads(self.cfg.eval_bleu_args)
            gen_args = Namespace(**gen_args)
            self.sequence_generator = self.build_generator([model], gen_args)

        return model

    def _inference_with_wer(self, generator, sample, model):
        import editdistance

        def decode(toks):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_wer_post_process,
                escape_unk=True,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        num_word_errors, num_char_errors = 0, 0
        num_chars, num_words = 0, 0
        gen_out = self.inference_step(generator, [model], sample, None)
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
            )
            num_char_errors += editdistance.eval(hyp, ref)
            num_chars += len(ref)
            hyp_words = hyp.split()
            ref_words = ref.split()
            num_word_errors += editdistance.eval(hyp_words, ref_words)
            num_words += len(ref_words)

        return {
            "num_char_errors": num_char_errors,
            "num_chars": num_chars,
            "num_word_errors": num_word_errors,
            "num_words": num_words,
        }

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, is_ref):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if is_ref else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens'], is_ref=False))
            refs.append(
                decode(
                    utils.strip_pad(
                        sample['target'][i],
                        self.target_dictionary.pad()
                    ),
                    is_ref=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info('H-{} {}'.format(sample["id"][0], hyps[0]))
            logger.info('T-{} {}'.format(sample["id"][0], refs[0]))

        eval_tokenization = 'none' if self.cfg.eval_tokenized_bleu else '13a'
        return sacrebleu.corpus_bleu(hyps, [refs], tokenize=eval_tokenization)

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if self.cfg.eval_wer:
            zero = torch.scalar_tensor(0.0)
            num_char_errors = sum(
                log.get("_num_char_errors", zero) for log in logging_outputs
            )
            num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
            num_word_errors = sum(
                log.get("_num_word_errors", zero) for log in logging_outputs
            )
            num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
            metrics.log_scalar("_num_char_errors", num_char_errors)
            metrics.log_scalar("_num_chars", num_chars)
            metrics.log_scalar("_num_word_errors", num_word_errors)
            metrics.log_scalar("_num_words", num_words)
            if num_chars > 0:
                metrics.log_derived(
                    "uer",
                    lambda meters: meters["_num_char_errors"].sum
                    * 100.0
                    / meters["_num_chars"].sum
                    if meters["_num_chars"].sum > 0
                    else float("nan"),
                )
            if num_words > 0:
                metrics.log_derived(
                    "wer",
                    lambda meters: meters["_num_word_errors"].sum
                    * 100.0
                    / meters["_num_words"].sum
                    if meters["_num_words"].sum > 0
                    else float("nan"),
                )
        if self.cfg.eval_bleu:
            len_keys = ["_bleu_sys_len", "_bleu_ref_len"]
            count_keys = [f"_bleu_counts_{i}" for i in range(4)]
            total_keys = [f"_bleu_totals_{i}" for i in range(4)]
            for k in len_keys + count_keys + total_keys:
                metrics.log_scalar(
                    k, sum(log.get(k, 0) for log in logging_outputs)
                )

            import sacrebleu
            metrics.log_derived(
                'bleu',
                lambda meters: sacrebleu.compute_bleu(
                    correct=[meters[k].sum for k in count_keys],
                    total=[meters[k].sum for k in total_keys],
                    sys_len=meters['_bleu_sys_len'].sum,
                    ref_len=meters['_bleu_ref_len'].sum,
                    smooth_method="exp"
                ).score
            )
