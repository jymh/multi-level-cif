# @Time    : 2021/7/14
# @Author  : Minglun Han
#            Modified by Yiming Rong
# @File    : cif_decoder.py

import os
import sys
import torch
import logging
import numpy as np
import itertools as it

# Control print options
# torch.set_printoptions(profile="full")
# torch.set_printoptions(profile="default")
# np.set_printoptions(threshold=sys.maxsize)

class CifVanillaDecoder(object):
    def __init__(self, args, tgt_dict, use_phone=False):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest
        self.beam = args.beam
        self.use_phone = use_phone

        # Get the index of special tokens
        self.blank = tgt_dict.index("<ctc_blank>") \
            if "<ctc_blank>" in tgt_dict.indices else tgt_dict.bos()
        self.bos = tgt_dict.bos()
        self.eos = tgt_dict.eos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()

        self.cif_decoder_mode = args.cif_decoder_mode

        if self.beam == 1:
            if self.cif_decoder_mode == "ar":
                logging.info("employ ar greedy decoder")
                self.decode = self.ar_batch_greedy_decode
            else:
                logging.info("employ nar greedy decoder")
                # self.decode = self.nar_batch_greedy_decode
                self.decode = self.nar_batch_parallel_greedy_decode # Parallel Decoding which is better for NAR decoder
        else:
            if self.cif_decoder_mode == "ar":
                logging.info("employ ar beam decoder")
                self.decode = self.ar_batch_beam_decode
            else:
                logging.info("employ nar beam decoder")
                self.decode = self.nar_batch_beam_decode

    def generate(self, models, sample, use_phone=False, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder

        # Prepare model inputs
        model_inputs = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }   # remove prev_output_tokens

        # Begin decoding
        if not self.use_phone:
            cif_outputs = models[0].get_cif_output(target_lengths_with_eos=None, **model_inputs)
            beam_results, beam_scores, out_seqlens = self.decode(models[0], cif_outputs)
        else:
            cif_outputs = models[0].get_phone_output(target_lengths_with_eos=None, **model_inputs)
            beam_results, beam_scores, out_seqlens = self.decode(models[0], cif_outputs, use_phone=True)

        return self.generate_hypos(
            beam_results=beam_results,
            beam_scores=beam_scores,
            out_seqlens=out_seqlens,
        )

    def generate_hypos(self, beam_results, beam_scores, out_seqlens):
        hypos = []
        for beam_result, scores, lengths in zip(beam_results, beam_scores, out_seqlens):
            # beam_ids: beam x id; score: beam; length: beam
            top = []
            for result, score, length in zip(beam_result, scores, lengths):
                top.append({
                    "tokens": self.get_tokens(result[:length]),
                    "score": score
                })
            hypos.append(top)
        return hypos

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        # Remove blank id and eos id
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        idxs = filter(lambda x: x != self.eos, idxs)
        idxs = filter(lambda x: x != self.pad, idxs)

        return torch.LongTensor(list(idxs))

    def ar_batch_greedy_decode(self, model, cif_outputs, use_phone=False):
        """
        :param model: the model in usage
        :param cif_outputs: the outputs of cif module
        :return: prev_tokens, out_seqlens, scores
        """
        # Get Cif outputs
        cif_out = cif_outputs["cif_out"]
        cif_out_padding_mask = cif_outputs["padding_mask"]

        # Get the maximum length of decoding steps
        batch_size, max_decode_length, _ = cif_out.size()
        out_seqlens = cif_out_padding_mask.sum(-1)
        # out_seqlens has shape [B]

        # Initialize previous decoded tokens
        prev_tokens = torch.ones([batch_size, 1]).long().cuda() * self.eos
        # B x 1, use <eos> as the beginning of sentence (<bos>)
        scores = torch.ones([batch_size]).cuda()  # B
        for step_i in range(max_decode_length):
            # Conduct forward of current step t
            cur_step_cif_outputs = cif_out[:, :(step_i + 1), :]   # B x t x C
            cur_step_cif_out_padding_mask = cif_out_padding_mask[:, :(step_i + 1)]  # B x t
            cur_step_cif_out = {
                "cif_out": cur_step_cif_outputs,
                "padding_mask": cur_step_cif_out_padding_mask,
            }

            # Get decoder outputs of current step
            if not use_phone:
                decoder_output_i, extra_outputs = model.step_forward_decoder(
                    prev_decoded_tokens=prev_tokens,
                    cif_outputs=cur_step_cif_out
                )
            else:
                decoder_output_i, extra_outputs = model.step_forward_phone_decoder(
                    prev_decoded_tokens=prev_tokens,
                    cif_outputs=cur_step_cif_out
                )

            # Update previous decoded tokens
            decoder_output_i = model.get_probs_from_logits(decoder_output_i[:, -1, :], log_probs=False)
            latest_token = torch.argmax(decoder_output_i, dim=-1).unsqueeze(dim=-1)    # shape = B x 1
            max_value_of_latest_token = decoder_output_i.max(-1)[0]   # shape = B
            prev_tokens = torch.cat([prev_tokens, latest_token], dim=-1)
            scores = scores * max_value_of_latest_token

        # Reform outputs
        prev_tokens = torch.unsqueeze(prev_tokens, dim=1)[:, :, 1:]     # B x 1 x T
        out_seqlens = torch.unsqueeze(out_seqlens, dim=1)               # B x 1
        scores = torch.unsqueeze(scores, dim=-1)                        # B x 1

        return prev_tokens, scores, out_seqlens

    def nar_batch_greedy_decode(self, model, cif_outputs):
        """
        :param model: the model in usage
        :param cif_outputs: the outputs of cif module
        :return: prev_tokens, out_seqlens, scores
        """

        # Get cif outputs
        cif_out = cif_outputs["cif_out"]
        ctxt_cif_outputs = cif_outputs["ctxt_cif_out"] \
            if "ctxt_cif_out" in cif_outputs.keys() and cif_outputs["ctxt_cif_out"] is not None else None
        cif_out_padding_mask = cif_outputs["padding_mask"]

        # Get the maximum length of decoding steps
        batch_size, max_decode_length, _ = cif_out.size()
        out_seqlens = cif_out_padding_mask.sum(-1)  # B

        # Initialize previous decoded tokens
        prev_tokens = torch.zeros([batch_size, 1]).long().cuda() * self.eos     # B x 1
        scores = torch.ones([batch_size]).cuda()                                # B

        # Begin Loop
        for step_i in range(max_decode_length): # 0, 1, 2, ......, (T - 1)
            # Conduct forward of current step t
            cur_step_cif_outputs = cif_out[:, :(step_i + 1), :]   # B x t x C
            cur_step_ctxt_cif_outputs = ctxt_cif_outputs[:, :(step_i + 1), :] if ctxt_cif_outputs is not None else None
            cur_step_cif_out_padding_mask = cif_out_padding_mask[:, :(step_i + 1)]  # B x t
            cur_step_cif_out = {
                "cif_out": cur_step_cif_outputs,
                "ctxt_cif_out": cur_step_ctxt_cif_outputs,
                "padding_mask": cur_step_cif_out_padding_mask}

            # Get decoder outputs of current step
            decoder_output_i, _ = model.step_forward_decoder(
                prev_decoded_tokens=prev_tokens, cif_outputs=cur_step_cif_out)

            # Update previous decoded tokens
            decoder_output_i = model.get_probs_from_logits(decoder_output_i[:, -1, :], log_probs=False)
            latest_token = torch.argmax(decoder_output_i, dim=-1).unsqueeze(dim=-1)     # shape = B x 1
            max_value_of_latest_token = decoder_output_i.max(-1)[0]                     # shape = B
            prev_tokens = torch.cat([prev_tokens, latest_token], dim=-1)                # shape = B x t
            scores = scores * max_value_of_latest_token

        # Reform outputs, now prev_tokens has shape B x (T + 1)
        prev_tokens = torch.unsqueeze(prev_tokens, dim=1)[:, :, 1:]     # B x 1 x (T + 1) --> B x 1 x T
        out_seqlens = torch.unsqueeze(out_seqlens, dim=1)               # B x 1
        scores = torch.unsqueeze(scores, dim=-1)                        # B x 1

        return prev_tokens, scores, out_seqlens

    def nar_batch_parallel_greedy_decode(self, model, cif_outputs):
        """
        :param model: the model in usage
        :param cif_outputs: the outputs of cif module
        :return: prev_tokens, out_seqlens, scores
        """

        # Get cif outputs
        cif_out = cif_outputs["cif_out"]
        ctxt_cif_outputs = cif_outputs["ctxt_cif_out"] \
            if "ctxt_cif_out" in cif_outputs.keys() and cif_outputs["ctxt_cif_out"] is not None else None
        cif_out_padding_mask = cif_outputs["padding_mask"]

        # Get the maximum length of decoding steps
        batch_size, max_decode_length, _ = cif_out.size()
        out_seqlens = cif_out_padding_mask.sum(-1)  # B

        # Initialize previous decoded tokens
        prev_decoded_tokens = torch.zeros([batch_size, max_decode_length])    # B x T

        decoder_output, _ = model.step_forward_decoder(
            prev_decoded_tokens=prev_decoded_tokens, cif_outputs=cif_outputs)  # B x T x V

        # Update previous decoded tokens
        decoder_output = model.get_probs_from_logits(decoder_output, log_probs=False)   # B x T x V
        decoded_tokens = torch.argmax(decoder_output, dim=-1)                           # B x T
        scores = torch.prod(decoder_output.max(-1)[0], dim=-1)                          # B

        # Reform outputs, now prev_tokens has shape B x (T + 1)
        prev_tokens = torch.unsqueeze(decoded_tokens, dim=1)    # B x 1 x T
        out_seqlens = torch.unsqueeze(out_seqlens, dim=1)       # B x 1
        scores = torch.unsqueeze(scores, dim=-1)                # B x 1

        return prev_tokens, scores, out_seqlens

    def ar_batch_beam_decode(self, model, cif_outputs, use_phone=False):
        """
         :param model: the model in usage
         :param cif_outputs: the outputs of cif module
         :return: prev_tokens, out_seqlens, scores
        """
        cif_out = cif_outputs["cif_out"]                                # B x T x C
        cif_out_padding_mask = cif_outputs["padding_mask"]      # B x T

        # Get the maximum length of decoding steps
        batch_size, max_decode_length, cif_out_dim = cif_out.size()     # B x T x C
        out_seqlens = cif_out_padding_mask.sum(-1)                      # B

        # Initialize all needed variables
        cif_out = torch.unsqueeze(cif_out, dim=1).repeat(1, self.beam, 1, 1)            # B x beam_size x T x C
        prev_tokens = torch.ones([batch_size, self.beam, 1]).long().cuda() * self.eos   # B x beam_size x 1
        scores = torch.zeros([batch_size, self.beam]).float().cuda()                    # B x beam_size
        cif_out_padding_mask = torch.unsqueeze(cif_out_padding_mask, dim=1).repeat([1, self.beam, 1])
        # B x beam_size x T

        cif_out = cif_out.reshape([batch_size * self.beam, max_decode_length, cif_out_dim]) # (B * beam_size) x T x C
        prev_tokens = prev_tokens.reshape([batch_size * self.beam, 1])  # (B * beam_size) x 1
        scores = scores.reshape([batch_size * self.beam])   # (B * beam_size)
        cif_out_padding_mask = cif_out_padding_mask.reshape(
            [batch_size * self.beam, max_decode_length])  # (B * beam_size) x T

        for step_i in range(1, max_decode_length + 1):
            # Get cif outputs of current step
            cur_step_cif_outputs = cif_out[:, :step_i, :]                       # (B * beam_size) x t x C
            cur_step_cif_out_padding_mask = cif_out_padding_mask[:, :step_i]      # (B * beam_size) x t
            cur_step_cif_out = {
                "cif_out": cur_step_cif_outputs,
                "padding_mask": cur_step_cif_out_padding_mask
            }

            # Get decoder outputs at step_i
            if not use_phone:
                decoder_output_i = model.step_forward_decoder(
                    prev_decoded_tokens=prev_tokens,  # (B x beam_size) x t
                    cif_outputs=cur_step_cif_out,
                    # cif_out: (B * beam_size) x t x C, cif_out_padding_mask: (B * beam_size) x t
                )   # decoder_output_i has shape [(B * beam_size), t, V]
            else:
                decoder_output_i = model.step_forward_phone_decoder(
                    cif_outputs=cur_step_cif_out
                )
            cur_decoder_output = model.get_probs_from_logits(
                decoder_output_i[:, -1, :], log_probs=True) # [B * beam_size, V]
            tmp_scores = scores # Backup scores, with shape [B * beam_size]
            scores = scores.unsqueeze(dim=-1).repeat([1, self.vocab_size])  # [B * beam_size, V]

            cur_score = cur_decoder_output
            # cur_score, with shape [(B x beam_size) x V]

            updated_scores = (scores + cur_score).reshape(
                [batch_size, self.beam * self.vocab_size]
            )   # converted from shape [B * beam_size, V] to [B, beam_size * V]

            # Handle the first timestep with special operation
            if step_i == 1:
                # For the first step, due to the same input token, only consider one beam.
                topk_scores, topk_indices = torch.topk(
                    updated_scores.reshape([batch_size, self.beam, self.vocab_size])[:, 0, :], k=self.beam, dim=-1)
                beam_indices = torch.zeros(batch_size, self.beam).long().cuda() # [B, beam_size] with all zero elements
                fixed_topk_indices = topk_indices   # [B, beam_size]
            else:
                # For all the other beams, due to their inputs are varying, consider all beams.
                topk_scores, topk_indices = torch.topk(
                    updated_scores, k=self.beam, dim=-1
                ) # topk_scores shape [B, beam_size], topk_indices shape [B, beam_size]
                beam_indices = topk_indices // self.vocab_size      # [B, beam_size]
                fixed_topk_indices = topk_indices % self.vocab_size # [B, beam_size]

            # Update previous decoded tokens and scores
            prev_tokens = prev_tokens.reshape([batch_size, self.beam, -1])  # [B, beam_size, t]
            tmp_scores = tmp_scores.reshape([batch_size, self.beam])        # previous scores, with shape [B, beam_size]
            prev_token_tmp_list = []
            scores_tmp_list = []
            for n in range(batch_size): # n ranges from 0 to (batch_size - 1)
                # Get the max length of current sample
                cur_output_maxlen = out_seqlens[n]

                # If some sample's decode length is smaller than current step id, keep its score and decoded results
                if step_i > cur_output_maxlen:
                    cur_scores = tmp_scores[n, :]           # beam_size
                    cur_prev_tokens = prev_tokens[n, :, :]  # beam_size x t
                else:
                    cur_scores = topk_scores[n, :]          # beam_size
                    cur_prev_tokens = prev_tokens[n, :, :]  # beam_size x t
                    cur_beam_indices = beam_indices[n, :]   # beam_size

                    # Get reformed previous tokens
                    cur_prev_tokens = \
                        torch.index_select(cur_prev_tokens, dim=0, index=cur_beam_indices) # beam_size x t

                scores_tmp_list.append(cur_scores.unsqueeze(dim=0))
                prev_token_tmp_list.append(cur_prev_tokens.unsqueeze(dim=0))

            fixed_prev_tokens = torch.cat(prev_token_tmp_list, dim=0)
            fixed_topk_indices = torch.where(
                step_i <= out_seqlens.unsqueeze(dim=-1).repeat([1, self.beam]),
                fixed_topk_indices,                                     # B x beam_size
                torch.ones_like(fixed_topk_indices).cuda() * self.pad,
            )   # Mask locations that outnumber cif max length using <pad>
            fixed_topk_indices = fixed_topk_indices.unsqueeze(dim=-1)   # [B, beam_size, 1]
            prev_tokens = torch.cat(
                [fixed_prev_tokens, fixed_topk_indices], dim=-1
            ).reshape([batch_size * self.beam, -1])     # [B * beam_size, t + 1]
            scores = torch.cat(scores_tmp_list, dim=0).reshape([batch_size * self.beam])    # [B * beam_size]

        scores = scores.reshape([batch_size, self.beam])[:, :self.nbest]    # B x beam_size
        prev_tokens = prev_tokens.reshape([batch_size, self.beam, -1])[:, :self.nbest, 1:]  # B x beam_size x T
        out_seqlens = torch.unsqueeze(out_seqlens, dim=-1).repeat(1, self.beam)[:, :self.nbest] # B x beam_size

        return prev_tokens, scores, out_seqlens

    def nar_batch_beam_decode(self, model, cif_outputs):
        """
         :param model: the model in usage
         :param cif_outputs: the outputs of cif module
         :return: prev_tokens, out_seqlens, scores
        """

        cif_out = cif_outputs["cif_out"]                                # B x T x C
        ctxt_cif_outputs = cif_outputs["ctxt_cif_out"] \
            if "ctxt_cif_out" in cif_outputs.keys() and cif_outputs["ctxt_cif_out"] is not None else None
        cif_out_padding_mask = cif_outputs["padding_mask"]      # B x T

        # Get the maximum length of decoding steps
        batch_size, max_decode_length, cif_out_dim = cif_out.size()     # B x T x C
        out_seqlens = cif_out_padding_mask.sum(-1)                      # B

        # Initialize all needed variables
        cif_out = torch.unsqueeze(cif_out, dim=1).repeat(1, self.beam, 1, 1)            # B x beam_size x T x C
        ctxt_cif_outputs = torch.unsqueeze(ctxt_cif_outputs, dim=1).repeat(1, self.beam, 1, 1) \
            if ctxt_cif_outputs is not None else None
        prev_tokens = torch.ones([batch_size, self.beam, 1]).long().cuda() * self.eos   # B x beam_size x 1
        scores = torch.zeros([batch_size, self.beam]).float().cuda()                    # B x beam_size
        cif_out_padding_mask = \
            torch.unsqueeze(cif_out_padding_mask, dim=1).repeat([1, self.beam, 1])      # B x beam_size x T

        cif_out = cif_out.reshape([batch_size * self.beam, max_decode_length, cif_out_dim]) # (B * beam_size) x T x C
        ctxt_cif_outputs = ctxt_cif_outputs.reshape([batch_size * self.beam, max_decode_length, cif_out_dim]) \
            if ctxt_cif_outputs is not None else None
        prev_tokens = prev_tokens.reshape([batch_size * self.beam, 1])                      # (B * beam_size) x 1
        scores = scores.reshape([batch_size * self.beam])                                   # (B * beam_size)
        cif_out_padding_mask = \
            cif_out_padding_mask.reshape([batch_size * self.beam, max_decode_length])       # (B * beam_size) x T

        for step_i in range(1, max_decode_length + 1):
            # Get cif outputs of current step
            cur_step_cif_outputs = cif_out[:, :step_i, :]                           # (B * beam_size) x t x C
            cur_step_ctxt_cif_outputs = ctxt_cif_outputs[:, :step_i, :] if ctxt_cif_outputs is not None else None
            cur_step_cif_out_padding_mask = cif_out_padding_mask[:, :step_i]        # (B * beam_size) x t
            cur_step_cif_out = {
                "cif_out": cur_step_cif_outputs,
                "ctxt_cif_out": cur_step_ctxt_cif_outputs,
                "padding_mask": cur_step_cif_out_padding_mask
            }

            # Get decoder outputs at step_i
            decoder_output_i, extra_outputs = model.step_forward_decoder(
                prev_decoded_tokens=prev_tokens,    # (B x beam_size) x t
                cif_outputs=cur_step_cif_out,
                # cif_out: (B * beam_size) x t x C, cif_out_padding_mask: (B * beam_size) x t
            )   # decoder_output_i has shape [(B * beam_size), t, V]
            cur_decoder_output = model.get_probs_from_logits(
                decoder_output_i[:, -1, :], log_probs=True) # [B * beam_size, V]
            tmp_scores = scores # Backup scores, with shape [B * beam_size]
            scores = scores.unsqueeze(dim=-1).repeat([1, self.vocab_size])  # [B * beam_size, V]

            cur_score = cur_decoder_output
            # cur_score, with shape [(B x beam_size) x V]

            updated_scores = (scores + cur_score).reshape(
                [batch_size, self.beam * self.vocab_size]
            )   # converted from shape [B * beam_size, V] to [B, beam_size * V]

            # Handle the first timestep with special operation
            if step_i == 1:
                # For the first step, due to the same input token, only consider one beam.
                topk_scores, topk_indices = torch.topk(
                    updated_scores.reshape([batch_size, self.beam, self.vocab_size])[:, 0, :], k=self.beam, dim=-1)
                beam_indices = torch.zeros(batch_size, self.beam).long().cuda() # [B, beam_size] with all zero elements
                fixed_topk_indices = topk_indices   # [B, beam_size]
            else:
                # For all the other beams, due to their inputs are varying, consider all beams.
                topk_scores, topk_indices = torch.topk(
                    updated_scores, k=self.beam, dim=-1
                ) # topk_scores shape [B, beam_size], topk_indices shape [B, beam_size]
                beam_indices = topk_indices // self.vocab_size      # [B, beam_size]
                fixed_topk_indices = topk_indices % self.vocab_size # [B, beam_size]

            # Update previous decoded tokens and scores
            prev_tokens = prev_tokens.reshape([batch_size, self.beam, -1])  # [B, beam_size, t]
            tmp_scores = tmp_scores.reshape([batch_size, self.beam])        # previous scores, with shape [B, beam_size]
            prev_token_tmp_list = []
            scores_tmp_list = []
            for n in range(batch_size): # n ranges from 0 to (batch_size - 1)
                # Get the max length of current sample
                cur_output_maxlen = out_seqlens[n]

                # If some sample's decode length is smaller than current step id, keep its score and decoded results
                if step_i > cur_output_maxlen:
                    cur_scores = tmp_scores[n, :]           # beam_size
                    cur_prev_tokens = prev_tokens[n, :, :]  # beam_size x t
                else:
                    cur_scores = topk_scores[n, :]          # beam_size
                    cur_prev_tokens = prev_tokens[n, :, :]  # beam_size x t
                    cur_beam_indices = beam_indices[n, :]   # beam_size

                    # Get reformed previous tokens
                    cur_prev_tokens = \
                        torch.index_select(cur_prev_tokens, dim=0, index=cur_beam_indices) # beam_size x t

                scores_tmp_list.append(cur_scores.unsqueeze(dim=0))
                prev_token_tmp_list.append(cur_prev_tokens.unsqueeze(dim=0))

            fixed_prev_tokens = torch.cat(prev_token_tmp_list, dim=0)
            fixed_topk_indices = torch.where(
                step_i <= out_seqlens.unsqueeze(dim=-1).repeat([1, self.beam]),
                fixed_topk_indices,                                     # B x beam_size
                torch.ones_like(fixed_topk_indices).cuda() * self.pad,
            )   # Mask locations that outnumber cif max length using <pad>
            fixed_topk_indices = fixed_topk_indices.unsqueeze(dim=-1)   # [B, beam_size, 1]
            prev_tokens = torch.cat(
                [fixed_prev_tokens, fixed_topk_indices], dim=-1
            ).reshape([batch_size * self.beam, -1])                     # [B * beam_size, t + 1]
            scores = torch.cat(scores_tmp_list, dim=0).reshape([batch_size * self.beam])        # [B * beam_size]

        scores = scores.reshape([batch_size, self.beam])[:, :self.nbest]                        # B x beam_size
        prev_tokens = prev_tokens.reshape([batch_size, self.beam, -1])[:, :self.nbest, 1:]      # B x beam_size x T
        out_seqlens = torch.unsqueeze(out_seqlens, dim=-1).repeat(1, self.beam)[:, :self.nbest] # B x beam_size

        return prev_tokens, scores, out_seqlens
