# transformer_lm.py

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
from utils import *


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 128):
        super().__init__()
        self.emb = nn.Embedding(num_positions, d_model)

    def forward(self, x):
        # x: (seq_len, d_model)
        seq_len = x.shape[0]
        indices = torch.arange(seq_len, dtype=torch.long, device=x.device)
        return x + self.emb(indices)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, nhead, num_positions):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, num_positions)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.mask_cache = {}

    def _causal_mask(self, seq_len, device):
        key = (seq_len, device)
        if key not in self.mask_cache:
            self.mask_cache[key] = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1
            )
        return self.mask_cache[key]

    def forward(self, indices):
        # indices: (seq_len,) LongTensor
        seq_len = indices.size(0)
        x = self.embed(indices)   # (seq_len, d_model)
        x = self.pos_enc(x)       # (seq_len, d_model)
        x = x.unsqueeze(0)        # (1, seq_len, d_model)
        # Causal mask: prevent position i from attending to j > i
        mask = self._causal_mask(seq_len, indices.device)
        x = self.transformer(x, mask=mask)  # (1, seq_len, d_model)
        x = x.squeeze(0)          # (seq_len, d_model)
        logits = self.output_proj(x)        # (seq_len, vocab_size)
        return torch.log_softmax(logits, dim=-1)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model: TransformerLM, vocab_index: Indexer, chunk_size: int):
        self.model = model
        self.vocab_index = vocab_index
        self.chunk_size = chunk_size
        self.char_to_idx = vocab_index.objs_to_ints

    def _context_to_indices(self, context: str) -> torch.LongTensor:
        """
        Convert context string to an input tensor for the model.
        Prepends space (SOS), then takes up to the last (chunk_size - 1) chars of context.
        """
        space_idx = self.vocab_index.index_of(' ')
        if len(context) == 0:
            return torch.LongTensor([space_idx])
        ctx = context[-(self.chunk_size - 1):]
        indices = [self.char_to_idx[c] for c in ctx]
        return torch.LongTensor([space_idx] + indices)

    def get_next_char_log_probs(self, context) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            indices = self._context_to_indices(context)
            log_probs = self.model(indices)   # (seq_len, vocab_size)
        return log_probs[-1].numpy()

    def get_log_prob_sequence(self, next_chars, context) -> float:
        total = 0.0
        for i, c in enumerate(next_chars):
            log_probs = self.get_next_char_log_probs(context + next_chars[:i])
            char_idx = self.vocab_index.index_of(c)
            total += float(log_probs[char_idx])
        return total


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    vocab_size = len(vocab_index)   # 27
    chunk_size = int(os.environ.get("LM_CHUNK_SIZE", "64"))
    d_model = int(os.environ.get("LM_D_MODEL", "96"))
    d_ff = int(os.environ.get("LM_D_FF", "256"))
    num_layers = int(os.environ.get("LM_NUM_LAYERS", "2"))
    nhead = int(os.environ.get("LM_NHEAD", "4"))
    num_positions = chunk_size      # max input length = chunk_size (SOS + chunk_size-1 chars)
    lr = float(os.environ.get("LM_LR", "0.001"))
    num_epochs = int(os.environ.get("LM_EPOCHS", "20"))

    if d_model % nhead != 0:
        raise ValueError("LM_D_MODEL must be divisible by LM_NHEAD")

    model = TransformerLM(vocab_size, d_model, d_ff, num_layers, nhead, num_positions)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fcn = nn.NLLLoss()

    space_idx = vocab_index.index_of(' ')
    char_to_idx = vocab_index.objs_to_ints
    # Store training indices compactly to avoid Python-list memory overhead.
    all_indices = np.fromiter((char_to_idx[c] for c in train_text), dtype=np.uint8, count=len(train_text))
    n = int(all_indices.shape[0])
    if n < chunk_size:
        raise ValueError("Training text shorter than chunk size")

    # Build chunk start positions (non-overlapping)
    chunk_starts = np.arange(0, n - chunk_size + 1, chunk_size, dtype=np.int64)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        np.random.shuffle(chunk_starts)

        for start in chunk_starts:
            start = int(start)
            target_indices = torch.from_numpy(all_indices[start:start + chunk_size].astype(np.int64, copy=False))
            # Use actual previous char as first input token (space for the very first chunk)
            first_idx = int(all_indices[start - 1]) if start > 0 else space_idx
            input_indices = torch.empty(chunk_size, dtype=torch.long)
            input_indices[0] = first_idx
            input_indices[1:] = target_indices[:-1]

            optimizer.zero_grad(set_to_none=True)
            log_probs = model(input_indices)   # (chunk_size, vocab_size)
            loss = loss_fcn(log_probs, target_indices)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print("Epoch %i loss: %.4f" % (epoch, total_loss / max(1, len(chunk_starts))))

    model.eval()
    return NeuralLanguageModel(model, vocab_index, chunk_size)
