import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import hashlib
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

KEYBOARD_NEIGHBORS = {
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "erfcxs",
    "e": "rdsw",
    "f": "rtgvcd",
    "g": "tyhbvf",
    "h": "yujnbg",
    "i": "uojk",
    "j": "uikmnh",
    "k": "iolmj",
    "l": "opk",
    "m": "njk",
    "n": "bhjm",
    "o": "ipkl",
    "p": "ol",
    "q": "wa",
    "r": "tfde",
    "s": "wedxza",
    "t": "ygfr",
    "u": "yihj",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "uhtg",
    "z": "asx",
}


def _inject_typo(token, rng):
    valid_positions = [idx for idx, ch in enumerate(token) if ch.lower() in KEYBOARD_NEIGHBORS]
    if not valid_positions:
        return token

    position = rng.choice(valid_positions)
    replacement = rng.choice(KEYBOARD_NEIGHBORS[token[position].lower()])
    if token[position].isupper():
        replacement = replacement.upper()

    chars = list(token)
    chars[position] = replacement
    return "".join(chars)


def _drop_vowel(token, rng):
    vowel_positions = [idx for idx, ch in enumerate(token) if ch.lower() in "aeiou"]
    if len(token) <= 4 or not vowel_positions:
        return token

    position = rng.choice(vowel_positions)
    return token[:position] + token[position + 1:]


def _duplicate_character(token, rng):
    if len(token) <= 4:
        return token

    valid_positions = [idx for idx, ch in enumerate(token) if ch.isalpha()]
    if not valid_positions:
        return token

    position = rng.choice(valid_positions)
    return token[:position + 1] + token[position] + token[position + 1:]


def _transpose_adjacent(token, rng):
    if len(token) <= 4:
        return token

    valid_positions = [
        idx
        for idx in range(1, len(token) - 2)
        if token[idx].isalpha() and token[idx + 1].isalpha()
    ]
    if not valid_positions:
        return token

    position = rng.choice(valid_positions)
    chars = list(token)
    chars[position], chars[position + 1] = chars[position + 1], chars[position]
    return "".join(chars)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    seed = int(hashlib.md5(example["text"].encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    transformed_tokens = []

    for token in word_tokenize(example["text"].lower()):
        transformed_token = token
        if token.isalpha() and len(token) > 3:
            noise_roll = rng.random()
            if noise_roll < 0.22:
                transformed_token = _inject_typo(transformed_token, rng)
            elif noise_roll < 0.31:
                transformed_token = _drop_vowel(transformed_token, rng)
            elif noise_roll < 0.39:
                transformed_token = _duplicate_character(transformed_token, rng)
            elif noise_roll < 0.45:
                transformed_token = _transpose_adjacent(transformed_token, rng)

        transformed_tokens.append(transformed_token)

    example["text"] = TreebankWordDetokenizer().detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
