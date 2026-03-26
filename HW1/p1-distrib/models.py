# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        for word in sentence:
            feat = "UNI=" + word
            feat_idx = self.indexer.add_and_get_index(feat, add=add_to_indexer)
            if feat_idx != -1:
                feats[feat_idx] += 1.0
        return feats


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        for i in range(0, len(sentence) - 1):
            feat = "BI=" + sentence[i] + "|" + sentence[i + 1]
            feat_idx = self.indexer.add_and_get_index(feat, add=add_to_indexer)
            if feat_idx != -1:
                feats[feat_idx] += 1.0
        return feats


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
            "he", "her", "hers", "him", "his", "i", "in", "is", "it", "its", "me", "my",
            "of", "on", "or", "our", "ours", "she", "that", "the", "their", "theirs",
            "them", "they", "this", "to", "us", "was", "we", "were", "with", "you", "your"
        }
        self.negators = {"not", "no", "never", "n't"}
        self.intensifiers = {"very", "really", "so", "too", "extremely", "super"}

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        clipped_counts = Counter()
        neg_scope = 0
        has_intensifier = False

        for raw_word in sentence:
            word = raw_word.strip().lower()
            if word == "":
                continue
            if word in self.intensifiers:
                has_intensifier = True
            if word in self.negators:
                neg_scope = 3
                continue
            if word in {".", "!", "?", ",", ";"}:
                neg_scope = 0
                continue

            if word in self.stopwords or len(word) <= 2:
                if neg_scope > 0:
                    neg_scope -= 1
                continue

            feat_word = word + "_NEG" if neg_scope > 0 else word
            clipped_counts[feat_word] += 1.0
            if neg_scope > 0:
                neg_scope -= 1

        for feat_word, count in clipped_counts.items():
            feat = "UNI_CLIP=" + feat_word
            feat_idx = self.indexer.add_and_get_index(feat, add=add_to_indexer)
            if feat_idx != -1:
                feats[feat_idx] = 2.0 if count >= 2.0 else 1.0

        if has_intensifier:
            feat_idx = self.indexer.add_and_get_index("HAS_INTENSIFIER", add=add_to_indexer)
            if feat_idx != -1:
                feats[feat_idx] = 1.0
        return feats


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, bias: float, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.bias = bias
        self.feat_extractor = feat_extractor

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = np.exp(x)
            return z / (1.0 + z)

    def _score(self, ex_words: List[str]) -> float:
        feats = self.feat_extractor.extract_features(ex_words, add_to_indexer=False)
        score = self.bias
        for feat_idx, value in feats.items():
            if feat_idx < len(self.weights):
                score += self.weights[feat_idx] * value
        return score

    def predict(self, ex_words: List[str]) -> int:
        prob = self._sigmoid(self._score(ex_words))
        return 1 if prob >= 0.5 else 0


def train_logistic_regression(train_exs: List[SentimentExample],
                              feat_extractor: FeatureExtractor,
                              num_epochs: int = 25,
                              learning_rate: float = 0.1,
                              l2_lambda: float = 1e-6) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    random.seed(42)
    np.random.seed(42)

    train_feat_vecs = []
    train_labels = []
    for ex in train_exs:
        train_feat_vecs.append(feat_extractor.extract_features(ex.words, add_to_indexer=True))
        train_labels.append(ex.label)

    num_features = len(feat_extractor.get_indexer())
    weights = np.zeros(num_features, dtype=np.float64)
    bias = 0.0

    for epoch in range(num_epochs):
        indices = list(range(len(train_exs)))
        random.shuffle(indices)
        step_size = learning_rate / (1.0 + 0.05 * epoch)
        for idx in indices:
            feats = train_feat_vecs[idx]
            y = train_labels[idx]
            score = bias
            for feat_idx, value in feats.items():
                score += weights[feat_idx] * value
            # p(y=1|x)
            if score >= 0:
                z = np.exp(-score)
                pred_prob = 1.0 / (1.0 + z)
            else:
                z = np.exp(score)
                pred_prob = z / (1.0 + z)

            error = y - pred_prob
            for feat_idx, value in feats.items():
                # L2-shrunk SGD update.
                weights[feat_idx] = (1.0 - step_size * l2_lambda) * weights[feat_idx] + step_size * error * value
            bias += step_size * error

    return LogisticRegressionClassifier(weights, bias, feat_extractor)


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    lr = args.lr if args.lr != 0.001 else 0.1
    num_epochs = args.num_epochs if args.num_epochs != 10 else 25
    model = train_logistic_regression(train_exs, feat_extractor, num_epochs=num_epochs, learning_rate=lr)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        self.network = network
        self.word_embeddings = word_embeddings
        self.word_indexer = word_embeddings.word_indexer
        self.unk_idx = self.word_indexer.index_of("UNK")

    def _words_to_tensor(self, ex_words: List[str]) -> torch.Tensor:
        idxs = []
        for word in ex_words:
            idx = self.word_indexer.index_of(word)
            idxs.append(idx if idx != -1 else self.unk_idx)
        if len(idxs) == 0:
            idxs = [self.unk_idx]
        return torch.tensor(idxs, dtype=torch.long)

    def predict(self, ex_words: List[str]) -> int:
        self.network.eval()
        with torch.no_grad():
            word_idxs = self._words_to_tensor(ex_words)
            log_probs = self.network(word_idxs)
            return int(torch.argmax(log_probs, dim=1).item())


class DANNetwork(nn.Module):
    def __init__(self, embedding_layer: nn.Embedding, embedding_dim: int, hidden_size: int, dropout: float = 0.2):
        super().__init__()
        self.embedding = embedding_layer
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_idxs: torch.Tensor) -> torch.Tensor:
        if word_idxs.dim() == 1:
            word_idxs = word_idxs.unsqueeze(0)
        embedded = self.embedding(word_idxs)
        avg_emb = torch.mean(embedded, dim=1)
        logits = self.ff(avg_emb)
        return self.log_softmax(logits)


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    word_indexer = word_embeddings.word_indexer
    unk_idx = word_indexer.index_of("UNK")
    embedding_dim = word_embeddings.get_embedding_length()
    hidden_size = args.hidden_size
    learning_rate = args.lr
    num_epochs = args.num_epochs

    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
    network = DANNetwork(embedding_layer, embedding_dim, hidden_size, dropout=0.2)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    def words_to_tensor(words: List[str]) -> torch.Tensor:
        idxs = []
        for word in words:
            idx = word_indexer.index_of(word)
            idxs.append(idx if idx != -1 else unk_idx)
        if len(idxs) == 0:
            idxs = [unk_idx]
        return torch.tensor(idxs, dtype=torch.long)

    train_inputs = [words_to_tensor(ex.words) for ex in train_exs]
    train_labels = [ex.label for ex in train_exs]
    dev_inputs = [words_to_tensor(ex.words) for ex in dev_exs]
    dev_labels = [ex.label for ex in dev_exs]

    best_dev_acc = -1.0
    best_state = None

    for epoch in range(num_epochs):
        network.train()
        indices = list(range(len(train_inputs)))
        random.shuffle(indices)
        running_loss = 0.0
        for i in indices:
            x = train_inputs[i]
            y = torch.tensor([train_labels[i]], dtype=torch.long)
            optimizer.zero_grad()
            log_probs = network(x)
            loss = loss_fn(log_probs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        network.eval()
        with torch.no_grad():
            num_correct = 0
            for x, gold in zip(dev_inputs, dev_labels):
                pred = int(torch.argmax(network(x), dim=1).item())
                if pred == gold:
                    num_correct += 1
            dev_acc = float(num_correct) / len(dev_labels)
        print("Epoch %d: train_loss=%.4f dev_acc=%.4f" % (epoch + 1, running_loss / len(train_inputs), dev_acc))
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}

    if best_state is not None:
        network.load_state_dict(best_state)
    network.eval()
    return NeuralSentimentClassifier(network, word_embeddings)
