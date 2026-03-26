import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import UnigramFeatureExtractor
from sentiment_data import read_sentiment_examples
from utils import Indexer


def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def log_likelihood(feats_list, labels, weights, bias):
    total = 0.0
    for feats, y in zip(feats_list, labels):
        score = bias
        for feat_idx, value in feats.items():
            score += weights[feat_idx] * value
        if y == 1:
            total += -np.logaddexp(0.0, -score)
        else:
            total += -np.logaddexp(0.0, score)
    return total / len(labels)


def dev_accuracy(dev_feats, dev_labels, weights, bias):
    correct = 0
    for feats, y in zip(dev_feats, dev_labels):
        score = bias
        for feat_idx, value in feats.items():
            score += weights[feat_idx] * value
        pred = 1 if sigmoid(score) >= 0.5 else 0
        if pred == y:
            correct += 1
    return correct / len(dev_labels)


def run(lr, num_epochs=25, l2_lambda=1e-6, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    train_exs = read_sentiment_examples('data/train.txt')
    dev_exs = read_sentiment_examples('data/dev.txt')

    feat_extractor = UnigramFeatureExtractor(Indexer())
    train_feats = [feat_extractor.extract_features(ex.words, add_to_indexer=True) for ex in train_exs]
    train_labels = [ex.label for ex in train_exs]
    dev_feats = [feat_extractor.extract_features(ex.words, add_to_indexer=False) for ex in dev_exs]
    dev_labels = [ex.label for ex in dev_exs]

    weights = np.zeros(len(feat_extractor.get_indexer()), dtype=np.float64)
    bias = 0.0

    ll_hist = []
    dev_hist = []

    for epoch in range(num_epochs):
        idxs = list(range(len(train_feats)))
        random.shuffle(idxs)
        step_size = lr / (1.0 + 0.05 * epoch)

        for i in idxs:
            feats = train_feats[i]
            y = train_labels[i]
            score = bias
            for feat_idx, value in feats.items():
                score += weights[feat_idx] * value
            p = sigmoid(score)
            error = y - p
            for feat_idx, value in feats.items():
                weights[feat_idx] = (1.0 - step_size * l2_lambda) * weights[feat_idx] + step_size * error * value
            bias += step_size * error

        ll_hist.append(log_likelihood(train_feats, train_labels, weights, bias))
        dev_hist.append(dev_accuracy(dev_feats, dev_labels, weights, bias))

    return ll_hist, dev_hist


def main():
    num_epochs = 25
    lrs = [0.05, 0.2]
    histories = {}

    for lr in lrs:
        ll_hist, dev_hist = run(lr=lr, num_epochs=num_epochs)
        histories[lr] = (ll_hist, dev_hist)
        print(f"lr={lr} final_train_ll={ll_hist[-1]:.4f} final_dev_acc={dev_hist[-1]:.4f}")

    epochs = np.arange(1, num_epochs + 1)
    plt.figure(figsize=(9, 3.8))

    plt.subplot(1, 2, 1)
    for lr in lrs:
        ll_hist, _ = histories[lr]
        plt.plot(epochs, ll_hist, label=f"lr={lr}")
    plt.xlabel('Epoch')
    plt.ylabel('Avg train log-likelihood')
    plt.title('Training Objective')
    plt.grid(alpha=0.25)
    plt.legend()

    plt.subplot(1, 2, 2)
    for lr in lrs:
        _, dev_hist = histories[lr]
        plt.plot(epochs, dev_hist, label=f"lr={lr}")
    plt.xlabel('Epoch')
    plt.ylabel('Dev accuracy')
    plt.title('Development Accuracy')
    plt.ylim(0.74, 0.79)
    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()
    plt.savefig('writeup/lr_schedule_curves.png', dpi=220)


if __name__ == '__main__':
    main()
