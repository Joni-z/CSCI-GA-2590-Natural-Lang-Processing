# HW1 Writeup (Sentiment Classification)

## 1. Implementation Summary

I implemented all required components in `models.py`:

- `UnigramFeatureExtractor`, `BigramFeatureExtractor`, `BetterFeatureExtractor`
- `LogisticRegressionClassifier` and `train_logistic_regression`
- `NeuralSentimentClassifier`, `DANNetwork`, and `train_deep_averaging_network`

### Logistic Regression

- Features are sparse `Counter` vectors.
- Training uses SGD on logistic log-likelihood with epoch-wise shuffling.
- Learning-rate schedule: `lr / (1 + 0.05 * epoch)`.
- A small L2 shrinkage term is applied during updates.

### DAN

- Input sentence is mapped to word indices using pretrained GloVe vectors.
- The model averages word embeddings.
- A feed-forward classifier (`Linear -> ReLU -> Dropout -> Linear`) predicts sentiment.
- Training uses `NLLLoss` + Adam, with best-dev checkpoint selection.

## 2. Required Results

### Command 1
`python sentiment_classifier.py --model LR --feats UNIGRAM`

- Dev Accuracy: **0.7718**
- Dev F1: **0.7858**
- Time: **1.47s**

### Command 2
`python sentiment_classifier.py --model DAN`

- Dev Accuracy: **0.7970**
- Dev F1: **0.8066**
- Time: **16.60s**

Both models pass the minimum dev-accuracy requirement (>= 77%).

## 3. Exploration

### Exploration A: LR learning-rate schedules + curves

Using `UNIGRAM` with 25 epochs:

- `--lr 0.05`: dev acc **0.7695**
- `--lr 0.2`: dev acc **0.7729**

I plotted:

- average train log-likelihood vs epoch
- dev accuracy vs epoch

(figure file: `writeup/lr_schedule_curves.png`)

Observation: larger LR improves the training objective faster, but dev accuracy remains close.

### Exploration B: Better feature modification (not unigram+bigram concat)

With `--lr 0.1 --num_epochs 25`:

- `--feats BIGRAM`: dev acc **0.7351**
- `--feats BETTER`: dev acc **0.8085**

`BETTER` uses:

- stopword/short-token filtering
- clipped unigram counts (`1/2`)
- negation scope marking (`_NEG` for next 3 content words)
- sentence-level intensifier indicator

Observation: bigram-only strongly overfits; the modified features generalize better.

## 4. Additional DAN Check

I also compared DAN variants:

- 50d embeddings: dev acc **0.7431**
- 300d embeddings + hidden 200: dev acc **0.7936**

This shows 300d embeddings are important in this setup.
