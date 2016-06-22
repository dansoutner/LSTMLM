# LSTM language model toolkit

====

LSTM Neural Network in Python and Chainer,
used for language modelling.

FEATS:
- runs on GPU, uses minibatches
- 1 to 3 layer architectures
- allows use of external word features (See: D Soutner, L Müller; On Continuous Space Word Representations as Input of LSTM Language Model
 Statistical Language and Speech Processing, 267-274)


Based on LSTM RNN, model proposed by Jürgen Schmidhuber
http://www.idsia.ch/~juergen/

Implemented by Daniel Soutner,
Department of Cybernetics, University of West Bohemia, Plzen, Czechia
dsoutner@kky.zcu.cz, 2016

Licensed under the 3-clause BSD.

### Requirements

You will need:
- python >= 2.6

Python libs:
- chainer >= 0.17 (chainer.org)
- numpy
- argparse (is in python 2.7 and higher)
- gensim (for FV extension)

### Usage

train LSTM LM on text and save
```
python lstm.py --train train.txt --valid dev.txt --test test.txt --hidden 100 --num-layers 2 --save-net example.lstm-lm
```

load net and evaluate on perplexity
```
python lstm.py --initmodel example.lstm-lm --ppl valid2.txt
```

load net, combine with ARPA LM (weight 0.2) and evaluate
```
python lstm.py --initmodel example.lstm-lm --ppl valid2.txt --ngram ngram.model.arpa 0.2
```

load net and rescore nbest list
```
python lstm.py --initmodel example.lstm-lm --nbest nbest.list --wip 0 --lmw 11
```

### Extranal feature vectors

You can use externally pre-computed feature vectors, from tool such as word2vec, GloVe etc. This can boost performance by about 5% on perplexity. More in D Soutner, L Müller; On Continuous Space Word Representations as Input of LSTM Language Model
 Statistical Language and Speech Processing, 267-274.

### TODO

- add hierarchical softmax on output layer for speed-up by big models
- better document FV option
- test n-gram interpolation and nbest scoring
