#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  lstmlm.py
#  
#  Copyright 2016 Daniel Soutner <dsoutner@kky.zcu.cz>
#  Department of Cybernetics, University of West Bohemia, Plzen, Czech rep.
#  dsoutner@kky.zcu.cz, 2014; Licensed under the 3-clause BSD.
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
"""
Sample script of recurrent neural network language model.
This code is ported from following implementation written in Torch.
https://github.com/tomsercu/lstm

Inspired by examples in chainer toolkit
"""

from __future__ import print_function
from __future__ import division

__version__ = "0.7.0"

# imports
import argparse
import math
import sys
import time
import tarfile
import cPickle
import numpy as np
import six
import os

# import chainer
import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers
from chainer.functions.evaluation import accuracy

# local imports
import net
import ArpaLM
import gensim2 # modified gensim package

# various configurations

TinyConfig = {
	"init_scale" : 0.1,
	"learning_rate" : 1.0,
	"max_grad_norm" : 5,
	"num_layers" : 1,
	"num_steps" : 20,
	"hidden_size" : 100,
	"max_epoch" : 4,
	"max_max_epoch" : 11,
	"keep_prob" : 1.0,
	"lr_decay" : 0.5,
	"batch_size" : 30,
	}

# PTB - optimal https://github.com/wojzaremba/lstm/pull/5
OptimalConfig = {
	"init_scale" : 0.11,
	"learning_rate" : 0.71,
	"max_grad_norm" : 16.57,
	"num_layers" : 2, #fixed parameter
	"num_steps" : 20, #fixed parameter
	"hidden_size" : 509,
	"max_epoch" : 8,
	"max_max_epoch" : 13, #fixed parameter
	"keep_prob" : 1-0.42,
	"lr_decay" : 1/2.79,
	"batch_size" : 30, # fixed parameter
	}

# the same as https://github.com/wojzaremba/lstm
SmallConfig = {
	"init_scale" : 0.1,
	"learning_rate" : 1.0,
	"max_grad_norm" : 5,
	"num_layers" : 2,
	"num_steps" : 20,
	"hidden_size" : 200,
	"max_epoch" : 4,
	"max_max_epoch" : 13,
	"keep_prob" : 1.0,
	"lr_decay" : 0.5,
	"batch_size" : 20,
	}

MediumConfig = {
	"init_scale" : 0.05,
	"learning_rate" : 1.0,
	"max_grad_norm" : 5,
	"num_layers" : 2,
	"num_steps" : 35,
	"hidden_size" : 640,
	"max_epoch" : 6,
	"max_max_epoch" : 39,
	"keep_prob" : 0.5,
	"lr_decay" : 0.8,
	"batch_size" : 20,
	}

LargeConfig = {
	"init_scale" : 0.04,
	"learning_rate" : 1.0,
	"max_grad_norm" : 10,
	"num_layers" : 2,
	"num_steps" : 35,
	"hidden_size" : 1000,
	"max_epoch" : 14,
	"max_max_epoch" : 55,
	"keep_prob" : 0.35,
	"lr_decay" : 1 / 1.15,
	"batch_size" : 30,
	}

# https://github.com/karpathy/char-rnn/blob/master/train.lua
CharConfig = {
	"init_scale" : 0.08,
	"learning_rate" : 0.002,
	"max_grad_norm" : 5,
	"num_layers" : 2,
	"num_steps" : 50,
	"hidden_size" : 128,
	"max_epoch" : 10,
	"max_max_epoch" : 50,
	"keep_prob" : 0.35,
	"lr_decay" : 0.97,
	"batch_size" : 20,
	}

UNK = "<unk>"
EOS = "</s>"


class LSTMLM:

	def __init__(self, args):

		if args.config == "tiny":
			self.config = TinyConfig
		if args.config == "optimal":
			self.config = OptimalConfig
		if args.config == "small":
			self.config = SmallConfig
		if args.config == "medium":
			self.config = MediumConfig
		if args.config == "large":
			self.config = LargeConfig
		if args.char:
			self.config = CharConfig
		# change config only if training
		if args.train:
			if args.hidden_size:
				self.config["hidden_size"] = args.hidden_size
			if args.num_layers:
				self.config["num_layers"] = args.num_layers
		self.vocab = {}
		self.ivocab = {}
		self.model = None
		self.optimizer = None
		self.arpaLM = None
		self.arpaLM_weight = None

		if args.save_net:
			self.save_net = args.save_net
		else:
			self.save_net = "_".join(["foo", str(self.config["hidden_size"]), str(self.config["num_layers"])])

		if args.train and args.valid:
			if args.char:
				train_data = self.load_char_data(args.train, update_vocab=True)
				valid_data = self.load_char_data(args.valid)
				print('#vocab =', len(self.vocab))
			else:
				train_data, self.train_text = self.load_text_data(args.train, update_vocab=True)
				valid_data, self.valid_text = self.load_text_data(args.valid)
				print('#vocab =', len(self.vocab))

		if args.ngram:
			assert os.path.exists(args.ngram[0])
			assert 1.0 >= float(args.ngram[1]) >= 0.0

			self.arpaLM = ArpaLM.ArpaLM(path=args.ngram[0])
			self.arpaLM_weight = float(args.ngram[1])

		# Init/Resume
		if args.initmodel:
			self.load_net(args.initmodel, resume=args.resume)
		else:
			self.init_net(self.config)

		if args.gpu >= 0 and not args.cpu:
			cuda.get_device(args.gpu).use()
			self.model.to_gpu()

		if args.train:
			self.train(train_data, valid_data)

		if args.test:
			if args.char:
				test_data = self.load_char_data(args.test)
			else:
				test_data, test_text = self.load_text_data(args.test)
			test_perp = self.evaluate(test_data, text=test_text)
			print('test perplexity:', test_perp)

		if args.ppl:
			if args.char:
				test_data = self.load_char_data(args.ppl)
			else:
				test_data, test_text = self.load_text_data(args.ppl)
			test_perp = self.evaluate(test_data, text=test_text)
			print('test perplexity:', test_perp)

		if args.nbest:
			self.nbest(args.nbest)


	def save(self, filename):
		""" Save the model, the optimizer, vocabulary and config"""
		filename = os.path.abspath(filename)
		serializers.save_hdf5(filename+'.model', self.model)
		serializers.save_hdf5(filename+'.state', self.optimizer)
		cPickle.dump(self.vocab, open(filename+'.vocab', "w"))
		cPickle.dump(self.config, open(filename+'.config', "w"))
		with tarfile.open(filename, "w") as tar:
			for fn in [filename+'.model', filename+'.state', filename+'.vocab', filename+'.config']:
				tar.add(fn, arcname=os.path.basename(fn))
				os.remove(fn)


	def load_text_data(self, filename, update_vocab=False):
		"""Loads text to ints, update vocab if text is training"""
		words = open(filename).read().replace('\n', ' '+EOS+' ').strip().split()
		dataset = np.ndarray((len(words),), dtype=np.int32)
		if EOS not in self.vocab and update_vocab:
			self.vocab[EOS] = len(self.vocab)
		if UNK not in self.vocab and update_vocab:
			self.vocab[UNK] = len(self.vocab)
		for i, word in enumerate(words):
			if word not in self.vocab:
				if update_vocab:
					self.vocab[word] = len(self.vocab)
				else:
					word = UNK
			dataset[i] = self.vocab[word]
		if update_vocab:
			tree = {}
			for word in words:
				idx = self.vocab[word]
				try:
					tree[idx] += 1
				except KeyError:
					tree[idx] = 1
			self.tree = L.BinaryHierarchicalSoftmax.create_huffman_tree(tree)
		return dataset, words


	def load_char_data(self, filename, update_vocab=False):
		"""Loads text to ints, update vocab if text is training"""
		words = open(filename).read()#.replace('\n', ' '+EOS+' ').strip().split()
		dataset = np.ndarray((len(words),), dtype=np.int32)
		for i, word in enumerate(words):
			if word not in self.vocab:
				if update_vocab:
					self.vocab[word] = len(self.vocab)
			dataset[i] = self.vocab[word]
		return dataset


	def evaluate(self, dataset, text=None):
		""""Evaluate net on input dataset 
			return perplexity"""
		evaluator = self.model.copy()
		evaluator.predictor.reset_state()  # initialize state
		evaluator.predictor.train = False  # dropout does nothing

		LOG10TOLOG = np.log(10)
		LOGTOLOG10 = 1. / LOG10TOLOG

		sum_log_perp = 0
		oov_arpa = 0
		oov_net = 0
		oov_comb = 0

		self.ivocab = {v: k for k, v in self.vocab.items()}

		for i in six.moves.range(dataset.size - 1):
			x = chainer.Variable(xp.asarray(dataset[i : i + 1]), volatile='on')
			t = chainer.Variable(xp.asarray(dataset[i + 1 : i + 2]), volatile='on')
			loss = evaluator(x, t)

			if self.arpaLM and text:
				ctx = text[max(0, i - 10): i + 2]
				ctx = ctx[::-1]

				# ARPA takes only the last sentence
				if "</s>" in ctx:
					idx = ctx.index("</s>")
					if idx == 0:
						try:
							idx = ctx[1:].index("</s>")
						except ValueError:
							idx = len(ctx) - 1
					ctx = ctx[:idx + 1]
				# add <s> to every </s>
				if len(ctx) == 2 and ctx[len(ctx) - 1] == "</s>":
					ctx.insert(1, "<s>")

				le_arpa = self.arpaLM.prob(*ctx)
				l10_arpa = LOGTOLOG10 * le_arpa
				# both OOV
				if l10_arpa <= -99 and self.ivocab[dataset[i + 1]] == UNK:
					# TODO if unk, then get net OOV prob or 0. (???)
					net_prob = math.exp(-loss.data)
					comb_prob = net_prob * (1. - self.arpaLM_weight)
					sum_log_perp -= math.log(comb_prob)
					oov_comb += 1
				# net OOV
				elif l10_arpa > -99 and self.ivocab[dataset[i+1]] == UNK:
					arpa_prob = math.exp(le_arpa)
					comb_prob = arpa_prob
					sum_log_perp -= math.log(comb_prob)
					oov_net += 1
				# ngram OOV
				elif l10_arpa <= -99:
					net_prob = math.exp(-loss.data)
					comb_prob = net_prob * (1. - self.arpaLM_weight)
					sum_log_perp -= math.log(comb_prob)
					oov_arpa += 1
				# combine
				else:
					arpa_prob = math.exp(le_arpa)
					net_prob = math.exp(-loss.data)
					comb_prob = net_prob * (1. - self.arpaLM_weight) + arpa_prob * self.arpaLM_weight
					sum_log_perp -= math.log(comb_prob)
			else:
				sum_log_perp += loss.data

		if self.arpaLM:
			print("OOV both", oov_comb)
			print("OOV n-gram", oov_arpa)
		print("OOV net", oov_net)
		print('logprob', sum_log_perp)
		return math.exp(float(sum_log_perp) / (dataset.size - 1))


	def load_nbest_data(self, filename):
		"""Helper to load nbestlist to ints"""
		for line in open(filename):
			words = line.strip().split() + [EOS]
			if len(words) == 1:
				words = [UNK, EOS]
			dataset = np.ndarray((len(words),), dtype=np.int32)
			for i, word in enumerate(words):
				if word not in self.vocab:
						word = UNK
				dataset[i] = self.vocab[word]
			yield dataset


	def nbest(self, filename):
		"""Print logprog computed for every line of input file"""

		evaluator = self.model.copy()  # to use different state
		evaluator.predictor.reset_state()  # initialize state

		for dataset in self.load_nbest_data(filename):
			sum_log_perp = 0
			for i in six.moves.range(dataset.size - 1):
				x = chainer.Variable(xp.asarray(dataset[i:i + 1]), volatile='on')
				t = chainer.Variable(xp.asarray(dataset[i + 1:i + 2]), volatile='on')
				loss = evaluator(x, t)
				sum_log_perp += loss.data
			evaluator.predictor.reset_state()
			print(-sum_log_perp)


	def init_net(self, config, randomize=True):
		"""Prepare RNNLM model, defined in net.py"""
		if config["num_layers"] == 1:
			self.lm = net.RNNLM_1layer(len(self.vocab), config["hidden_size"], ratio=config["keep_prob"], train=True)
		elif config["num_layers"] == 2:
			self.lm = net.RNNLM_2layer(len(self.vocab), config["hidden_size"], ratio=config["keep_prob"], train=True)
		elif config["num_layers"] == 3:
			self.lm = net.RNNLM_3layer(len(self.vocab), config["hidden_size"], ratio=config["keep_prob"], train=True)
		else:
			raise KeyError("Num of layers could be only from 1 to 3")

		# softmax
		self.model = L.Classifier(self.lm)
		# or hieararchical softmax
		#self.model = HSMmodel(len(self.vocab), config["hidden_size"], self.tree, ratio=config["keep_prob"], train=True)
		self.model.compute_accuracy = False  # we only want the perplexity

		if randomize:
			for param in self.model.params():
				param.data[:] = np.random.uniform(-config["init_scale"], config["init_scale"], param.data.shape)

		# Setup optimizer
		self.optimizer = optimizers.SGD(lr=config["learning_rate"])
		#self.optimizer = optimizers.Adam()
		self.optimizer.setup(self.model)
		self.optimizer.add_hook(chainer.optimizer.GradientClipping(config["max_grad_norm"]))


	def train(self, train_data, valid_data):
		"""
		Trains all epochs from self.config (max_expoch and max_max_epoch)
		"""
		whole_len = train_data.shape[0]
		jump = whole_len // self.config["batch_size"]
		cur_log_perp = xp.zeros(())
		epoch = 0
		start_at = time.time()
		cur_at = start_at
		accum_loss = 0
		batch_idxs = list(range(self.config["batch_size"]))
		print('Going to train through {} words in {} epochs'.format(jump * self.config["max_max_epoch"] * self.config["batch_size"], self.config["max_max_epoch"]))
		PRINT_POINT = (jump // 10)

		#self.save(self.save_net+"."+str(epoch)+".lstm")
		self.ivocab = {v: k for k, v in self.vocab.items()}

		for i in six.moves.range(jump * self.config["max_max_epoch"]):

			x = chainer.Variable(xp.asarray(
				[train_data[(jump * j + i) % whole_len] for j in batch_idxs]))

			t = chainer.Variable(xp.asarray(
				[train_data[(jump * j + i + 1) % whole_len] for j in batch_idxs]))

			loss_i = self.model(x, t)
			#print(loss_i.data)
			accum_loss += loss_i
			cur_log_perp += loss_i.data

			if (i + 1) % self.config["num_steps"] == 0:  # Run truncated BPTT
				self.model.zerograds()
				accum_loss.backward()
				accum_loss.unchain_backward()  # truncate
				accum_loss = 0
				self.optimizer.update()

			if (i + 1) % PRINT_POINT == 0:
				now = time.time()
				throuput = PRINT_POINT * self.config["batch_size"] / (now - cur_at)
				perp = math.exp(float(cur_log_perp) / PRINT_POINT)
				#perp = float(cur_log_perp) / PRINT_POINT
				print('iter {:.1f} training perplexity: {:.2f} ({:.2f} words/sec)'.format((i + 1) / jump, perp, throuput))
				cur_at = now
				cur_log_perp.fill(0)
				sys.stdout.flush()

			if (i + 1) % jump == 0:
				epoch += 1
				now = time.time()
				perp = self.evaluate(valid_data, text=self.valid_text)
				print('epoch {} validation perplexity: {:.2f}'.format(epoch, perp))
				cur_at += time.time() - now  # skip time of evaluation

				self.save(self.save_net+"."+str(epoch)+".lstm")

				if epoch >= self.config["max_epoch"]:
					self.optimizer.lr *= self.config["lr_decay"]
					print('learning rate =', self.optimizer.lr)
				sys.stdout.flush()


	def load_net(self, filename, resume=None):
		"""
		Loads nnet from filename
		"""
		filename = os.path.abspath(filename)
		pathname = os.path.dirname(filename)
		with tarfile.open(filename, "r") as tar:
			tar.extractall(path=pathname)
		self.config = cPickle.load(open(filename+'.config', "r"))
		self.vocab = cPickle.load(open(filename+'.vocab', "r"))
		self.init_net(self.config, randomize=False)
		serializers.load_hdf5(filename+'.model', self.model)
		if resume:
			serializers.load_hdf5(filename+'.state', self.optimizer)


	def getLogP(list_of_words):
		"""Gets logprob of list of input words"""
		pass


class LSTMLM_FV(LSTMLM):
	
	def __init__(self, args):

		if args.config == "optimal":
			self.config = OptimalConfig
		if args.config == "tiny":
			self.config = TinyConfig
		if args.config == "small":
			self.config = SmallConfig
		if args.config == "medium":
			self.config = MediumConfig
		if args.config == "large":
			self.config = LargeConfig

		# change config only if training
		if args.train:
			if args.hidden_size:
				self.config["hidden_size"] = args.hidden_size
			if args.num_layers:
				self.config["num_layers"] = args.num_layers

		self.args = args

		self.vocab = {}
		self.model = None
		self.optimizer = None
		self.stopwords = []
		self.cache_len = 0

		if args.save_net:
			self.save_net = args.save_net
		else:
			self.save_net = "_".join(["foo", str(self.config["hidden_size"]), str(self.config["num_layers"])])

		if args.train and args.valid:
			train_data, train_words = self.load_text_data(args.train, update_vocab=True)
			valid_data, valid_words = self.load_text_data(args.valid)
			print('#vocab =', len(self.vocab))

		if args.fv1:
			
			self.fv1model = gensim2.models.Word2Vec.load(args.fv1)

			k = self.fv1model.vocab.keys()[0]
			# get length
			self.input_size = self.fv1model[k].shape[0]
			self.fv_len = self.input_size

			if not UNK in self.fv1model.vocab.keys():
				self.fv[UNK] = np.zeros((self.fv_len, )).astype(dtype=xp.float32)

			self.fv = {}
			for k in self.fv1model.vocab.keys():
				self.fv[k] = self.fv1model[k]

			# load words in vocab and not in FVs
			for k in self.vocab.keys():
				if k not in self.fv:
					self.fv[k] = self.fv[UNK]

			self.make_fv = self.FV
		if args.fv:
			self.fv = cPickle.load(open(args.fv))
			for k in self.fv.keys():
				self.fv[k] = self.fv[k].astype(dtype=xp.float32)

			# get length
			self.input_size = self.fv[k].shape[0]
			self.fv_len = self.fv[k].shape[0]

			# set UNK if not in FVs
			if not UNK in self.fv.keys():
				self.fv[UNK] = np.zeros((self.fv_len, )).astype(dtype=xp.float32)

			# load words in vocab and not in FVs
			for k in self.vocab.keys():
				if k not in self.fv:
					self.fv[k] = self.fv[UNK]
			self.make_fv = self.FV

		# Init/Resume
		if args.initmodel:
			self.load_net(args.initmodel, resume=args.resume)
		else:
			self.init_net(self.config)

		if args.gpu >= 0 and not args.cpu:
			cuda.get_device(args.gpu).use()
			self.model.to_gpu()

		if args.train:
			self.train(train_data, train_words, valid_data, valid_words)

		if args.test:
			if args.fv1:
				sentences = gensim2.models.word2vec.LineSentence(args.test)
				self.fv1model.build_vocab(sentences, update=True)

			test_data, test_words = self.load_text_data(args.test)
			test_perp = self.evaluate(test_data, test_words)
			print('test perplexity:', test_perp)

		if args.ppl:
			if args.fv1:
				sentences = gensim2.models.word2vec.LineSentence(args.ppl)
				self.fv1model.build_vocab(sentences, update=True)

			test_data, test_words = self.load_text_data(args.ppl)
			test_perp = self.evaluate(test_data, test_words)
			print('test perplexity:', test_perp)

		if args.nbest:
			self.nbest(args.nbest)

	def FV(self, wrd, cache):
		return self.fv[wrd]

	def load_text_data(self, filename, update_vocab=False):
		"""Loads text to ints, update vocab if text is training"""
		words = open(filename).read().replace('\n', ' '+EOS+' ').strip().split()
		dataset = np.ndarray((len(words),), dtype=np.int32)
		if not EOS in self.vocab and update_vocab:
			self.vocab[EOS] = len(self.vocab)
		if not UNK in self.vocab and update_vocab:
			self.vocab[UNK] = len(self.vocab)
		for i, word in enumerate(words):
			if word not in self.vocab:
				if update_vocab:
					self.vocab[word] = len(self.vocab)
				else:
					word = UNK
					words[i] = UNK
			dataset[i] = self.vocab[word]
		return dataset, words


	def evaluate(self, dataset, dataset_text, fname=None):
		""""Evaluate net on input dataset 
			return perplexity"""
		evaluator = self.model.copy()
		evaluator.predictor.reset_state()  # initialize state
		evaluator.predictor.train = False  # dropout does nothing

		if args.fv or args.fv1:
			if args.fv1:
				if args.fv_type == 0: #update only UNK words
					for wrd in dataset_text:
						if wrd not in self.fv:
							if wrd in self.fv1model.vocab.keys():
								self.fv[wrd] = self.fv1model[wrd]
							else:
								self.fv[wrd] = self.fv[UNK]

				if args.fv_type == 1: # update all words in valid
					for wrd in dataset_text:
						if wrd in self.fv1model.vocab.keys():
							self.fv[wrd] = self.fv1model[wrd]
						else:
							self.fv[wrd] = self.fv[UNK]

				if args.fv_type == 2: # update all words in train+valid
					for k in self.fv1model.vocab.keys():
						self.fv[k] = self.fv1model[k]
					for wrd in dataset_text:
						if wrd not in self.fv1model.vocab.keys():
							self.fv[wrd] = self.fv[UNK]

		sum_log_perp = 0
		for i in six.moves.range(dataset.size - 1):
			cache = dataset_text[max(0, i-self.cache_len) : i + 1]
			dx = self.make_fv(dataset_text[i], cache)
			x = chainer.Variable(xp.asarray([dx]), volatile="on")
			t = chainer.Variable(xp.asarray(dataset[i + 1 : i + 2]), volatile='on')
			loss = evaluator(x, t)
			sum_log_perp += loss.data
		return math.exp(float(sum_log_perp) / (dataset.size - 1))


	def load_nbest_data(self, filename):
		"""Helper to load nbestlist to ints"""
		for line in open(filename):
			words = line.strip().split() + [EOS]
			if len(words) == 1:
				words = [UNK, EOS]
			dataset = np.ndarray((len(words),), dtype=np.int32)
			for i, word in enumerate(words):
				if word not in self.vocab:
					word = UNK
					words[i] = UNK
				dataset[i] = self.vocab[word]
			yield dataset, words


	def nbest(self, filename):
		"""Print logprog computed for every line of input file"""

		evaluator = self.model.copy()  # to use different state
		evaluator.predictor.reset_state()  # initialize state

		for dataset, dataset_text in self.load_nbest_data(filename):

			# update
			if args.fv or args.fv1:
				if args.fv1:
					if args.fv_type == 0:  # update only UNK words
						for wrd in dataset_text:
							if wrd not in self.fv:
								if wrd in self.fv1model.vocab.keys():
									self.fv[wrd] = self.fv1model[wrd]
								else:
									self.fv[wrd] = self.fv[UNK]

					if args.fv_type == 1: # update all words in valid
						for wrd in dataset_text:
							if wrd in self.fv1model.vocab.keys():
								self.fv[wrd] = self.fv1model[wrd]
							else:
								self.fv[wrd] = self.fv[UNK]

					if args.fv_type == 2: # update all words in train+valid
						for k in self.fv1model.vocab.keys():
							self.fv[k] = self.fv1model[k]
						for wrd in dataset_text:
							if wrd not in self.fv1model.vocab.keys():
								self.fv[wrd] = self.fv[UNK]


			sum_log_perp = 0
			for i in six.moves.range(dataset.size - 1):
				cache = dataset_text[max(0, i-self.cache_len) : i + 1]
				dx = self.make_fv(dataset_text[i], cache)
				x = chainer.Variable(xp.asarray([dx]), volatile="on")
				t = chainer.Variable(xp.asarray(dataset[i + 1 : i + 2]), volatile='on')
				loss = evaluator(x, t)
				sum_log_perp += loss.data
			evaluator.predictor.reset_state()
			print(-sum_log_perp)



	def init_net(self, config, randomize=True):
		"""Prepare RNNLM model, defined in net.py"""
		if config["num_layers"] == 1:
			self.lm = net.RNNLM_FV_1layer(self.input_size, config["hidden_size"], len(self.vocab), ratio=config["keep_prob"], train=True)
		else:
			self.lm = net.RNNLM_FV_2layer(self.input_size, config["hidden_size"], len(self.vocab), ratio=config["keep_prob"], train=True)
		self.model = L.Classifier(self.lm)
		self.model.compute_accuracy = False  # we only want the perplexity
		if randomize:
			for param in self.model.params():
				param.data[:] = np.random.uniform(-config["init_scale"], config["init_scale"], param.data.shape)

		# Setup optimizer
		self.optimizer = optimizers.SGD(lr=config["learning_rate"])
		self.optimizer.setup(self.model)
		self.optimizer.add_hook(chainer.optimizer.GradientClipping(config["max_grad_norm"]))


	def train(self, train_data, train_words, valid_data, valid_words):
		# Learning loop
		whole_len = train_data.shape[0]
		jump = whole_len // self.config["batch_size"]
		cur_log_perp = xp.zeros(())
		epoch = 0
		start_at = time.time()
		cur_at = start_at
		accum_loss = 0
		batch_idxs = list(range(self.config["batch_size"]))
		print('Going to train {} words'.format(jump * self.config["max_max_epoch"] * self.config["batch_size"]))
		PRINT_POINT = (jump // 10)

		# set unk words
		for wrd in train_words + valid_words:
			if wrd not in self.fv:
				self.fv[wrd] = self.fv[UNK]

		dx = [[]] * len(batch_idxs)
		for i in six.moves.range(jump * self.config["max_max_epoch"]):

			for j in batch_idxs:
				idx = (jump * j + i) % whole_len
				cache = train_words[max(0, idx-self.cache_len) : idx + 1]
				dx[j] = self.make_fv(train_words[idx], cache)

			x = chainer.Variable(xp.asarray(dx))
			t = chainer.Variable(xp.asarray(
				[train_data[(jump * j + i + 1) % whole_len] for j in batch_idxs]))

			loss_i = self.model(x, t)
			accum_loss += loss_i
			cur_log_perp += loss_i.data

			if (i + 1) % self.config["num_steps"] == 0:  # Run truncated BPTT
				self.model.zerograds()
				accum_loss.backward()
				accum_loss.unchain_backward()  # truncate
				accum_loss = 0
				self.optimizer.update()

			if (i + 1) % PRINT_POINT == 0:
				now = time.time()
				throuput = PRINT_POINT * self.config["batch_size"] / (now - cur_at)
				perp = math.exp(float(cur_log_perp) / PRINT_POINT)
				print('iter {:.1f} training perplexity: {:.2f} ({:.2f} words/sec)'.format((i + 1) / jump, perp, throuput))
				cur_at = now
				cur_log_perp.fill(0)
				sys.stdout.flush()

			if (i + 1) % jump == 0:
				# end of epoch
				epoch += 1
				now = time.time()
				if self.args.fv1:
					sentences = gensim2.models.word2vec.LineSentence(self.args.valid)
					self.fv1model.build_vocab(sentences, update=True)

				perp = self.evaluate(valid_data, valid_words)
				print('epoch {} validation perplexity: {:.2f}'.format(epoch, perp))
				cur_at += time.time() - now  # skip time of evaluation

				self.save(self.save_net+"."+str(epoch)+".lstm")

				if epoch >= self.config["max_epoch"]:
					self.optimizer.lr *= self.config["lr_decay"]
					print('learning rate =', self.optimizer.lr)
				sys.stdout.flush()



if __name__ == "__main__":

	DESCRIPTION = """
		Recurrent neural network based statistical language modelling toolkit
		(based on LSTM neural networks)
		Implemented by Daniel Soutner,
		New Technologies for the Information Society,
		University of West Bohemia, Plzen, Czechia
		dsoutner@kky.zcu.cz, 2016
		"""

	parser = argparse.ArgumentParser()
	parser.add_argument('--initmodel', '-m', default='',
						help='Initialize the model from given file')
	parser.add_argument('--resume', '-r', action="store_true",
						help='Resume the training of the model')
	parser.add_argument('--char', action="store_true",
						help='Swith to char LM')
	parser.add_argument('--save-net', dest='save_net', metavar="FILE", default=None,
						help='Computes PPL of net on text file (if we train, do that after training)')

	parser.add_argument('--gpu', '-g', default=0, type=int,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--cpu', action="store_true",
						help='force to run on CPU')

	parser.add_argument('--random-seed', default=None, type=int, dest="random_seed",
						help='Set random seed')

	parser.add_argument('--train', default=None, metavar="FILE",
						help='Train text file')
	parser.add_argument('--valid', default=None, metavar="FILE",
						help='Valid text file')
	parser.add_argument('--test', default=None, metavar="FILE",
						help='Test text file')
	parser.add_argument('--ppl', metavar="FILE", default=None,
						help='Computes PPL of net on text file (if we train, do that after training)')
	parser.add_argument('--nbest', metavar="FILE", default=None,
						help='Computes logprobs on nbest file')
	parser.add_argument('--ngram', metavar="FILE weight", default=None, nargs=2,
						help='ARPA n-gram model with interpoalting weight as second parameter')

	parser.add_argument('--fv', metavar="FILE", default=None,
						help='cPickled python dictionary with feature vectors for every word in vocab')
	parser.add_argument('--fv+', metavar="FILE", default=None, dest="fv1",
						help='cPickled python dictionary with feature vectors for every word in vocab')
	parser.add_argument('--fv-type', metavar="FILE", dest="fv_type",
						type=int, choices=[0,1,2], default=0,
						help='type == 0: update only UNK words, '
								'type == 1: # update all words in valid, '
								'type == 2: # update all words in train+valid')

	parser.add_argument('--num-layers', type=int, default=None, choices=[1,2,3],
						help='Number of LSTM layers (1 or 2)')
	parser.add_argument('--hidden-size', type=int, default=None,
						help='Size of hidden layer')
	parser.add_argument('--config', type=str, default="small", choices="tiny small medium large optimal".split(),
						help='Basic configuration')

	args = parser.parse_args()
	
	if args.gpu >= 0 and not args.cpu:
		xp = cuda.cupy
	else:
		xp = np

	xp.random.seed(args.random_seed)

	# if no args are passed
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()

	if args.fv or args.fv1:
		lstmlm = LSTMLM_FV(args)
	else:
		lstmlm = LSTMLM(args)
