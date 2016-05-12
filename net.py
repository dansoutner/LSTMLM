import chainer
import chainer.functions as F
import chainer.links as L


class RNNLM_1layer(chainer.Chain):
	"""Recurrent neural net languabe model for penn tree bank corpus.
	This is an example of deep LSTM network for infinite length input.
	"""
	def __init__(self, n_vocab, n_units, ratio=1., train=True):
		super(RNNLM_1layer, self).__init__(
			embed=L.EmbedID(n_vocab, n_units),
			l1=L.LSTM(n_units, n_units),
			l2=L.Linear(n_units, n_vocab),
		)
		self.train = train
		self.ratio = 1. - ratio

	def reset_state(self):
		self.l1.reset_state()

	def __call__(self, x):
		h0 = self.embed(x)
		h1 = self.l1(F.dropout(h0, ratio=self.ratio, train=self.train))
		y = self.l2(F.dropout(h1, ratio=self.ratio, train=self.train))
		return y


class RNNLM_2layer(chainer.Chain):
	"""Recurrent neural net languabe model for penn tree bank corpus.
	This is an example of deep LSTM network for infinite length input.
	"""
	def __init__(self, n_vocab, n_units, ratio=1., train=True):
		super(RNNLM_2layer, self).__init__(
			embed=L.EmbedID(n_vocab, n_units),
			l1=L.LSTM(n_units, n_units),
			l2=L.LSTM(n_units, n_units),
			l3=L.Linear(n_units, n_vocab),
		)
		self.train = train
		self.ratio = 1. - ratio

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	def __call__(self, x):
		h0 = self.embed(x)
		h1 = self.l1(F.dropout(h0, ratio=self.ratio, train=self.train))
		h2 = self.l2(F.dropout(h1, ratio=self.ratio, train=self.train))
		y = self.l3(F.dropout(h2, ratio=self.ratio, train=self.train))
		return y


class RNNLM_3layer(chainer.Chain):
	"""Recurrent neural net languabe model for penn tree bank corpus.
	This is an example of deep LSTM network for infinite length input.
	"""
	def __init__(self, n_vocab, n_units, ratio=1., train=True):
		super(RNNLM_3layer, self).__init__(
			embed=L.EmbedID(n_vocab, n_units),
			l1=L.LSTM(n_units, n_units),
			l2=L.LSTM(n_units, n_units),
			l3=L.LSTM(n_units, n_units),
			l4=L.Linear(n_units, n_vocab),
		)
		self.train = train
		self.ratio = 1. - ratio

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()
		self.l3.reset_state()

	def __call__(self, x):
		h0 = self.embed(x)
		h1 = self.l1(F.dropout(h0, ratio=self.ratio, train=self.train))
		h2 = self.l2(F.dropout(h1, ratio=self.ratio, train=self.train))
		h3 = self.l3(F.dropout(h2, ratio=self.ratio, train=self.train))
		y = self.l4(F.dropout(h3, ratio=self.ratio, train=self.train))
		return y


class RNNLM_FV_1layer(chainer.Chain):
	"""Recurrent neural net languabe model for penn tree bank corpus.
	This is an example of deep LSTM network for infinite length input.
	"""
	def __init__(self, n_fv, n_units, n_vocab, ratio=1., train=True):
		super(RNNLM_FV_1layer, self).__init__(
			l1 = L.LSTM(n_fv, n_units),
			l2 = L.Linear(n_units, n_vocab),
		)
		self.train = train
		self.ratio = 1. - ratio

	def reset_state(self):
		self.l1.reset_state()

	def __call__(self, x):
		h1 = self.l1(F.dropout(x, ratio=self.ratio, train=self.train))
		y = self.l2(F.dropout(h1, ratio=self.ratio, train=self.train))
		return y


class RNNLM_FV_2layer(chainer.Chain):
	"""Recurrent neural net languabe model for penn tree bank corpus.
	This is an example of deep LSTM network for infinite length input.
	"""
	def __init__(self, n_fv, n_units, n_vocab, ratio=1., train=True):
		super(RNNLM_FV_2layer, self).__init__(
			l1 = L.LSTM(n_fv, n_units),
			l2 = L.LSTM(n_units, n_units),
			l3 = L.Linear(n_units, n_vocab),
		)
		self.train = train
		self.ratio = 1. - ratio

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	def __call__(self, x):
		h0 = self.l1(F.dropout(x, ratio=self.ratio, train=self.train))
		h1 = self.l2(F.dropout(h0, ratio=self.ratio, train=self.train))
		y = self.l3(F.dropout(h1, ratio=self.ratio, train=self.train))
		return y


class SoftmaxCrossEntropyLoss(chainer.Chain):
	def __init__(self, n_in, n_out):
		super(SoftmaxCrossEntropyLoss, self).__init__(
			W=L.Linear(n_in, n_out),
		)

	def __call__(self, x, t):
		return F.softmax_cross_entropy(self.W(x), t)


class HSMmodel(chainer.Chain):

	def __init__(self, n_vocab, n_units, tree, ratio=1., train=True):
		super(HSMmodel, self).__init__(
			embed=L.EmbedID(n_vocab, n_units),
			l1=L.LSTM(n_units, n_units),
			l2=L.LSTM(n_units, n_units),
			l3=L.Linear(n_units, n_units),
			#lossfun=SoftmaxCrossEntropyLoss(n_units, n_vocab),
			lossfun=L.BinaryHierarchicalSoftmax(n_units, tree),
		)
		self.train = train
		self.ratio = 1. - ratio
		self.y = None
		self.loss = None
		self.accuracy = None
		self.compute_accuracy = True

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	def predictor(self, x):
		h0 = self.embed(x)
		h1 = self.l1(F.dropout(h0, ratio=self.ratio, train=self.train))
		h2 = self.l2(F.dropout(h1, ratio=self.ratio, train=self.train))
		y = self.l3(F.dropout(h2, ratio=self.ratio, train=self.train))
		return y

	def __call__(self, x, t):
		self.y = self.predictor(x)
		self.loss = self.lossfun(self.y, t)
		#print(self.loss.data)
		if self.compute_accuracy:
			self.accuracy = chainer.functions.evaluation.accuracy(self.y, t)
		return self.loss
