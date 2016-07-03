'''http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/

to do:
- regularization
- implement every optimization thing here: http://sebastianruder.com/optimizing-gradient-descent/index.html#momentum
http://cs231n.github.io/neural-networks-3/#ada
'''
import numpy as np 
import sys
sys.path.append("../")
from util import loadBookRating
import random

class MF:
	def __init__(self, X, y, dimU, dimI, k=5):
		self.X = X
		self.y = y
		self.k = k
		self.p = np.random.rand(dimU, k)
		self.q = np.random.rand(dimI, k)
		self.last_p_grad = None
		self.last_p_grad = None

	def predict(self, userId, itemId):
		p_entry = self.p[userId]
		q_entry = self.q[itemId]
		return p_entry.dot(q_entry), p_entry, q_entry

	def currentError(self):
		count = 0.0
		total_error = 0.0
		for i, (entry, score) in enumerate(zip(self.X, self.y)):
			e = score - self.predict(entry[0], entry[1])[0]
			total_error += e * e
			count += 1
		return total_error / count

	def train(self, epoch=10, alpha=0.001):
		for e in range(epoch):
			print "iter: %d, error: %f" % (e, self.currentError())
			curr_alpha = alpha - (alpha - 0.00000001) * e / epoch
			for i, (entry, score) in enumerate(zip(self.X, self.y)):
				# error = self.sgd(entry, score, curr_alpha)
				error = self.als(entry, score, curr_alpha)
				# error = self.momentum_sgd(entry, score, curr_alpha)
				# error = self.nesterov_sgd(entry, score, curr_alpha)
				# error = self.adagrad_sgd(entry, score)
				# error = self.rmsprop(entry, score)

	def sgd(self, entry, score, alpha):
		r, p_entry, q_entry = self.predict(entry[0], entry[1])
		e = score - r
		p_grad = 2 * e * q_entry
		q_grad = 2 * e * p_entry
		p_entry += alpha * p_grad
		q_entry += alpha * q_grad
		self.last_p_grad = p_grad
		self.last_q_grad = q_grad
		return e

	def als(self, entry, score, alpha):
		r, p_entry, q_entry = self.predict(entry[0], entry[1])
		e = score - r
		if random.random() >= 0.5:
			p_grad = 2 * e * q_entry
			p_entry += alpha * p_grad
			self.last_p_grad = p_grad
		else:
			q_grad = 2 * e * p_entry
			q_entry += alpha * q_grad
			self.last_q_grad = q_grad
		return e

	def momentum_sgd(self, entry, score, alpha, momentum=0.9):
		if not hasattr(self, 'm_p'):
			self.m_p = np.zeros(self.p.shape)
			self.m_q = np.zeros(self.q.shape)
		r, p_entry, q_entry = self.predict(entry[0], entry[1])
		e = score - r	
		p_grad = 2 * e * q_entry
		q_grad = 2 * e * p_entry
		p_grad = momentum * self.m_p[entry[0]] + alpha * p_grad
		q_grad = momentum * self.m_q[entry[1]] + alpha * q_grad
		p_entry += p_grad
		q_entry += q_grad
		self.m_p[entry[0]] = p_grad
		self.m_q[entry[1]] = q_grad
		return e

	def nesterov_sgd(self, entry, score, alpha, momentum=0.9):
		if not hasattr(self, 'm_p'):
			self.m_p = np.zeros(self.p.shape)
			self.m_q = np.zeros(self.q.shape)
		p_entry = self.p[entry[0]]
		q_entry = self.q[entry[1]]
		p_ahead = p_entry + momentum * self.m_p[entry[0]]
		q_ahead = q_entry + momentum * self.m_q[entry[1]]
		r = p_ahead.dot(q_ahead)
		e = score - r	
		p_grad = 2 * e * q_ahead
		q_grad = 2 * e * p_ahead
		p_grad = momentum * self.m_p[entry[0]] + alpha * p_grad
		q_grad = momentum * self.m_q[entry[1]] + alpha * q_grad
		p_entry += p_grad
		q_entry += q_grad
		self.m_p[entry[0]] = p_grad
		self.m_q[entry[1]] = q_grad
		return e

	def adagrad_sgd(self, entry, score, alpha=0.001):
		if not hasattr(self, 'G_p'):
			self.G_p = np.zeros(self.p.shape)
			self.G_q = np.zeros(self.q.shape)
		r, p_entry, q_entry = self.predict(entry[0], entry[1])
		e = score - r
		p_grad = 2 * e * q_entry
		q_grad = 2 * e * p_entry
		# print alpha * p_grad / (np.sqrt(self.G_p[entry[0]] + 0.1))
		p_entry += alpha * p_grad / (np.sqrt(self.G_p[entry[0]] + 0.1))
		q_entry += alpha * q_grad / (np.sqrt(self.G_q[entry[1]] + 0.1))
		self.G_p[entry[0]] += p_grad ** 2
		self.G_q[entry[1]] += q_grad ** 2
		return e

	def rmsprop(self, entry, score, alpha=0.001, decay_rate=0.9):
		if not hasattr(self, 'G_p'):
			self.G_p = np.zeros(self.p.shape)
			self.G_q = np.zeros(self.q.shape)
		r, p_entry, q_entry = self.predict(entry[0], entry[1])
		e = score - r
		p_grad = 2 * e * q_entry
		q_grad = 2 * e * p_entry
		# print alpha * p_grad / (np.sqrt(self.G_p[entry[0]] + 0.1))
		p_entry += alpha * p_grad / (np.sqrt(self.G_p[entry[0]] + 0.1))
		q_entry += alpha * q_grad / (np.sqrt(self.G_q[entry[1]] + 0.1))
		self.G_p[entry[0]] = decay_rate * self.G_p[entry[0]] + (1 - decay_rate) * p_grad ** 2
		self.G_q[entry[1]] = decay_rate * self.G_q[entry[1]] + (1 - decay_rate) * q_grad ** 2
		return e

X, y, dimU, dimI = loadBookRating()
mf = MF(X, y, dimU, dimI, k=6)
mf.train(epoch=25, alpha=0.001)
