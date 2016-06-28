'''http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/

to do:
- regularization
- implement every optimization thing here: http://sebastianruder.com/optimizing-gradient-descent/index.html#momentum
'''
import numpy as np 
import sys
sys.path.append("../")
from util import loadBookRating

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
				error = self.nesterov_sgd(entry, score, curr_alpha)

	def momentum_sgd(self, entry, score, alpha, momentum=0.9):
		r, p_entry, q_entry = self.predict(entry[0], entry[1])
		e = score - r
		p_grad = 2 * e * q_entry
		q_grad = 2 * e * p_entry
		if self.last_p_grad is not None:
			p_grad = momentum * self.last_p_grad + alpha * p_grad
			q_grad = momentum * self.last_q_grad + alpha * q_grad
		p_entry += alpha * p_grad
		q_entry += alpha * q_grad
		self.last_p_grad = p_grad
		self.last_q_grad = q_grad
		return e

	def nesterov_sgd(self, entry, score, alpha, momentum=0.9):
		p_entry = self.p[entry[0]]
		q_entry = self.q[entry[1]]
		if self.last_p_grad is not None:
			r = (p_entry + momentum * self.last_p_grad).dot(q_entry + momentum * self.last_q_grad)
		else:
			r = p_entry.dot(q_entry)
		e = score - r
		p_grad = 2 * e * q_entry
		q_grad = 2 * e * p_entry
		if self.last_p_grad is not None:
			p_grad = momentum * self.last_p_grad + alpha * p_grad
			q_grad = momentum * self.last_q_grad + alpha * q_grad
		p_entry += alpha * p_grad
		q_entry += alpha * q_grad
		self.last_p_grad = p_grad
		self.last_q_grad = q_grad
		return e

X, y, dimU, dimI = loadBookRating()
mf = MF(X, y, dimU, dimI, k=6)
mf.train(epoch=25, alpha=0.01)
