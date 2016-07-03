'''
https://en.wikipedia.org/wiki/Collaborative_filtering#Memory-based

 the value of ratings user 'u' gives to item 'i' is calculated as an aggregation of some similar users' rating of the item

'''
import numpy as np 
import sys
sys.path.append("../")
from util import loadBookRating
from scipy.sparse import lil_matrix

class SimilarityRetrieval:
	def __init__(self, X, y, dimU, dimI):
		self.X = X
		self.y = y
		self.dimU = dimU
		self.dimI = dimI
		self._toSparseMatrix()

	def _toSparseMatrix(self):
		self.s = lil_matrix((self.dimU, self.dimI), dtype=np.int8)
		print "constructing sparse matrix"
		for i in range(self.X.shape[0]):
			if i % 100 == 0:
				print i
			entry = self.X[i]
			rows = np.array([entry[0]]).reshape(-1, 1)
			cols = np.array([entry[1]])
			self.s[rows, cols] += np.ones([rows.size, cols.size])

X, y, dimU, dimI = loadBookRating()
sr = SimilarityRetrieval(X, y, dimU, dimI)



