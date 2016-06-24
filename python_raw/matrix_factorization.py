'''http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/

to do:
1. make class like structure
2. regularization
3. adagrad 
'''
import numpy as np 
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
import math

df = pd.read_csv("../BX-CSV-Dump/Bx-Book-Ratings.csv", sep=';')
print "cleaning data"
df[['User-ID']] = df[['User-ID']].astype(str)
X = df[['User-ID','ISBN']].values
y = df[['Book-Rating']].values
y.shape = (y.shape[0],)
e_uid = LabelEncoder()
X[:,0] = e_uid.fit_transform(X[:,0])
e_isbn = LabelEncoder()
X[:,1] = e_isbn.fit_transform(X[:,1])
dimU = np.max(X[:,0]) + 1
dimI = np.max(X[:,1]) + 1

# learning matrix stuff 
print "starting training"
epoch = 25
alpha = 0.001
k = 5
p = np.random.rand(dimU, k)
q = np.random.rand(dimI, k)
def gradient_descent(p, q, X, y, alpha=0.005):
	count = 0.0
	total_error = 0.0
	for i, (entry, score) in enumerate(zip(X, y)):
		# if i % 5000 == 0:
		# 	print "%d / %d" % (i, X.shape[0])
		p_entry = p[entry[0]]
		q_entry = q[entry[1]]
		r = p_entry.dot(q_entry)
		# print r
		e = score - r # still need to square this
		p_entry += 2 * alpha * e * q_entry
		q_entry += 2 * alpha * e * p_entry
		count += 1
		total_error += e * e
	return total_error / count

for i in range(epoch):
	curr_alpha = alpha - (alpha - 0.00000001) * i / epoch
	error = gradient_descent(p, q, X, y, alpha=curr_alpha)
	print "iter: %d, error: %f" % (i, error)

