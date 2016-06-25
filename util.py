import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import math

def loadBookRating():
	df = pd.read_csv("../BX-CSV-Dump/Bx-Book-Ratings.csv", sep=';')
	print "loading data"
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
	return X, y, dimU, dimI