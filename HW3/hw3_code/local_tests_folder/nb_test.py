import numpy as np

class NB_Test():
	def __init__(self):

		# Sample data
		self.x_train = np.array([[1, 4],
						       [0, 6],
						       [3, 2],
						       [3, 1],
						       [4, 0]])
		self.y_train = np.array([0, 0, 1, 2, 2]) 

		self.list_of_labels = [np.array([[1, 4],
							        [0, 6]]), 
								np.array([[3, 2]]), 
								np.array([[3, 1],
							        [4, 0]])]








		
