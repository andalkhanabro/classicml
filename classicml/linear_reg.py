"""
classicml.linear_reg.py 

A simple linear regression model implementation. 

NOTE: This defaults to numerical solutions for optimal parameters if feature space is large (or degenerate). 

"""

import numpy as np
from .regression import Regressor

class LinearRegressor(Regressor):

    def __init__(self):
        self.feature_weights = None 
        self.intercept = None

    def fit(self, X, y):

        # attempt an analytical sol if condition number makes analytical sol viable 
        # TODO: check for cond number 
        # TODO: check for undetermined systems 

        # if its not a very big dataset, then, augment a one-vec and absorb b 

        n_columns = X.shape[1] 
        n_samples = X.shape[0]
        one_vec = np.ones((n_samples, 1))
        X_aug = np.hstack([one_vec, X]) # where the first col represents vals for w_0 (absorbed intercept)

        # optimal parameters for min w is just the normal eq solution 

        try:

            weights = (np.linalg.inv(X_aug.T @ X_aug)) @ (X_aug.T @ y)
            self.intercept = weights[0]
            self.feature_weights = weights[1:]
        
        except np.linalg.LinAlgError:

            print("The Design matrix is not invertible.")

            #TODO: implement GD/SVD solution later 


    def single_predict(self, feature_vector):

        return np.array(feature_vector) @ self.feature_weights + self.intercept 

    def predict(self, X):
        pass 
















    



    