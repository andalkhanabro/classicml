"""
classicml.gnb 

A simple implementaton of a Gaussian Naive Bayes classifer. 

Author: Andal Abro

"""

import numpy as np

from .naivebayes import NaiveBayes
from .utils import Gaussian

class GaussianNaiveBayes(NaiveBayes):

    def __init__(self):
        self.class_priors = None 
        self.conditional_feature_gaussians = None 
        self.data_labels = None

    
    def fit(self, X, y):

        data_labels = np.unique(y)
        self.data_labels = data_labels
        self.class_priors = self.compute_class_priors(y) # priors set from data before estimation 

        # Likelihood estimation using Gaussian distributions which are conditionally independent, given the class
        # For i classes and j features, we need to evaluate ij distributions, and 2ij parameters (assuming independence and no shared covariance)

        conditional_feature_gaussians = {} # format is (class, feature_idx) -> Gaussian(mean, variance)

        for label in data_labels:
            
            X_rows_per_c = X[y == label]

            # Evaluations of MLE estimates

            feature_means_per_c = np.mean(X_rows_per_c, axis=0)             # Collapses columns into their means as a list [mean_1, mean_2, mean_3..]
            feature_variances_per_c = np.var(X_rows_per_c, axis=0)          # Collapses columns into their variances as a list [var_1, var_2, var_3..]

            # Add per class, per feature distributions to the hashmap defined earlier 

            for feature_idx in range(len(feature_means_per_c)): # DETAIL: feature index is its characterisation, not the name bec its numpy rn so 0 means X_0 (first feature)

                conditional_feature_gaussians[(label, feature_idx)] = Gaussian(feature_means_per_c[feature_idx], feature_variances_per_c[feature_idx])
            
        self.conditional_feature_gaussians = conditional_feature_gaussians

            # After fitting we have the class priors, and the feature gaussians evaluated from the training data as the model


    def single_predict(self, feature_vector):

        class_scores = {}

        for label in self.data_labels:

            # DETAIL: Compute log(P(Y=y)) + sum over log(Gaussian(mean_feature_idx, var_feature_idx)) over feature indices 

            log_class_prior = np.log(self.class_priors[label])
            log_likelihood_per_label = 0

            for feature_idx in range(len(feature_vector)):

                distribution = self.conditional_feature_gaussians[(label, feature_idx)]    # Get the relevant distribution G for class y_j, feature X_i
                value = np.log(distribution.PDF(feature_vector[feature_idx])) # this is G(mean, var, X=x) #TODO: can omit exponentiation in log-space? 

                log_likelihood_per_label += value

            class_scores[label] = log_class_prior + log_likelihood_per_label
        
        # return class with max score 

        return max(class_scores, key=class_scores.get) # max (unnormalised) score [proportional to probability]
    
    def predict(self, X):

        #TODO: implement for batch prediction (several input vectors!)
        
        pass 
    
    
    def compute_class_priors(self, y):

        #NOTE: this will be slower than NumPy operations, because the logic is displayed so the operations are not vectorised. 

        n_samples = len(y)
        counts = {}  # a hashmap of distinct class -> counts in observed data 

        for class_ob in y:
            if class_ob in counts:
                counts[class_ob] += 1
            else:
                counts[class_ob] = 1

        normalised_class_priors = {label: label_count/n_samples for label, label_count in counts.items()}

        return normalised_class_priors # e.g {a: 0.3, b: 0.5, c : 0.3} (this is our prior belief of a random class prediction, ignoring X. This is P(Y))
        
# TODO: Implement log_PDF as a helper instead, exponentiation is not needed since we take the log anyway?









