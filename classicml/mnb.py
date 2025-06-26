"""
classicml.mnb 

A simple implementaton of a Multinomial Naive Bayes classifer. 

Author: Andal Abro

"""

import numpy as np

from .naivebayes import NaiveBayes

class MultinomialNaiveBayes(NaiveBayes):

    def __init__(self):
        self.class_priors = None 
        self.word_parameters = None
        self.data_labels = None
        self.vocab_size = None

    
    def fit(self, X, y):
        
        # X is a design matrix, where X_i is drawn from Multinomial with probability parameter vector (theta). 

        self.class_priors = self.compute_class_priors(y)
        self.vocab_size = X.shape[1]

        #DETAIL: 
        # MLE estimate is frequency of word (i.e count over total words in that class)

        # Meaning, get rows that correspond to spam: X1 X2 X3 X4 X5
        # then getting X1 once has probability all X1 counts, sum across rows and columns according to MLE 

        data_labels = np.unique(y)              # returns a list of unique labels (i.e spam, not spam)
        self.data_labels = data_labels

        parameters = {} # (label, feature_idx) -> p #DETAIL: p = (thetta_i,j) (word_i, under class_j)

        for label in data_labels:

            X_rows_per_label = X[y == label] 
            total_word_count = np.sum(X_rows_per_label)        # e.g all words that appear in a spam email (per class label)
            n_columns = X.shape[1]

            for feature_idx in range(n_columns): # i.e n columns 

                # find total number of times X_i appears across all training samples, given a class 

                frequency_of_word = np.sum(X_rows_per_label[:, feature_idx]) 

                cond_probability_of_word = (frequency_of_word +  1) \
                                           /(total_word_count + n_columns) # laplace smoothing to avoid 0 prob for
                                                                           # rare words 
                                                                           # where n_columns = |V| (vocabulary size)

                                                                           # TODO: allow dirchlet smoothing? 

                parameters[(label, feature_idx)] = cond_probability_of_word
    
        
        parameter_vals = list(parameters.values())
        assert np.all((parameter_vals >= 0) & parameter_vals <= 1), "parameters are not valid probabilities"

        self.word_parameters = parameters


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

        return normalised_class_priors
    

    def single_predict(self, feature_vector):

        assert(len(feature_vector) == self.vocab_size)

        # feature vector is a list of counts, with feature_vector[feature_idx] being the count of feature_vector(i)

        # compute probability of being in a single class by finding the prob of the vector acc to the multinomial distribution:

        scores_per_label = {}

        for label in self.data_labels:

            log_class_prior = np.log(self.class_priors[label])              # P(Y=y) avg prob of being spam, regardless of data 
            log_likelihood_per_label = 0 

            for feature_idx in range(len(feature_vector)):                  # we work in log space to prevent underflow 

                log_word_prob = np.log(self.word_parameters[(label, feature_idx)])
                count = feature_vector[feature_idx]
                log_likelihood_per_label += (count * log_word_prob)

            scores_per_label[label] = log_likelihood_per_label + log_class_prior

        return max(scores_per_label, key=scores_per_label.get)          # returns label with max posterior (unnormalised) probability 
    








    

                









