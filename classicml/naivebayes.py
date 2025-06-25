
from classicml.base_model import Model

class NaiveBayes(Model):

    def __init__(self):
        #TODO: implement functionality
        pass 

    def fit(self, X, y):

        # X is a design matrix, so it has n_samples rows and n_features columns 

        # DETAIL: NB assumes condiitonal independence, given the class, so need to evaluate (4 * n_features + n_classes) parameters
        
        pass 

