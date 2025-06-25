"""
classicml.utils 

These are model agnostic methods that are required for ML workflows (e.g splitting [ADD Later]).

Author: Andal Abro

"""

import numpy as np
from typing import Optional, Tuple

from constants import DEFAULT_SPLITTING_RATIO


def split_dataset(
        
        X: np.ndarray, 
        y: np.ndarray, 
        split_ratio: float = DEFAULT_SPLITTING_RATIO, 
        random_state: Optional[int] = None                                  #TODO: use this later  

        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """docstring for documentation here"""

    # shuffle but keep X and y correspondence! 

    n_samples = len(X)

    data_indices = np.arange(n_samples) # indices initialised from 0 to n_samples - 1
    np.random.shuffle(data_indices)  # row indices shuffled                 TODO: stratification is not implemented yet 
    
    #                                                                       TODO: reproducibility feature not implemented yet using RNGs 

    X_shuffled = X[data_indices]
    y_shuffled = y[data_indices]

    # now need to portion into X_train, X_test, y_train, and y_test 

    # X_shuffled has dimensions [n_samples, n_features]
    # y_shuffled has dimensions (n_samples, 1? for now)                     TODO: multi-dimensional y not supported yet 

    # dividing the data by reading first (1-TEST_RATIO)% indices of X and y 

    n_training = int((1-split_ratio) * n_samples)  # so 0.75 *100 means 75 rows for training, 25 for testing out of 100. DETAIL: casted to int for slicing 

    X_train = X_shuffled[0: n_training]
    X_test = X_shuffled[n_training: ] 

    y_train = y_shuffled[0: n_training]
    y_test = y_shuffled[n_training: 0] 

    assert len(X_train) + len(X_test) == len(X), "Train-test split does not preserve total sample size."
    assert len(y_train) + len(y_test) == len(y), "Train-test split does not preserve label count."

    return (X_train, y_train, X_test, y_test)       
    

