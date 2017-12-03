"""
An abstract class used to describe the interface expected.

Object:
    Model: A simple interface for machine learning algorithms.
"""

from abc import ABC, abstractmethod

class Model(ABC):
    """
    The Model represents a simple interface for general machine learning algorithsm.
    """

    @abstractmethod
    def fit(self, x_train, y_train, x_eval=None, y_eval=None, **kwa):
        """
        Fit the model to the data provided, using the validation set if present.
        """
        pass

    @abstractmethod
    def predict(self, x_val):
        """
        Create a prediction using the model.
        """
        pass
