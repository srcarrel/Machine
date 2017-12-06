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

    def __init__(self, loss):
        self.loss_fn = loss

    @abstractmethod
    def fit(self, train_x, train_y, valid_x=None, valid_y=None, **kwa):
        """
        Fit the model to the data provided, using the validation set if present.
        """
        pass

    @abstractmethod
    def predict(self, data):
        """
        Create a prediction using the model.
        """
        pass

    def loss(self, y_true, y_pred):
        """
        Return the calculated loss.
        """
        return self.loss_fn(y_true, y_pred)
