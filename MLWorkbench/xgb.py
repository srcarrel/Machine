"""
The XGBoost Model for use in the machine learning workbench.

Objects:
    Xgb: The XGBoost Model.
"""

import xgboost as xgb

from MLWorkbench.model import Model

class Xgb(Model):
    """
    The XGBoost Model.
    """

    def __init__(self, params, loss, n_iter=400):
        super().__init__(loss)
        self.params = params
        self.n_iter = n_iter
        self.model = None
        self.feature_names = None

    def fit(self, train_x, train_y, valid_x=None, valid_y=None, **kwa):
        """
        Fit the XGBoost Model to the train data, using valid as the validation set.
        """
        params = self.params.copy()

        self.feature_names = kwa.get('feature_names', None)

        dtrain = xgb.DMatrix(train_x, label=train_y, feature_names=self.feature_names)

        if valid_x is None:
            watchlist = [(dtrain, 'train')]
        else:
            dvalid = xgb.DMatrix(valid_x, label=valid_y, feature_names=self.feature_names)
            watchlist = [(dtrain, 'train'), (dvalid, 'validation')]

        self.model = xgb.train(params, dtrain, self.n_iter, watchlist,
                               verbose_eval=50, early_stopping_rounds=100)

    def predict(self, data):
        """
        Construct a prediction of data based on the trained model.
        """
        return self.model.predict(xgb.DMatrix(data, feature_names=self.feature_names))
