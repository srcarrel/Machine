"""
The main workhorse of the machine learning workbench, providing the train_model method.
"""

import numpy as np

from sklearn.model_selection import StratifiedKFold

from MLWorkbench.dataset import View, Dataset
from MLWorkbench.model import Model

def train_model(model, train_ds, target_view, test_ds, **kwa):
    """
    The machine learning workbench training method.

    Args:
        model (Model): The model to be trained.

        train_ds (Dataset): The dataset used for training.
        target_view (View): The view which comprises the target for the training dataset.

        test_ds (Dataset): The test dataset.

    **kwa args:
        n_splits: The number of times to create the stratified folds (default 1).
        n_folds: The number of folds to use in training (default 5).
        n_bags: The number of times to run each fold (default 1).

        eval_fn: A function which will be used to evaluate the out of fold predictions
                 made on the training set. This is only used for output (presently).

    Output: train_p, test_p
        The train_model method returns two multi-dimensional numpy arrays:

            train_p: which has dimensions (n_samples, n_splits, n_bags) and contains
                     the out of fold predictions on the training set,
            test_p:  which has dimensions (n_samples, n_splits, n_folds, n_bags) and
                     contains the predictions computed at each level of training.

        Note also that n_samples for train_p is the length of the first dimension in
        the training dataset and n_samples for test_p is the length of the first dimension
        in the test dataset.
    """
    assert isinstance(model, Model)
    assert isinstance(train_ds, Dataset)
    assert isinstance(target_view, View)
    assert isinstance(test_ds, Dataset)

    n_splits = kwa.get('n_splits', 1)
    n_folds = kwa.get('n_folds', 5)
    n_bags = kwa.get('n_bags', 1)

    train_x = train_ds.get_values()
    train_y = target_view.view_data
    train_p = np.zeros((train_x.shape[0], n_splits, n_bags))

    test_x = test_ds.get_values()
    test_p = np.zeros((test_x.shape[0], n_splits, n_folds, n_bags))

    for split in range(n_splits):
        print()
        print("Training split %d of %d..." % (split + 1, n_splits))

        train_p[:, split, :], test_p[:, split, :, :] \
                = train_folds(model=model,
                              train_data=(train_x, train_y),
                              test_data=(test_x,), **kwa)
    return train_p, test_p

def train_folds(model, train_data, test_data, **kwa):
    """
    Construct and train folds, performing bagged prediction / evaluation on each fold.
    """
    n_folds = kwa.get('n_folds', 5)
    n_bags = kwa.get('n_bags', 1)
    k_fold = StratifiedKFold(n_folds, shuffle=True)

    train_p = np.zeros((train_data[0].shape[0], n_bags))
    test_p = np.zeros((test_data[0].shape[0], n_folds, n_bags))

    for fold, (train_idx, valid_idx) in enumerate(k_fold.split(train_data[0], train_data[1])):
        print()
        print("  Fold %d of %d..." % (fold + 1, n_folds))

        train_p[valid_idx, :], test_p[:, fold, :] \
                = train_bags(model=model,
                             train_data=(train_data[0][train_idx], train_data[1][train_idx]),
                             valid_data=(train_data[0][valid_idx], train_data[1][valid_idx]),
                             test_data=test_data, **kwa)
    return train_p, test_p

def train_bags(model, train_data, valid_data, test_data, **kwa):
    """
    Train a fold n_bags times with OOF validation and prediction on test.
    """
    n_bags = kwa.get('n_bags', 1)

    train_p = np.zeros((valid_data[0].shape[0], n_bags))
    test_p = np.zeros((test_data[0].shape[0], n_bags))

    for bag in range(n_bags):
        print()
        print("    Training model %d of %d..." % (bag + 1, n_bags))

        model.fit(train_x=train_data[0],
                  train_y=train_data[1],
                  valid_x=valid_data[0],
                  valid_y=valid_data[1],
                  kwa=kwa)

        train_p[:, bag] = model.predict(valid_data[0])
        test_p[:, bag] = model.predict(test_data[0])

        if kwa.get('eval_fn', None) is not None:
            print("    OOF Evaluation:",
                  kwa.get('eval_fn', None)(valid_data[1], train_p[:, bag]))

    return train_p, test_p
