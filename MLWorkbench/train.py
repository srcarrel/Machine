"""
The main workhorse of the machine learning workbench, providing the train method.
"""

import numpy as np

from sklearn.model_selection import StratifiedKFold

from dataset import View, Dataset
from model import Model

def train(model, train_ds, target_view, test_ds, **kwa):
    """
    The machine learning workbench train method.

    Args:
        model (Model): The model to be trained.

        train_ds (Dataset): The dataset used for training.
        target_view (View): The view which comprises the target for the training dataset.

        test_ds (Dataset): The test dataset.

    **kwa args:
        n_splits: The number of times to create the stratified folds.
        n_folds: The number of folds to use in training.
        n_bags: The number of times to run each fold.
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
        print("Training split %d..." % split)

        k_fold = StratifiedKFold(n_folds, shuffle=True)

        for fold, (fold_train_idx, fold_eval_idx) in enumerate(k_fold.split(train_x, train_y)):
            print()
            print("  Fold %d..." % fold)

            fold_train_x = train_x[fold_train_idx]
            fold_train_y = train_y[fold_train_idx]

            fold_eval_x = train_x[fold_eval_idx]
            fold_eval_y = train_y[fold_eval_idx]

            for bag in range(n_bags):
                print("    Training model %d..." % bag)

                model.fit(train=(fold_train_x, fold_train_y),
                          val=(fold_eval_x, fold_eval_y),
                          feature_names=train_x.get_feature_names())

                train_p[fold_eval_idx, split, bag] = model.predict(fold_eval_x)

                test_p[:, split, fold, bag] = model.predict(test_x)

                print("    OOF Train Loss:",
                      model.loss(train_p[fold_eval_idx, split, bag], fold_train_y))
    return train_p, test_p
