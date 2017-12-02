"""
The basic data manipulation objects.

Objects:
    View: A wrapper for numpy arrays which conceptually is a view into some dataset.
    Dataset: A collection of Views into the same dataset.

Notes:
    This module contains the constant CACHE_DIR which, by default, points to /cache
    inside the main project directory.
"""

import os
import pickle

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache')

class View(object):
    """
    The View object represents the fundamental datatype inside of the MLWorkbench framework.
    """

    def __init__(self, view_name, view_data, feature_names):
        """
        Construct a view from a given dataset with columns names.

        Note that this class is essentially a wrapper for numpy arrays and/or
        pandas dataframes.

        Args:
            view_name (str): The name of this view.
            view_data (array_like): The (usually) numpy array which contains the data for
                                    this view
            feature_names: A list of strings which represent the columns names of the
                           view_data, in order. Note that the feature names should all
                           be unique. Passing in a list of feature names with repetition
                           will result in a value error.
        """
        self.view_name = view_name
        self.view_data = view_data
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Feature names must be unique.")
        self.feature_names = feature_names

    @classmethod
    def from_df(cls, view_name, view_df):
        """
        Construct a view from a pandas dataframe.
        """
        return cls(view_name, view_df.values, list(view_df.columns))

    @classmethod
    def from_cache(cls, ds_name, view_name):
        """
        Load a view of the dataset ds_name from the cache directory.
        """
        return cls(view_name,
                   np.load("%s/%s-%s.npy" % (CACHE_DIR, view_name, ds_name)),
                   pickle.load(open("%s/%s-features.pickle" % (CACHE_DIR, view_name), "rb")))

    def to_cache(self, ds_name):
        """
        Save this view of the dataset ds_name to the cache directory.
        """
        pickle.dump(self.feature_names,
                    open("%s/%s-features.pickle" % (CACHE_DIR, self.view_name), "wb"))
        np.save("%s/%s-%s.npy" % (CACHE_DIR, self.view_name, ds_name), self.view_data)

    def get_df(self, index=None):
        """
        Return this view as a pandas dataframe.

        Args:
            index (array_like): An index for this view as a dataframe.
        """
        return pd.DataFrame(self.view_data,
                            columns=self.feature_names,
                            index=index)


class Dataset(object):
    """
    The Dataset object represents a single named dataset with multiple views into it.
    """

    def __init__(self, ds_name, **views):
        """
        Construct a Dataset object from a collection of views into named dataset.

        This is object is, of course, mostly a wrapper for a collection of views into
        a particular named data set, stored as a dictionary. Included are a few helper
        functions for standard operations on a Dataset.

        Notes:
            The Dataset object does not have a to_cache method or any other way of saving
            it to disk. This is intentional since a Dataset should be thought of as a
            read only object which represents static data, not something to be created.
            If you wish to create new views of a given dataset, it is better to construct
            the views manually using the View class.

        Args:
            ds_name (str): The name of the dataset.
            views (dict): This is the dictionary containing the views into the dataset.
                          It is assumed that the keys in the dictionary correspond to
                          the names of the views.
        """
        self.ds_name = ds_name
        self.views = views

    def __getitem__(self, key):
        """
        Get one of the views from the Dataset object.
        """
        return self.views[key]

    @classmethod
    def from_cache(cls, ds_name, views):
        """
        Construct a Dataset from a list of views loaded from the cache directory.

        Args:
            ds_name (str): The name of the dataset to load.
            views (list of str): The list of named views to load from the cache directory.
        """
        return cls(ds_name,
                   **{view_name: View.from_cache(ds_name, view_name) for view_name in views})

    def get_values(self):
        """
        Return the combined values of the views into this dataset.
        """
        val_list = [view.view_data for view in self.views]
        return np.hstack(val_list)

    def get_feature_names(self):
        """
        Return the combined list of feature names of the views into this dataset.
        """
        return [view.feature_names for view in self.views]
