from sklearn.utils import check_random_state
from functools import partial
from lime import lime_base
import os,sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

import lime_trace.explanation_trace as explanation
from lime_trace.utils import get_offset

import numpy as np
import scipy as sp
import pandas as pd
import sklearn
#from lime.lime_text import IndexedCharacters,TextDomainMapper,IndexedString

class TraceExplainer(object):
    """Explains text classifiers.
       Currently, we are using an exponential kernel on cosine distance, and
       restricting explanations to words that are present in documents."""

    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 #split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 random_state=None,
                 char_level=False):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True (bag of words), will perturb input data by removing
                all occurrences of individual words or characters.
                Explanations will be in terms of these words. Otherwise, will
                explain in terms of word-positions, so that a word may be
                important the first time it appears and unimportant the second.
                Only set to false if the classifier uses word order in some way
                (bigrams, etc), or if you set char_level=True.
            mask_string: String used to mask tokens or characters if bow=False
                if None, will be 'UNKWORDZ' if char_level=False, chr(0)
                otherwise.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            char_level: an boolean identifying that we treat each character
                as an independent occurence in the string
        """

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel_fn, verbose,
                                       random_state=self.random_state)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.mask_string = mask_string
        #self.split_expression = split_expression
        self.char_level = char_level

    def explain_instance(self,
                         mode,
                         this_row,
                         b_text_ids,
                         b_code_ids,
                         sorted_idx,
                         cve_desc_end_idx,
                         text_len,
                         code_lens,
                         classifier_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).

        Args:
            text_instance: raw text string to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a list of d strings and outputs a (d, k) numpy array with
                prediction probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        if mode == "textonly":
            data, yss, distances, sizes_cumsum = self.__data_labels_distances_textonly(
                this_row, b_text_ids, b_code_ids, cve_desc_end_idx, text_len, code_lens, classifier_fn, num_samples,
                distance_metric=distance_metric)
        elif mode == "both":
            data, yss, distances, sizes_cumsum = self.__data_labels_distances(
                this_row, b_text_ids, b_code_ids, cve_desc_end_idx, text_len, code_lens, classifier_fn, num_samples,
                distance_metric=distance_metric)
        elif mode == "codeonly":
            data, yss, distances, sizes_cumsum = self.__data_labels_distances_codeonly(
                this_row, b_text_ids, b_code_ids, sorted_idx, cve_desc_end_idx, text_len, code_lens, classifier_fn, num_samples,
                distance_metric=distance_metric)

        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation_trace(class_names=self.class_names,
                                          random_state=self.random_state)

        ret_exp.sizes_cumsum = sizes_cumsum
        ret_exp.score = {}
        ret_exp.local_pred = {}

        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def __data_labels_distances(self,
                                this_row,
                                b_text_ids,
                                b_code_ids,
                                cve_desc_end_idx,
                                text_len,
                                code_lens,
                                classifier_fn,
                                num_samples,
                                distance_metric='cosine'):

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100
        import copy

        sizes = [text_len - cve_desc_end_idx - 2] + [code_lens[x] - cve_desc_end_idx - 2 for x in range(len(code_lens))] # need to remove the 0 and 2
        sizes_cumsum = [0] + list(np.cumsum(sizes))

        total_size = sum(sizes)

        samples = self.random_state.randint(1, total_size + 1, num_samples - 1)
        data = np.ones((num_samples, total_size))
        new_row_list = [copy.deepcopy(this_row)]

        features_range = range(total_size)
        for i, size in enumerate(samples, start=1):
            inactive_offset = self.random_state.choice(features_range, size,
                                                replace=False)
            data[i, inactive_offset] = 0

            row_deepcopy = list(copy.deepcopy(this_row))
            text_deepcopy = copy.deepcopy(this_row[2])
            code_deepcopy = copy.deepcopy(this_row[5])

            for x in inactive_offset:
                component, bin_id, offset = get_offset(x, sizes_cumsum, this_row)
                if component == "text":
                    text_deepcopy[cve_desc_end_idx + 1 + offset] = 3
                else:
                    code_deepcopy[bin_id][cve_desc_end_idx + 1 + offset] = 3

            row_deepcopy[2] = text_deepcopy
            row_deepcopy[5] = code_deepcopy
            new_row_list.append(tuple(row_deepcopy))

        labels = classifier_fn(new_row_list)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances, sizes_cumsum


    def __data_labels_distances_textonly(self,
                                this_row,
                                b_text_ids,
                                b_code_ids,
                                cve_desc_end_idx,
                                text_len,
                                code_lens,
                                classifier_fn,
                                num_samples,
                                distance_metric='cosine'):

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100
        import copy

        sizes = [text_len - cve_desc_end_idx - 2] + [code_lens[x] - cve_desc_end_idx - 2 for x in range(len(code_lens))] # need to remove the 0 and 2
        sizes_cumsum = [0] + list(np.cumsum(sizes))

        text_size = text_len - cve_desc_end_idx - 2

        samples = self.random_state.randint(1, text_size + 1, num_samples - 1)
        data = np.ones((num_samples, text_size))
        new_row_list = [copy.deepcopy(this_row)]

        features_range = range(text_size)
        for i, size in enumerate(samples, start=1):
            inactive_offset = self.random_state.choice(features_range, size,
                                                replace=False)
            data[i, inactive_offset] = 0

            row_deepcopy = list(copy.deepcopy(this_row))
            text_deepcopy = copy.deepcopy(this_row[2])

            for x in inactive_offset:
                text_deepcopy[cve_desc_end_idx + 1 + x] = 3

            row_deepcopy[2] = text_deepcopy
            new_row_list.append(tuple(row_deepcopy))

        labels = classifier_fn(new_row_list)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances, sizes_cumsum

    def __data_labels_distances_codeonly(self,
                                this_row,
                                b_text_ids,
                                b_code_ids,
                                sorted_idx,
                                cve_desc_end_idx,
                                text_len,
                                code_lens,
                                classifier_fn,
                                num_samples,
                                distance_metric='cosine'):

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100
        import copy

        assert len(code_lens) == 1
        bin_idx = sorted_idx[0]

        sizes = [text_len - cve_desc_end_idx - 2] + [code_lens[x] - cve_desc_end_idx - 2 for x in range(len(code_lens))] # need to remove the 0 and 2
        sizes_cumsum = [0] + list(np.cumsum(sizes))

        code_size = code_lens[0] - cve_desc_end_idx - 2

        samples = self.random_state.randint(1, code_size + 1, num_samples - 1)
        data = np.ones((num_samples, code_size))
        new_row_list = [copy.deepcopy(this_row)]

        features_range = range(code_size)
        for i, size in enumerate(samples, start=1):
            inactive_offset = self.random_state.choice(features_range, size,
                                                replace=False)
            data[i, inactive_offset] = 0

            row_deepcopy = list(copy.deepcopy(this_row))
            code_deepcopy = copy.deepcopy(this_row[5])

            for x in inactive_offset:
                code_deepcopy[bin_idx][cve_desc_end_idx + 1 + x] = 3

            row_deepcopy[5] = code_deepcopy
            new_row_list.append(tuple(row_deepcopy))

        labels = classifier_fn(new_row_list)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances, sizes_cumsum
