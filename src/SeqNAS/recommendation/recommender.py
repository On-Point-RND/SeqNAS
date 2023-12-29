from .priors import Priors
from . import data as data_pkg

# `files` is new in python 3.9, `open_binary` is deprecated in python 3.11
try:
    from importlib.resources import files
except ImportError:
    from importlib.resources import open_binary
else:

    def open_binary(package, resource):
        return files(package).joinpath(resource).open("rb")


import numpy as np


class Recommender:
    """
    Prime class to get recommendations.

    :param path_to_priors: path to pickle with Priors object which contains information to predict
    :type path_to_priors: string, default to None for the builtin priors
    :param treshold: distance treshold, if dataset farther than treshold distance to them will be set to 0.99
    :type treshold: float, default to 0.4
    """

    def __init__(self, path_to_priors=None, treshold=0.4):
        self.treshold = treshold

        if path_to_priors is None:
            self._load_builtin_priors()
        else:
            try:
                with open(path_to_priors, "rb") as f:
                    self.priors = Priors.from_pkl(f)
            except Exception as e:
                print(
                    f"Can't open file with priors, error: {e}"
                    " Falling back to builtin priors"
                )
                self._load_builtin_priors()

    def _load_builtin_priors(self):
        try:
            f = open_binary(data_pkg, "default_priors.pkl")
            self.priors = Priors.from_pkl(f)
            f.close()
        except Exception as e:
            print(
                f"Can't open builtin priors, error: {e}" " Falling back to empty priors"
            )
            self.priors = Priors()

    def add_exp(self, dataset, space, searcher, result):
        """
        Add new experiment (check if dataset was added before).

        :param dataset: name of experiment (must be added with add_dataset before!)
        :type dataset: string
        :param space: name of searcspace (can be new)
        :type space: string
        :param searcher: name of searcher (can be new)
        :type searcher: string
        :param result: result of experiment (value of scoring metric)
        :type result: float
        """
        if dataset not in self.priors.datasets:
            print(
                "Can't add this experiment, dataset not added. Please add this dataset"
            )
        else:
            self.priors.add_exp(dataset, space, searcher, result)

    def remove_exp(self, dataset, space, searcher):
        """
        Remove experiment (check if experiment was added before).

        :param dataset: name of experiment
        :type dataset: string
        :param space: name of searcspace
        :type space: string
        :param searcher: name of searcher
        :type searcher: string
        """
        if (dataset, space, searcher) in self.priors.experiments:
            self.priors.remove_exp(dataset, space, searcher)
        else:
            print("Experiment not found")

    def add_dataset(
        self,
        name,
        seq_len,
        size,
        num_categorical,
        num_continious,
        is_financial,
        is_table,
        is_signal,
        is_classification,
    ):
        """
        Add new dataset (check if name is unique)

        :param name: name of dataset
        :type name: string
        :param seq_len: number of timesteps in one sample
        :type seq_len: int
        :param size: number of samples in dataset
        :type size: int
        :param num_categorical: number of categorical features
        :type num_categorical: int
        :param num_continious: number of continuous features
        :type num_continious: int
        :param is_financial: is dataset contain a financial data
        :type is_financial: int (0 or 1)
        :param is_table: is dataset contain a table data
        :type is_table: int (0 or 1)
        :param is_signal: is dataset contain a signal data
        :type is_signal:int (0 or 1)
        :param is_classification: is dataset for classification task
        :type is_classification: int (0 or 1)
        """
        if name in self.priors.datasets:
            print("Dataset with this name already exist, please create another name")
        else:
            self.priors.add_dataset(
                name,
                seq_len,
                size,
                num_categorical,
                num_continious,
                is_financial,
                is_table,
                is_signal,
                is_classification,
            )

    def remove_dataset(self, dataset):
        """
        Remove dataset (check if dataset was added before).

        :param dataset: name of experiment
        :type dataset: string
        """
        if dataset in self.priors.datasets:
            self.priors.remove_dataset(dataset)

    def remove_space(self, space):
        """
        Remove searchspace.

        :param space: name of searcspace
        :type space: string
        """
        self.priors.remove_space(space)

    def remove_searcher(self, searcher):
        """
        Remove searcher.

        :param searcher: name of searcher
        :type searcher: string
        """
        self.priors.remove_searcher(searcher)

    def get_searcher(
        self,
        seq_len,
        size,
        num_categorical,
        num_continious,
        is_financial,
        is_table,
        is_signal,
        is_classification,
    ):
        """
        Range all available searchers by their predictable performance at new dataset with passed parameters.

        :param seq_len: number of timesteps in one sample
        :type seq_len: int
        :param size: number of samples in dataset
        :type size: int
        :param num_categorical: number of categorical features
        :type num_categorical: int
        :param num_continious: number of continuous features
        :type num_continious: int
        :param is_financial: is dataset contain a financial data
        :type is_financial: int (0 or 1)
        :param is_table: is dataset contain a table data
        :type is_table: int (0 or 1)
        :param is_signal: is dataset contain a signal data
        :type is_signal:int (0 or 1)
        :param is_classification: is dataset for classification task
        :type is_classification: int (0 or 1)
        """

        # Create point in datasets space
        size = (size / self.priors.max) ** (0.33)
        vector = np.array(
            [
                seq_len,
                num_categorical,
                num_continious,
                is_financial,
                size,
                is_table,
                is_signal,
                is_classification,
            ]
        ).astype(float)
        vector = self.norm_dataset(vector)

        datasets_prob = self.get_dataset_prob(vector)
        ans = self.get_scores(datasets_prob, "searchers")
        for i in sorted(ans, reverse=True):
            print(ans[i], "has probability", i)

    def get_space(
        self,
        seq_len,
        size,
        num_categorical,
        num_continious,
        is_financial,
        is_table,
        is_signal,
        is_classification,
    ):
        """
        Range all available searchspaces by their predictable performance at new dataset with passed parameters.

        :param seq_len: number of timesteps in one sample
        :type seq_len: int
        :param size: number of samples in dataset
        :type size: int
        :param num_categorical: number of categorical features
        :type num_categorical: int
        :param num_continious: number of continuous features
        :type num_continious: int
        :param is_financial: is dataset contain a financial data
        :type is_financial: int (0 or 1)
        :param is_table: is dataset contain a table data
        :type is_table: int (0 or 1)
        :param is_signal: is dataset contain a signal data
        :type is_signal:int (0 or 1)
        :param is_classification: is dataset for classification task
        :type is_classification: int (0 or 1)
        """

        # Create point in datasets space
        size = (size / self.priors.max) ** (0.33)
        vector = np.array(
            [
                seq_len,
                num_categorical,
                num_continious,
                is_financial,
                size,
                is_table,
                is_signal,
                is_classification,
            ]
        ).astype(float)
        vector = self.norm_dataset(vector)

        datasets_prob = self.get_dataset_prob(vector)
        ans = self.get_scores(datasets_prob, "spaces")
        for i in sorted(ans, reverse=True):
            print(ans[i], "has probability", i)

    def get_space_and_searcher(
        self,
        seq_len,
        size,
        num_categorical,
        num_continious,
        is_financial,
        is_table,
        is_signal,
        is_classification,
        treshold=0.5,
    ):
        """
        Range all available pairs (searchspace, searcher) by their predictable performance at new dataset with passed parameters,
        trying to predict value of scoring metric (Warning! If there is no experiments with passed dataset, score may be inadequate).

        :param seq_len: number of timesteps in one sample
        :type seq_len: int
        :param size: number of samples in dataset
        :type size: int
        :param num_categorical: number of categorical features
        :type num_categorical: int
        :param num_continious: number of continuous features
        :type num_continious: int
        :param is_financial: is dataset contain a financial data
        :type is_financial: int (0 or 1)
        :param is_table: is dataset contain a table data
        :type is_table: int (0 or 1)
        :param is_signal: is dataset contain a signal data
        :type is_signal:int (0 or 1)
        :param is_classification: is dataset for classification task
        :type is_classification: int (0 or 1)
        """

        # Create point in datasets space
        size = (size / self.priors.max) ** (0.33)
        vector = np.array(
            [
                seq_len,
                num_categorical,
                num_continious,
                is_financial,
                size,
                is_table,
                is_signal,
                is_classification,
            ]
        ).astype(float)
        vector = self.norm_dataset(vector)

        datasets_prob = self.get_dataset_prob(vector)
        ans = self.weight_exp(
            datasets_prob, max(1 - treshold, self.treshold) - self.treshold + 1e-2
        )
        if not ans:
            print("There is no experiments to get the score")
        for i in sorted(ans, reverse=True):
            print(ans[i], "would has score", i)

    def save_to(self, path):
        """
        Save Priors instance to path.

        :param path: path to save Priors instance
        :type path: string
        """
        with open(path, "wb") as f:
            self.priors.to_pkl(f)

    # Usefull functions which SHOULDN`T BE CALLED manually
    def norm_dataset(self, vector):
        for i in range(3):
            vector[i] = (vector[i] - self.priors.scale[i][0]) / self.priors.scale[i][1]
        return vector

    def get_dataset_prob(self, vector):
        return {
            name: max(
                self.treshold,
                (d["clear"] * vector).sum()
                / (np.linalg.norm(d["clear"]) * np.linalg.norm(vector)),
            )
            - self.treshold
            + 1e-2
            for name, d in self.priors.datasets.items()
        }

    def weight_exp(self, datasets_prob, tr):
        # Calculate scale for datasets_prob
        datasets_scale = {
            (exp[1], exp[2]): np.sum(
                [
                    datasets_prob[ex[0]]
                    for ex in self.priors.experiments
                    if ex[1] == exp[1] and ex[2] == exp[2]
                ]
            )
            for exp in self.priors.experiments
        }

        # return weigthed average of experiments' results across dataset with weights = datasets_prob
        # count if only at least one dataset is closer to target than 0.3
        return {
            np.sum(
                [
                    datasets_prob[ex[0]]
                    * self.priors.experiments[ex]
                    / datasets_scale[(exp[1], exp[2])]
                    for ex in self.priors.experiments
                    if ex[1] == exp[1] and ex[2] == exp[2]
                ]
            ): (exp[1], exp[2])
            for exp in self.priors.experiments
            if datasets_prob[exp[0]] >= tr
        }

    def get_scores(self, datasets_prob, type_):
        ind = 2 if type_ == "searchers" else 1
        ans = {i[ind]: 0 for i in self.priors.experiments}
        # Calculate scale for exp probabilities
        exp_scale = {
            (exp[3 - ind], exp[0]): [
                self.priors.experiments[exp_]
                for exp_ in self.priors.experiments
                if exp[3 - ind] == exp_[3 - ind] and exp[0] == exp_[0]
            ]
            for exp in self.priors.experiments
        }
        exp_scale = {k: np.sum(i) for k, i in exp_scale.items()}

        self.mean_priors = (
            self.priors.spaces if type_ == "searchers" else self.priors.searchers
        )
        for exp in self.priors.experiments:
            # Realize p(searcher) = Sum(d in Datasets) (Sum(s in Spaces) (p(searcher | d, s) * p(d)**2 * p(s | d)))
            ans[exp[ind]] += (
                self.priors.experiments[exp]
                * (datasets_prob[exp[0]] ** 2)
                * self.mean_priors[(exp[0], exp[3 - ind])]
                / exp_scale[(exp[3 - ind], exp[0])]
            )
        scores = np.array([score for name, score in ans.items()])
        scores = scores / np.sum(scores)
        ans = {scores[i]: name for i, name in enumerate(ans)}
        return ans
