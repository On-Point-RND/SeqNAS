import numpy as np
import pickle


class Priors:
    """
    Service class to hold information about past experiments
    """

    def __init__(self):
        self.datasets = {}
        self.experiments = {}
        self.spaces = {}
        self.searchers = {}
        self.scale = [[0, 1] for i in range(3)]
        self.max = 1

    @classmethod
    def from_pkl(cls, file):
        """Reconstructs priors from pickle file.

        :param file: opened pkl file with data
        :type file: byte file-like object

        :return: instance of the class
        :rtype: Priors
        """

        attrs = pickle.load(file)
        priors = object.__new__(Priors)
        priors.__dict__.update(attrs)
        return priors

    def to_pkl(self, file):
        """Saves priors to pickle file.

        :param file: opened pkl file with data
        :type file: byte file-like object
        """
        pickle.dump(self.__dict__, file)

    def add_exp(self, dataset, space, searcher, result):
        """
        Add new experiment

        Args:
            dataset (string): name of experiment (must be added with add_dataset before!)
            space (string): name of searcspace (can be new)
            searcher (string): name of searcher (can be new)
            result (double): result of experiment (value of scoring metric)
        """
        if (dataset, space, searcher) in self.experiments:
            print("Experiment always exist, so statistics will be updated")
            self.experiments[(dataset, space, searcher)] += result
            self.experiments[(dataset, space, searcher)] /= 2
        else:
            self.experiments[(dataset, space, searcher)] = result
        self.recount()

    def remove_exp(self, dataset, space, searcher):
        """
        Remove experiment

        Args:
            dataset (string): name of experiment
            space (string): name of searcspace
            searcher (string): name of searcher
            result (double): result of experiment
        """
        del self.experiments[(dataset, space, searcher)]
        self.recount()

    def recount(self):
        """
        This class holds information in prepare to analysis form, this function calculate that form
        """
        self.spaces = {(exp[0], exp[1]): [] for exp in self.experiments}
        self.searchers = {(exp[0], exp[2]): [] for exp in self.experiments}
        for exp in self.experiments:
            self.spaces[(exp[0], exp[1])].append(self.experiments[exp])
            self.searchers[(exp[0], exp[2])].append(self.experiments[exp])
        for sp in self.spaces:
            self.spaces[sp] = np.mean(self.spaces[sp])
        for s in self.searchers:
            self.searchers[s] = np.mean(self.searchers[s])

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
        Add new dataset

        Args:
            name (string): name of dataset
            seq_len (int): number of timesteps in one sample
            size (int): number of samples in dataset
            num_categorical (int): number of categorical features
            num_continious (int): number of continuous features
            is_financial (bool): is dataset contain a financial data
            is_table (bool): is dataset contain a table data
            is_signal (bool): is dataset contain a signal data
            is_classification (bool): is dataset for classification task
        """

        # Create initial features of dataset
        self.max = max(size, self.max)
        self.datasets[name] = {
            "raw": np.array(
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
            )
        }

        # Recount points in dataset space
        clear = np.vstack([d["raw"] for name, d in self.datasets.items()]).astype(float)
        for i in range(3):
            add_part = clear[:, i].min()
            div_part = clear[:, i].max() - clear[:, i].min() + 1e-9
            self.scale[i] = [add_part, div_part]
            clear[:, i] = (clear[:, i] - add_part) / div_part
        clear[:, 4] /= np.max(clear[:, 4])
        clear[:, 4] = clear[:, 4] ** 0.33
        for i, name in enumerate(self.datasets):
            self.datasets[name]["clear"] = clear[i]

    def remove_dataset(self, name):
        """
        Remove dataset

        Args:
            name (str): name of dataset to remove
        """
        del self.datasets[name]
        for exp in list(self.experiments.keys()):
            if exp[0] == name:
                del self.experiments[exp]
        self.recount()

    def remove_space(self, name):
        """
        Remove searchspace

        Args:
            name (str): name of searchspace to remove
        """
        for sp in list(self.spaces.keys()):
            if sp[1] == name:
                del self.spaces[sp]
        for exp in list(self.experiments.keys()):
            if exp[1] == name:
                del self.experiments[exp]
        self.recount()

    def remove_searcher(self, name):
        """
        Remove searcher

        Args:
            name (str): name of searcher to remove
        """
        for sp in list(self.spaces.keys()):
            if sp[1] == name:
                del self.searchers[sp]
        for exp in list(self.experiments.keys()):
            if exp[2] == name:
                del self.experiments[exp]
        self.recount()
