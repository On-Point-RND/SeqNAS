from ..base_searcher import RandomSearcher
from .proxies import calculate_proxies
import numpy as np
from tqdm import tqdm


class RandomSearcherTrainFree(RandomSearcher):
    def __init__(
        self,
        model,
        loader,
        criterion,
        device="cpu",
        scoring_proxies=[
            "RegCor",
            "NTK",
        ],
        num_trails=3,
        logger=None,
        max_search_space=1e6,
        n=1,
        number_arch=1,
        n_warmup=0,
        trainer=None,
    ):
        """
        Searcher which evaluate model without training (or with minimal training if n_warmup > 0).
        Recommended to use with number_arch > 1 due to high varience in answer.

        :param model: model which represents the searchspace
        :type model: omnimodel
        :param loader: dictionary of two loaders: validation and train
        :type loader: dict
        :param criterion: loss function to dataset
        :type criterion: function
        :param device: device on which will be all calculations
        :type device: string or int
        :param scoring_proxies: names of proxies to evaluate architectures. After calculate each of this proxies their values are summed up, defaults to [ "NReg", "NReg_eo", "RegCor", "RegCor_eo", "NTK", "Snip", "Synflow", "Fisher", ]
        :type scoring_proxies: list of string, optional
        :param num_trails: number of architectures to train-free searcher
        :type num_trails: int
        :param logger: logger to searcher
        :type logger: nash_logging.common.UnitedLogger
        :param max_search_space: service parameter which stop seaarch if nuber of equal architectures which was sampled will be more than it
        :type max_search_space: int
        :param n: number of batchs to calculate proxies, defaults to 1
        :type n: int, optional
        :param number_arch: number of best architectures to train-free searcher pass into hyperband
        :type number_arch: int
        :param n_warmup: number of train epochs to warmup before scoring proxies in train-free searcher, defaults to 0
        :type n_warmup: int, optional
        :param trainer: trainer which is needful if n_warmup is greater than 0
        "type trainer: trainers.trainer.Trainer
        """
        super().__init__(
            model,
            None,
            None,
            logger=logger,
            max_search_space=max_search_space,
        )

        self.n = n
        self.model = model
        self.device = device
        self.criterion = criterion
        self.num_trails = num_trails
        self.number_arch = number_arch
        self.trainer = trainer
        self.n_warmup = n_warmup
        if self.n_warmup > 0:
            if self.trainer is None:
                print("Can't warmup without trainer!")
                return
            self.trainer.set_dataloaders(loader)
            self.trainer.set_model(
                self.model, param_groups={"main": self.model.parameters()}
            )
            self.reset_num_epochs = self.trainer.num_epochs
            self.trainer.num_epochs = self.n_warmup

        self.logger = logger
        self.val_loader = loader["validation"]
        self.loader = loader
        self.scoring_proxies = scoring_proxies
        self.max_search_space = max_search_space
        self.computed_arches = dict()

    def choose_best(self):
        """
        function to choose best number_arch architectures from num_trails architectures
        """
        proxies = {}
        for k in self.computed_arches:
            for proxy in self.computed_arches[k]["proxies"]:
                if proxy in proxies:
                    proxies[proxy].append(self.computed_arches[k]["proxies"][proxy])
                else:
                    proxies[proxy] = [self.computed_arches[k]["proxies"][proxy]]
        mean, std = {}, {}
        for proxy in proxies:
            mean[proxy] = np.mean(proxies[proxy])
            std[proxy] = np.std(proxies[proxy])
        scores = []
        for k in self.computed_arches:
            self.computed_arches[k]["score"] = np.sum(
                [
                    (self.computed_arches[k]["proxies"][proxy] - mean[proxy])
                    / std[proxy]
                    for proxy in mean
                ]
            )
            scores.append([self.computed_arches[k]["score"], k])
        scores.sort()
        len_ = min(len(scores), self.number_arch)
        ans = {}
        for i in scores[-len_:]:
            ans[i[1]] = self.computed_arches[i[1]]
        return ans

    def search(self):
        """
        Start search.
        """
        for i in tqdm(range(self.num_trails)):
            current_arch, arch_hash, end = self.get_arch()
            self.computed_arches[arch_hash] = dict()

            if end:
                break
            # RandomSearcher method
            if self.n_warmup > 0:
                self.trainer.set_model(
                    self.model, param_groups={"main": self.model.parameters()}
                )
                self.trainer.train()
            current_score = calculate_proxies(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
                self.n,
                self.scoring_proxies,
            )

            self.computed_arches[arch_hash]["proxies"] = current_score
            self.computed_arches[arch_hash]["arch"] = current_arch

        if self.n_warmup > 0:
            self.trainer.num_epochs = self.reset_num_epochs

        return self.choose_best()  # list of self.number_arch best architectures
