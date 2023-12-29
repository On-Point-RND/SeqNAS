import numpy as np
from catboost import CatBoostRegressor


class CatBoost_Ensemble:
    """
    Type of metapredictor : catboost regressor with uncertainty estimation.

    Parameters
    ----------
    n_ensemble:
        Number of estimators in ensemble
    epochs:
        Number of fitting epochs
    lr:
        Learning rate
    loss_func:
        Loss function
    """

    def __init__(self, n_ensemble, iterations, predictor_lr, predictor_objective):
        self.n_ensemble = n_ensemble
        self.iterations = iterations
        self.objective = predictor_objective
        self.learning_rate = predictor_lr
        self.reset()

    def reset(self):
        """
        Re-initialize parameters.
        """
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            posterior_sampling=True,
            loss_function=self.objective,
            nan_mode="Max",
            task_type="CPU",
            thread_count=self.n_ensemble,
        )

    def train(self, X, y, **kwargs):
        """
        Train ensemble.

        Parameters
        ----------
        X:
            Input data (arch features)
        y:
            Target values
        """

        self.model.fit(np.array(X), np.array(y), verbose=False)

    def predict(self, X):
        """
        Estimate mean and uncertainty

        Parameters
        ----------
        X:
            Input data (arch features)

        Returns
        -------
        ndarray:
            Predicted mean values
        ndarray:
            Standart deviations
        """
        pred = self.model.virtual_ensembles_predict(
            np.array(X),
            prediction_type="TotalUncertainty",
            virtual_ensembles_count=self.n_ensemble,
            thread_count=self.n_ensemble,
        )
        mean = pred[:, 0]
        std = np.sqrt(pred[:, 1])
        return mean, std
