"""SVR models for predicting the U and V chrominance channels."""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class ChromaSVR:
    def __init__(
        self,
        C=1.0,
        epsilon=0.01,
        gamma="scale",
        max_samples=15000,
        seed=0
    ):
        self.max_samples = max_samples
        self.seed = seed

        def create_svr():
            return make_pipeline(
                StandardScaler(),
                SVR(
                    kernel="rbf",
                    C=C,
                    epsilon=epsilon,
                    gamma=gamma
                )
            )

        self.svr_u = create_svr()
        self.svr_v = create_svr()

    def fit(self, X, U, V):
        """Train separate SVR models for the U and V channels."""
        if len(X) > self.max_samples:
            idx = np.random.RandomState(self.seed).choice(
                len(X),
                self.max_samples,
                replace=False
            )
            X, U, V = X[idx], U[idx], V[idx]

        self.svr_u.fit(X, U)
        self.svr_v.fit(X, V)

        return self

    def predict(self, X):
        """Predict U and V chrominance values for the input features."""
        return self.svr_u.predict(X), self.svr_v.predict(X)

    @property
    def scaler(self):
        """Return the feature scaler used by the U-channel SVR."""
        return self.svr_u.named_steps["standardscaler"]