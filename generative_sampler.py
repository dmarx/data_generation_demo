import datetime as dt
import inspect

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils import check_array

def is_fitted(model):
    """Checks if model object has any attributes ending with an underscore"""
    return 0 < len( [k for k,v in inspect.getmembers(model) if k.endswith('_') and not k.startswith('__')] )

class GenerativeSampler(object):
    """
    Given an sklearn model encoding P(Y|X), generates samples from P(X|Y) via MCMC.
    """
    def __init__(self,
                 model=None,
                 likelihood=None,
                 log_likelihood=None,
                 proposal=None,
                 X=None,
                 y=None,
                 x0=None,
                 target_class=None,
                 class_err_prob=None,
                 cache=True,
                 use_empirical=True,
                 n_change=1, # Number of features to modify when using empirical proposal
                 rw_std=1, # default std to use for random walk proposal
                 prior=None, # Funtion to compute P(X). Must be one of [None, 'kde', 'kde_heterog'] or a function
                 log_prior=None,
                 verbose=False
                ):
        self.model = model
        self.likelihood = likelihood
        self.log_likelihood = log_likelihood
        self.proposal = proposal
        self.X = X
        if X is not None:
            self.X = check_array(X)
        self.y = y
        self._x0 = x0
        self.target_class = target_class
        self._class_err_prob = class_err_prob
        self.cache = cache
        self.use_empirical = use_empirical
        self.n_change = n_change
        self.rw_std = rw_std
        self.prior = prior
        self.log_prior = log_prior
        self.verbose = verbose
        if use_empirical:
            assert self.X is not None
            assert self.y is not None
            assert self.target_class is not None
        self._set_proposal()
        self._set_likelihood()
        self._set_prior()
        self._ensure_fitted()
    def _msg(self, *args, **kwargs):
        if not self.verbose:
            return
        ts = "[{:%Y-%m-%d %H:%M:%S}]".format(dt.datetime.now())
        print(ts, *args)
        for k,v in kwargs.items():
            print("{} {}: {}".format(ts, k,v))
    @property
    def x0(self):
        if self._x0 is None:
            self._msg("Selecting an x0")
            assert self.X is not None
            assert self.y is not None
            X = self.X
            cand_ix = np.where(self.y == self.target_class)[0]
            ix = np.random.choice(cand_ix)
            #print("x0 ix", ix)
            self._x0 = X[ix,:]
        return self._x0
    @property
    def class_err_prob(self):
        """
        The class-conditional error probability, to be used by proposals as the
        probability of generating a sample that the model would classify incorrectly.
        """
        if self._class_err_prob is None:
            self._class_err_prob = self._calc_err_prob()
        return self._class_err_prob
    def _calc_err_prob(self, class_label=None):
        """
        For classifiers, returns(FP+FN)/N.
        """
        self._msg("Calculating class_err_prob")
        if not self.model_type is "classifier":
            raise AttributeError
        if hasattr(self.model, "oob_score_"):
            return 1-self.model.oob_score_

        assert self.X is not None
        assert self.y is not None
        assert self.target_class is not None
        if class_label is None:
            class_label = self.target_class
        X, y = self.X, self.y
        class_ix = np.where(y == self.target_class)[0]
        #print("class_ix shape", class_ix.shape)
        y_pred = self.model.predict(X[class_ix,:])
        return np.mean(y[class_ix] != y_pred)
    @property
    def model_type(self):
        """Returns either "classifier" or "regressor"."""
        class_type = "regressor"
        if isinstance(self.model, ClassifierMixin):
            class_type = "classifier"
        return class_type
    def _get_class_id(self, class_label):
        self._ensure_fitted()
        return np.where(self.model.classes_ == self.target_class)[0]
    @property
    def class_id(self):
        if not hasattr(self, "_class_id"):
            self._msg("Determining class_id")
            if self.model_type != "classifier":
                raise AttributeError
            if self.target_class is None:
                # assume we have a binary classification and want to sample from the positive class
                self.target_class = 1
                self._class_id = 1
            else:
                self._class_id = self._get_class_id(self.target_class)
        return self._class_id
    def _set_proposal(self):
        if self.proposal is None:
            if self.use_empirical:
                self.proposal = self.empirical_proposal
            else:
                self.proposal = self.random_walk_proposal
    def _set_likelihood(self):
        if (self.likelihood is None) and (self.model_type=="classifier"):
            self.likelihood = self.class_cond_prob
        else:
            raise NotImplementedError
        if self.log_likelihood is None:
            self.log_likelihood = lambda x: np.log(self.likelihood(x))
    def _set_prior(self):
        if self.prior is None:
            self.prior = lambda x: 1
        elif type(self.prior) == str:
            p_str = self.prior.lower()
            if not p_str.startswith('kde'):
                 raise NotImplementedError
            assert self.X is not None
            self._msg("Fitting KDE to approximate P(X)")
            if p_str.endswith('heterog'):
                self._fit_kde_prior_stastmodels()
            else:
                self._fit_kde_prior_sklearn()
        if self.log_prior is None:
            self.log_prior = lambda x: np.log(self.prior(x))
    def _fit_kde_prior_sklearn(self):
        from sklearn.neighbors import KernelDensity
        from sklearn.model_selection import GridSearchCV
        grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv=10, refit=True)
        kde = grid.fit(X).best_estimator_
        def log_prior(x):
            if x.ndim == 1:
                x = x.reshape(1,-1)
            return kde.score_samples(x)
        self.log_prior = log_prior
        self.prior = lambda x: np.exp(self.log_prior(x))
    def _fit_kde_prior_stastmodels(self):
        raise NotImplementedError
        pass
    def _ensure_fitted(self):
        self._msg("Ensuring model fitted")
        if not is_fitted(self.model):
            assert self.X is not None
            assert self.y is not None
            self.model.fit(self.X, self.y)

    #def class_cond_prob(self, x, model=RFC, class_id=0, class_err_prob=0.05):
    def class_cond_prob(self, x, class_id=None, class_err_prob=None):
        #print(x)
        if class_id is None:
            class_id = self.class_id
        elif class_err_prob is None:
            class_err_prob = 0
        if class_err_prob is None:
            class_err_prob = self.class_err_prob
        if x.ndim == 1:
            x = x.reshape(1,-1)
        score = class_err_prob
        if self.model.predict(x) == self.model.classes_[class_id]:
            score = self.model.predict_proba(x)[0,class_id]
        #print("score", score)
        return score
    def random_walk_proposal(self, old, std=None):
        """Gaussian random walk"""
        if std is None:
            std = self.rw_std
        try:
            n_feats = old.shape[1]
        except IndexError:
            n_feats = len(old)
        return old + np.random.randn(1, n_feats)*std
    #def empirical_proposal(self, old, X=X, y=y, class_label=0, class_err_prob=0.05, n_change=1):
    def empirical_proposal(self, old):
        """
        Generate proposals by independently sampling obvserved values for each feature from the data
        to ensure that no feature takes a value it isn't capable of taking in the real world.
        Samples are primarily constrained to the target class, but can be sampled other classes
        with probability equal to class_err_prob.
        """
        X, y = self.X, self.y
        n_feats = X.shape[1]
        U = np.random.random(n_feats)
        use_target_class = U > self.class_err_prob
        #candidate = np.empty_like(X[0,:])
        candidate = old.copy()
        to_change = np.random.choice(n_feats, self.n_change)
        for j, test in enumerate(use_target_class):
            if j not in to_change:
                continue
            feasible_ix = np.where(y==self.target_class)[0]
            if not test:
                feasible_ix = np.where(y!=self.target_class)[0]
            i = np.random.choice(feasible_ix)
            candidate[j] = X[i,j]
        return candidate

    def metropolis(self, old, u):
        # via https://gist.github.com/alexsavio/9ecdc1279c9a7d697ed3
        """
        basic metropolis algorithm, according to the original,
        (1953 paper), needs symmetric proposal distribution.
        """
        new = self.proposal(old)
        #numr = self.likelihood(new) * self.prior(new)
        #denom = self.likelihood(old) * self.prior(old)
        #alpha = np.min([numr/denom, 1])
        numr  = self.log_likelihood(new) + self.log_prior(new)
        denom = self.log_likelihood(old) + self.log_prior(old)
        alpha = np.exp(numr - denom)
        #u = np.random.uniform()
        accepted = 0
        if (u < alpha):
            old = new
            accepted = 1
        return old, accepted

    def run_chain(self, n, start=None, take=1):
        """
        _start_ is the initial start of the Markov Chain
        _n_ length of the chain
        _take_ thinning
        """
        # via https://gist.github.com/alexsavio/9ecdc1279c9a7d697ed3
        if start is None:
            start = self.x0
        count = 0
        samples = [start]
        U = np.random.uniform(size=n)
        for i in range(n):
            start, c = self.metropolis(start, U[i])
            count = count + c
            if i%take is 0:
                samples.append(start)
        self._x0 = start
        return np.vstack(samples), count


if __name__ is '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    np.random.seed(123)
    X, y = make_blobs(n_samples=1000, n_features=10, centers=5, cluster_std=3)

    RFC = RandomForestClassifier(n_estimators=80, oob_score=True)
    RFC.fit(X,y)
    print("oob_score_", RFC.oob_score_)

    _x0 = np.random.randn(10)
    sample_gen = GenerativeSampler(model=RFC, target_class=0, class_err_prob=1-RFC.oob_score_, use_empirical=False)
    test = sample_gen.run_chain(n=10, start=_x0)

    # Test that class_err_prob self populates correctly
    sample_gen = GenerativeSampler(model=RFC, X=X, y=y, target_class=0, use_empirical=False)
    #assert sample_gen.class_err_prob == 0
    print("calculated class_err_prob", sample_gen.class_err_prob) # For RFC this will always be 0 because it's calculated against the training data.
    test = sample_gen.run_chain(n=10, start=_x0)

    # test that x0 self populates correctly
    sample_gen = GenerativeSampler(model=RFC, X=X, y=y, target_class=0, use_empirical=False)
    test = sample_gen.run_chain(n=10)
    sample_gen = GenerativeSampler(model=RFC, X=X, y=y, target_class=0, use_empirical=False)
    test = sample_gen.run_chain(n=10, start=_x0)
    test = sample_gen.run_chain(n=10)

    # Test empirical proposal
    sample_gen = GenerativeSampler(model=RFC, X=X, y=y, target_class=0, use_empirical=True)
    test = sample_gen.run_chain(n=5)

    # Test unfitted model
    RFC = RandomForestClassifier(n_estimators=80, oob_score=True, n_jobs=-1)
    sample_gen = GenerativeSampler(model=RFC, X=X, y=y, target_class=0, use_empirical=True)
    test = sample_gen.run_chain(n=5)

    #############
    # Iris demo #
    #############

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    import time

    iris = load_iris()
    X, y = iris.data, iris.target
    RFC = RandomForestClassifier(n_estimators=80, oob_score=True) # inexplicably, n_jobs=-1 makes this 40x slower
    RFC.fit(X, y)

    n_generate = 1000

    iris_sample_gens = {}
    iris_samples = {}
    #for i in range(3):
    for i in range(3):
        class_label = iris.target_names[i]
        iris_sample_gens[class_label] = GenerativeSampler(model=RFC, X=X, y=y, target_class=i,
            prior='kde', class_err_prob=0, use_empirical=False, rw_std=.05, verbose=True)
        #iris_sample_gens[class_label] = GenerativeSampler(model=RFC, X=X, y=y, target_class=i, use_empirical=True, rw_std=.1, verbose=True)
        start = time.time()
        iris_samples[class_label], cnt = iris_sample_gens[class_label].run_chain(n=n_generate)
        print("elapsed:", time.time() - start)

        #import cProfile
        #cProfile.run('iris_sample_gens[class_label].run_chain(n=20)')
        burn = 100
        pca = PCA(n_components=2)
        X1 = X[y==i,:]
        X2 = iris_samples[class_label][burn:]
        #X3 = np.vstack([X2, X1])
        X3 = np.vstack([X1, X2])
        #X3 = X2
        pca.fit(X3)
        X1_r = pca.transform(X1)
        X2_r = pca.transform(X2)

        plt.scatter(X2_r[:,0], X2_r[:,1], color='r', alpha=.3)
        plt.scatter(X1_r[:,0], X1_r[:,1], color='b', alpha=.3)
        plt.show()

        y_score = RFC.predict_proba(X2)[:,i]
        plt.hist(y_score)
        plt.show()

    # When not using the empirical proposal, the chain seems to just pick a direction and run loose, way out of the set of feasible values.
    # Need to constrain random walk to observed range.

    # Also, I'm a little concerned that the empirical proposal is just walking the entire available grid

    import seaborn as sns
    import pandas as pd

    sns.pairplot(pd.DataFrame(X2))
    plt.show()

    pca.fit(X)
    cmaps = ["Reds", "Blues", "Greens"]
    for i in range(3):
        sns.kdeplot(pca.transform(X[y==i,:]), cmap=cmaps[i])
        X_gen = iris_samples[iris.target_names[i]]
        sns.kdeplot(pca.transform(X_gen), cmap=sns.dark_palette("purple", as_cmap=True))
    plt.show()
