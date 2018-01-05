import copy
import datetime as dt
import inspect

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils import check_array

def is_fitted(model):
    """Checks if model object has any attributes ending with an underscore"""
    return 0 < len( [k for k,v in inspect.getmembers(model) if k.endswith('_') and not k.startswith('__')] )

def safe_log(x, buffer=1e-9):
    try:
        if x == 0:
            x = buffer
    except ValueError:
        print("ARRAY PASSED TO SAFE_LOG")
        print(x.shape)
        x[x==0] = buffer
    return np.log(x)

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
                 prior=None, # Funtion to compute P(X). Must be one of [None, 'kde', 'kde_heterog', 'cade'] or a function
                 prior_weight = 0.5,
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
        self.prior_weight = prior_weight
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
        """
        If a starting value for the chain was not specified, use a random observation
        from the target class.
        """
        if self._x0 is None:
            self._msg("Selecting an x0")
            assert self.X is not None
            assert self.y is not None
            X = self.X
            cand_ix = np.where(self.y == self.target_class)[0]
            ix = np.random.choice(cand_ix)
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
        return np.where(self.model.classes_ == class_label)[0]
    @property
    def class_id(self):
        """Returns the index of the class label, for use with estimator.predict_proba"""
        if not hasattr(self, "_class_id"):
            self.set_target_class(self.target_class)
        return self._class_id
    def set_target_class(self, target_class):
        """
        Use this method to change the target_class of an instance. Alternatively,
        you could create a separate instance for each target class, but if you are
        using a KDE prior then each instance will repeat the KDE fitting operation,
        which could be expensive.
        """
        self._msg("Determining class_id")
        if self.model_type != "classifier":
            raise AttributeError
        if target_class is None:
            self._msg("target_class not provided, assuming binary classification and using target_class = 1")
            self.target_class = 1
            self._class_id = 1
        else:
            self.target_class = target_class
            self._class_id = self._get_class_id(target_class)
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
            self.log_likelihood = lambda x: safe_log(self.likelihood(x))
    def _set_prior(self):
        if self.prior is None:
            self.prior = lambda x: 1
            self.log_prior = lambda x: 0
        elif type(self.prior) == str:
            assert self.X is not None
            p_str = self.prior.lower()
            self._msg("Fitting", p_str," to approximate P(X)")
            if p_str == 'cade':
                self._fit_cade_prior()
            elif p_str == 'kde':
                self._fit_kde_prior_sklearn()
            elif p_str == 'kde_heterog':
                self._fit_kde_prior_stastmodels()
            else:
                 raise NotImplementedError
            if self.log_prior is None:
                self.log_prior = lambda x: safe_log(self.prior(x))
    def _fit_kde_prior_sklearn(self):
        """
        Fits a KDE to be used for P(X). The bandwidth is determined by grid search over
        np.linspace(0.1, 1.0, 30) using 10-fold CV.
        """
        from sklearn.neighbors import KernelDensity
        from sklearn.model_selection import GridSearchCV
        grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv=10, refit=True)
        kde = grid.fit(self.X).best_estimator_
        def log_prior(x):
            if x.ndim == 1:
                x = x.reshape(1,-1)
            return kde.score_samples(x)
        self.log_prior = log_prior
        self.prior = lambda x: np.exp(self.log_prior(x))
    def _fit_kde_prior_stastmodels(self):
        """
        Unlike the sklearn KDE, statsmodels.nonparametric.kernel_density.KDEMultivariate
        supports heterogenous data. Bandwidth is fitted via LOO-CV, which is going to be
        expensive with large datasets.
        """
        raise NotImplementedError
        pass
    def _ensure_fitted(self):
        self._msg("Ensuring model fitted")
        if not is_fitted(self.model):
            assert self.X is not None
            assert self.y is not None
            self.model.fit(self.X, self.y)

    def class_cond_prob(self, x, class_id=None, class_err_prob=None):
        """
        Returns the class-conditional likelihood for x given by the model. If the
        model does not classify x as belonging to the conditioning class, then
        class_err_prob is returned instead.
        """
        if class_id is None:
            class_id = self.class_id
        elif class_err_prob is None:
            class_err_prob = 1e-9
        if class_err_prob is None:
            class_err_prob = self.class_err_prob
        if x.ndim == 1:
            x = x.reshape(1,-1)
        score = class_err_prob
        if self.model.predict(x) == self.model.classes_[class_id]:
            score = self.model.predict_proba(x)[0,class_id]
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
    def run_chain(self, n, x0=None):
        self._msg("Generating samples")
        if x0 is None:
            x0 = self.x0
        accepted = 0
        samples = []
        U = np.random.uniform(size=n)
        a, b = 1- self.prior_weight, self.prior_weight
        for i in range(n):
            x1 = self.proposal(x0)
            if i == 0:
                denom = a * self.log_likelihood(x0) + b * self.log_prior(x0)
            numr  = a * self.log_likelihood(x1) + b * self.log_prior(x1)
            alpha = np.exp(numr - denom)
            if (U[i] < alpha):
                x0, denom = x1, numr
                accepted += 1
            samples.append(x0)
        self._x0 = x0 # so we can easily pick up where we left off.
        return np.vstack(samples), accepted
    def _fit_cade_prior(self, base_estimator=None):
        """
        This doesn't seem to work particularly well, but it might at least be an interesting
        option for heterogenous data.
        """
        # A big issue here is probably that the empirical proposal won't take us "out of the box"
        # of the data far enough, which is really what we need here.
        if base_estimator is None:
            base_estimator = RandomForestClassifier(n_estimators=80)
        X_sim, X = [], self.X
        old_n_change = self.n_change
        self.n_change = X.shape[1]
        n = X.shape[0]
        for i in range(n):
            x0 = X[np.random.choice(n),:]
            x1 = self.empirical_proposal(x0)
            X_sim.append(x1)
        self.n_change = old_n_change
        X_sim = np.vstack(X_sim)
        X = np.vstack([X, X_sim])
        y = np.concatenate([np.ones(n), np.zeros(n)])
        cade = base_estimator.fit(X, y)
        def cade_prior(x):
            if x.ndim == 1:
                x = x.reshape(1,-1)
            return cade.predict_proba(x)[:,1]
        self.prior = cade_prior
        self.log_prior = lambda x: safe_log(self.prior(x))




if __name__ is '__main__':
    import time

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(123)

    iris = load_iris()
    X, y = iris.data, iris.target
    RFC = RandomForestClassifier(n_estimators=80, oob_score=True) # inexplicably, n_jobs=-1 makes this 40x slower
    RFC.fit(X, y)

    n_generate = 5000

    iris_sample_gens = {}
    iris_samples = {}

    sampler = None
    for i in range(3):
        start = time.time()
        class_label = iris.target_names[i]
        if sampler is None:
            sampler = GenerativeSampler(model=RFC, X=X, y=y, target_class=i, prior='kde',
                prior_weight=0.5, class_err_prob=0, use_empirical=False, rw_std=.05, verbose=True)
        sampler.set_target_class(i)
        iris_sample_gens[class_label] = copy.deepcopy(sampler)
        iris_samples[class_label], cnt = iris_sample_gens[class_label].run_chain(n=n_generate)
        print("elapsed:", time.time() - start)

        burn = 100
        thin = 20
        pca = PCA(n_components=2)
        X1 = X[y==i,:]
        X2 = iris_samples[class_label][burn::thin, :]
        X3 = np.vstack([X1, X2])

        pca.fit(X3)
        X1_r = pca.transform(X1)
        X2_r = pca.transform(X2)

        plt.scatter(X2_r[:,0], X2_r[:,1], color='r', alpha=.3)
        plt.scatter(X1_r[:,0], X1_r[:,1], color='b', alpha=.3)
        plt.show()

        y_score = RFC.predict_proba(X2)[:,i]
        plt.hist(y_score)
        plt.show()


    pca.fit(X)
    cmaps = ["Reds", "Blues", "Greens"]
    cmaps2 = ["red", "blue", "green"]
    for i in range(3):
        sns.kdeplot(pca.transform(X[y==i,:]), cmap=cmaps[i])
        X_gen = iris_samples[iris.target_names[i]][burn::thin, :]
        sns.kdeplot(pca.transform(X_gen), cmap=sns.dark_palette(cmaps2[i], as_cmap=True))
    plt.show()

    # Our posterior doesn't look super different from our prior. Not sure if this is good or bad.
    # Maybe I need to demo this with messier data?
    for i in range(3):
        f, axes = plt.subplots(1, 4)#, sharex='col', sharey='row')
        #axes = np.concatenate(axes)
        X_gen = iris_samples[iris.target_names[i]][burn::thin, :]
        for j in range(4):
            sns.kdeplot(X_gen[:,j], ax=axes[j])
            sns.kdeplot(X[y==i,j], ax=axes[j], color='r')
        plt.show()
