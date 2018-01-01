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
                 proposal=None,
                 X=None,
                 y=None,
                 x0=None,
                 target_class=None,
                 class_err_prob=None,
                 cache=True,
                 use_empirical=True,
                 n_change=1 # Number of features to modify when using empirical proposal
                ):
        self.model = model
        self.likelihood = likelihood
        self.proposal = proposal
        self.X = X
        self.y = y
        self._x0 = x0
        self.target_class = target_class
        self._class_err_prob = class_err_prob
        self.cache = cache
        self.use_empirical = use_empirical
        self.n_change = n_change
        self._set_proposal()
        self._set_likelihood()
        self._ensure_fitted()
    @property
    def x0(self):
        if self._x0 is None:
            assert self.X is not None
            assert self.y is not None
            X = check_array(self.X)
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
        if not self.model_type is "classifier":
            raise AttributeError
        if hasattr(self.model, "oob_score_"):
            return 1-self.model.oob_score_

        assert self.X is not None
        assert self.y is not None
        assert self.target_class is not None
        if class_label is None:
            class_label = self.target_class
        X, y = check_array(self.X), self.y
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
        return np.where(self.model.classes_ == self.target_class)
    @property
    def class_id(self):
        if not hasattr(self, "_class_id"):
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
    def _ensure_fitted(self):
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
    def random_walk_proposal(self, old, std=1):
        """Gaussian random walk"""
        return old + np.random.randn(1,len(old))*std
    #def empirical_proposal(self, old, X=X, y=y, class_label=0, class_err_prob=0.05, n_change=1):
    def empirical_proposal(self, old):
        """
        Generate proposals by independently sampling obvserved values for each feature from the data
        to ensure that no feature takes a value it isn't capable of taking in the real world.
        Samples are primarily constrained to the target class, but can be sampled other classes
        with probability equal to class_err_prob.
        """
        assert self.X is not None
        assert self.y is not None
        assert self.target_class is not None
        X, y = check_array(self.X), self.y
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

    def metropolis(self, old):
        # via https://gist.github.com/alexsavio/9ecdc1279c9a7d697ed3
        """
        basic metropolis algorithm, according to the original,
        (1953 paper), needs symmetric proposal distribution.
        """
        new = self.proposal(old)
        alpha = np.min([self.likelihood(new)/self.likelihood(old), 1])
        u = np.random.uniform()
        # _cnt_ indicates if new sample is used or not.
        cnt = 0
        if (u < alpha):
            old = new
            cnt = 1
        return old, cnt

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
        for i in range(n):
            start, c = self.metropolis(start)
            count = count + c
            if i%take is 0:
                samples.append(start)
        self._x0 = start
        return samples, count


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
