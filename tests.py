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
test = sample_gen.run_chain(n=10, x0=_x0)

# Test that class_err_prob self populates correctly
sample_gen = GenerativeSampler(model=RFC, X=X, y=y, target_class=0, use_empirical=False)
#assert sample_gen.class_err_prob == 0
print("calculated class_err_prob", sample_gen.class_err_prob) # For RFC this will always be 0 because it's calculated against the training data.
test = sample_gen.run_chain(n=10, x0=_x0)

# test that x0 self populates correctly
sample_gen = GenerativeSampler(model=RFC, X=X, y=y, target_class=0, use_empirical=False)
test = sample_gen.run_chain(n=10)
sample_gen = GenerativeSampler(model=RFC, X=X, y=y, target_class=0, use_empirical=False)
test = sample_gen.run_chain(n=10, x0=_x0)
test = sample_gen.run_chain(n=10)

# Test empirical proposal
sample_gen = GenerativeSampler(model=RFC, X=X, y=y, target_class=0, use_empirical=True)
test = sample_gen.run_chain(n=5)

# Test unfitted model
RFC = RandomForestClassifier(n_estimators=80, oob_score=True, n_jobs=-1)
sample_gen = GenerativeSampler(model=RFC, X=X, y=y, target_class=0, use_empirical=True)
test = sample_gen.run_chain(n=5)
