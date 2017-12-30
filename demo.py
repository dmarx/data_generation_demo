import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=1000, n_features=10, centers=5, cluster_std=1.0)

RFC = RandomForestClassifier(n_estimators=80)
RFC.fit(X,y)

# Generate observation which maximizes a particular class probability
def class_cond_prob(x, model=RFC, class_id=0, class_err_prob=0.05):
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
    score = class_err_prob
    if model.predict(x) == model.classes_[class_id]:
        score = model.predict_proba(x)[0,class_id]
    return score

#x0 = X[y==0,][0,None]
#x_max_0 = minimize(class_spec_loss, np.ones([1,10]), method='Nelder-Mead')

# via https://gist.github.com/alexsavio/9ecdc1279c9a7d697ed3
def metropolis(f, proposal, old):
    """
    basic metropolis algorithm, according to the original,
    (1953 paper), needs symmetric proposal distribution.
    """
    new = proposal(old)
    alpha = np.min([f(new)/f(old), 1])
    u = np.random.uniform()
    # _cnt_ indicates if new sample is used or not.
    cnt = 0
    if (u < alpha):
        old = new
        cnt = 1
    return old, cnt

def run_chain(chainer, f, proposal, start, n, take=1):
    """
    _chainer_ is one of Metropolis, MH, Gibbs ...
    _f_ is the unnormalized density function to sample
    _proposal_ is the proposal distirbution
    _start_ is the initial start of the Markov Chain
    _n_ length of the chain
    _take_ thinning
    """
    count = 0
    samples = [start]
    for i in range(n):
        start, c = chainer(f, proposal, start)
        count = count + c
        if i%take is 0:
            samples.append(start)
    return samples, count

np.random.seed(123)

samples, _ = run_chain(chainer=metropolis,
                 f=class_cond_prob,
                 proposal=lambda old: old + np.random.randn(1,10)/10,
                 start=X[y==0,:][0,:],
                 n=10000,
                 take=1
                 )

burnin=1000
samples = np.concatenate(samples[burnin:])
y_pred = RFC.predict(samples)
np.mean(y_pred==0) # 0.57

pos_samples = samples[y_pred==0,:]

probs = RFC.predict_proba(pos_samples)[:,0]
top_ix = np.where(probs >= np.percentile(probs, 90))[0] # top 10% of positively classified samples by class likelihood
top_samples = pos_samples[top_ix, :]

plt.hist(probs[top_ix])
plt.show()

import seaborn as sns

f, axes = plt.subplots(2, 5, sharex='col', sharey='row')
axes = np.concatenate(axes)
for i in range(10):
    sns.kdeplot(samples[:,i], ax=axes[i])
plt.show()

f, axes = plt.subplots(2, 5, sharex='col', sharey='row')
axes = np.concatenate(axes)
for i in range(10):
    sns.kdeplot(top_samples[:,i], ax=axes[i])
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_r = pca.fit_transform(top_samples)

plt.scatter(X_r[:,0], X_r[:,1])
plt.show()


X_r = pca.fit_transform(samples)

plt.scatter(X_r[:,0], X_r[:,1], c=RFC.predict_proba(samples)[:,0])
plt.show()
