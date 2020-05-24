# JDA的核心理念是将数据映射到一个分布，同时减小源域和目标域的边缘分布差异和条件分布差异
import numpy as np
import scipy.sparse.linalg as SSL
import scipy.linalg as SL
import scipy.stats as stats
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split

def kernel(ker, X1, X2, gamma):
    """
    :param ker: 核函数类型
    :param X1: 
    :param X2: 
    :param gamma: 
    :return: 
    """
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K

def JDA(Xs, Xt, Ys, Yt0, options):
    """
    :param Xs: 源域
    :param Xt: 目标域
    :param Ys: 源域标签
    :param Yt0: 目标域标签
    :param options: JDA参数
    :return: 变换后的特征值A和特征向量Z
    """
    k = options["k"]
    lmbda = options["lmbda"]
    ker = options["ker"] # 'primal' | 'linear' | 'rbf'
    gamma = options["gamma"]

    # Set predefined variables
    X = np.hstack((Xs, Xt))
    X = np.matmul(X, np.diag(1 / np.sqrt(np.sum(np.square(X), 0))))
    m, n = X.shape
    ns = Xs.shape[-1]
    nt = Xt.shape[-1]
    C = len(np.unique(Ys))

    # Construct MMD matrix
    a = 1 / (ns * np.ones([ns, 1]))
    b = -1 / (nt * np.ones([nt, 1]))
    e = np.vstack((a, b))
    M = np.matmul(e, e.T) * C

    if len(Yt0) != 0 and len(Yt0) == nt:
        for c in np.unique(Yt0):
            e = np.zeros([n, 1])
            e[Ys == c] = 1 / len(e[Ys == c])
            e[ns + np.where(Yt0 == c)[0]] = -1 / np.where(Yt0 == c)[0].shape[0]
            e[np.where(np.isinf(e))[0]] = 0
            M = M + np.matmul(e, e.T)

    # ‘fro’  A和A‘的积的对角线和的平方根，即sqrt(sum(diag(A'*A)))
    divider = np.sqrt(np.sum(np.diag(np.matmul(M.T, M))))
    M = M / divider

    # Construct centering matrix
    a = np.eye(n)
    b = 1 / (n * np.ones([n, n]))
    H = a - b

    # Joint Distribution Adaptation: JDA
    if "primal" == ker:
        pass
    else:
        K = kernel(ker, X, None, gamma)
        a = np.matmul(np.matmul(K, M), K.T) + options["lmbda"] * np.eye(n)
        b = np.matmul(np.matmul(K, H), K.T)
        print("calculate eigen value and eigen vector")
        eigenvalue, eigenvector = SL.eig(a, b)
        print("eigen value and eigen vector calculated!")
        av = np.array(list(map(lambda item: np.abs(item), eigenvalue)))
        idx = np.argsort(av)[:k]
        _ = eigenvalue[idx]
        A = eigenvector[:, idx]
        Z = np.matmul(A.T, K)

    print('Algorithm JDA terminated!!!\n\n')

    return Z, A

if __name__ == '__main__':
    options = {"k": 200, "lmbda": 1.0, "ker": 'linear', "gamma": 1.0}
    T = 20
    result = []

    data = pd.read_excel('tra_sample.xlsx')
    feature_lst = ['zx_score', 'msg_cnt', 'phone_num_cnt', 'register_days']
    target = data[data.type == 'target'].reset_index().copy()
    origin = data[data.type == 'origin'].reset_index().copy()
    val = data[data.type == 'offtime'].reset_index().copy()

    # Source
    fts, _, Ys, _ = train_test_split(origin[feature_lst], origin['bad_ind'], test_size=0.9, random_state=64)
    fts = fts.values
    fts = list(map(lambda item: item / sum(item), fts))
    mean = np.mean(fts)
    std = np.std(fts)
    Xs = (fts - mean) / std
    Xs = Xs.T
    Ys = Ys.values

    # Target
    fts, _, Yt, _ = train_test_split(target[feature_lst], target['bad_ind'], test_size=0.9, random_state=64)
    fts = fts.values
    fts = list(map(lambda item: item / sum(item), fts))
    mean = np.mean(fts)
    std = np.std(fts)
    Xt = (fts - mean) / std
    Xt = Xt.T
    Yt = Yt.values

    print("data prepared!")

    ns = Xs.shape[-1]
    nt = Xt.shape[-1]

    # soft label
    clf = LogisticRegression()
    clf.fit(Xs.T, Ys)

    Cls = clf.predict(Xt.T)
    acc = len(Cls[Cls == Yt]) / len(Yt)
    print("first cls: {}".format(acc))

    Cls = []
    Acc = []
    for t in range(T):
        print('==============================Iteration {} =============================='.format(t))
        Z, A = JDA(Xs, Xt, Ys, Cls, options)
        Z = np.matmul(Z, np.diag(1 / np.sqrt(np.sum(np.square(Z), 0))))

        Zs = Z[:, :ns]
        Zt = Z[:, ns:]

        clf = LogisticRegression()
        clf.fit(Zs.T, Ys)

        Cls = clf.predict(Zt.T)
        acc = len(Cls[Cls == Yt]) / len(Yt)
        print("JDA + NN = {}".format(acc))
        Acc.append(acc)
    np.save(feature_prefix.format(source, target, "source"), {"feature": Zs, "label": Ys})
    np.save(feature_prefix.format(target, target, "target"), {"feature": Zt, "label": Yt})

    result.append(Acc[-1])
