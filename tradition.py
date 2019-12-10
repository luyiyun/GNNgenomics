import numpy as np
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import \
    ComponentwiseGradientBoostingSurvivalAnalysis as CGBSA
from sksurv.ensemble import GradientBoostingSurvivalAnalysis as GBSA
from sksurv.svm import FastKernelSurvivalSVM as FKSVM
from sksurv.svm import FastSurvivalSVM as FSVM


def traditional_surv_analysis(datas, opts):

    # tidy data as ndarray
    train_X, train_Y = datas["train"].xs.numpy(), datas["train"].ys.numpy()
    test_X, test_Y = datas["test"].xs.numpy(), datas["test"].ys.numpy()
    if "val" in datas.keys():
        train_X = np.concatenate([train_X, datas["val"].xs])
        train_Y = np.concatenate([train_Y, datas["val"].ys])
    # construct structured array
    train_Y = Surv.from_arrays(train_Y[:, 1].astype("bool"), train_Y[:, 0])
    test_Y = Surv.from_arrays(test_Y[:, 1].astype("bool"), test_Y[:, 0])

    # construct estimators
    estimators = {
        "CoxPH": CoxPHSurvivalAnalysis(),
        "CGBSA": CGBSA(n_estimators=500, random_state=opts.random_seed),
        "GBSA": GBSA(n_estimators=500, random_state=opts.random_seed),
        "FKSVM": FKSVM(random_state=opts.random_seed),
        "FSVM": FSVM(random_state=opts.random_seed)
    }

    # training
    for name, estimator in estimators.items():
        print("%s training." % name)
        estimator.fit(train_X, train_Y)

    # evaluation
    train_scores = {}
    test_scores = {}
    for name, estimator in estimators.items():
        print("%s evaluation." % name)
        train_scores[name] = estimator.score(train_X, train_Y)
        test_scores[name] = estimator.score(test_X, test_Y)

    # return
    return train_scores, test_scores
