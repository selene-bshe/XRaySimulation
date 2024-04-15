import numpy as np


def p0_dist(beta, kbar):
    return np.exp(-1 / beta * np.log1p(kbar * beta))


def p1_dist(beta, kbar):
    return kbar * np.exp(- (1 + beta) / beta * np.log1p(kbar * beta))


def p2_dist(beta, kbar):
    return 0.5 * (1 + beta) * kbar ** 2 * np.exp(- (1 + 2 * beta) / beta * np.log1p(kbar * beta))


def chisqs_NB(ps, kbar, M, npx):
    # chisqs only based on ps
    # kbar = np.tile(kbar,(3,1)).transpose()
    #     print (kbar.shape)

    kMeanNum = np.size(kbar)
    k = np.zeros((kMeanNum, 3))
    k[:, 0] = 0.
    k[:, 1] = 1.
    k[:, 2] = 2.

    kMeanHolder = np.zeros((kMeanNum, 3))
    kMeanHolder = kbar[:, np.newaxis]
    prob_distribution = stats.nbinom.pmf(k=k, n=M, p=1. / (1. + kMeanHolder / M))
    return -2 * np.nansum(ps * npx * np.log(prob_distribution / ps))


def getContrast_NB(ps, kbar, npx):
    M_all = np.linspace(1, 1000, 1000, dtype=int)
    betas = 1 / M_all
    chi2 = np.zeros(betas.size)
    for ii, M in enumerate(M_all):
        chi2[ii] = chisqs_NB(ps=ps, kbar=kbar, M=M, npx=npx)
    pos = np.argmin(chi2)
    beta0 = betas[pos]
    # curvature as error analysis
    dbeta = np.diff(betas)[0]
    delta_beta = np.sqrt(2 * dbeta ** 2 / (chi2[pos + 1] + chi2[pos - 1] - 2 * chi2[pos]))
    return betas, chi2, beta0, delta_beta
