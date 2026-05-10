#ABIDE.py>

# functions

#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from dadapy import Data
from dadapy._utils import utils as ut
from scipy.optimize import minimize
from scipy.linalg import eigh, qr, solve, svd
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
from scipy.stats import ks_2samp


rng = np.random.default_rng()

class ABIDE:
    def __init__(self, X, initial_id, n_iter):
        self.X = X
        self.initial_id = initial_id
        self.n_iter = n_iter

    def return_ids_kstar_binomial(
            self, initial_id=None, Dthr=6.67, r='opt', verbose = True
        ):
            """Return the id estimates of the binomial algorithm coupled with the kstar estimation of the scale.

            Args:
                initial_id (float): initial estimate of the id default uses 2NN
                n_iter (int): number of iteration
                Dthr (float): threshold value for the kstar test
                r (float): parameter of binomial estimator, 0 < r < 1
            Returns:
                ids, ids_err, kstars, log_likelihoods
            """
            data = Data(self.X)
            # start with an initial estimate of the ID
            if initial_id is None:
                data.compute_id_2NN(algorithm='base')
            else:
                data.compute_distances()
                data.set_id(initial_id)

            ids = np.zeros(self.n_iter)
            ids_err = np.zeros(self.n_iter)
            kstars = np.zeros((self.n_iter, data.N), dtype=int)
            log_likelihoods = np.zeros(self.n_iter)
            ks_stats = np.zeros(self.n_iter)
            p_values = np.zeros(self.n_iter)

            for i in range(self.n_iter):
                # compute kstar
                data.compute_kstar(Dthr)
                if verbose == True:
                    print("iteration ", i)
                    print("id ", data.intrinsic_dim)

                # set new ratio
                r_eff = min(0.95,0.2032**(1./data.intrinsic_dim)) if r == 'opt' else r
                # compute neighbourhoods shells from k_star
                rk = np.array([dd[data.kstar[j]] for j, dd in enumerate(data.distances)])
                rn = rk * r_eff
                n = np.sum([dd < rn[j] for j, dd in enumerate(data.distances)], axis=1)
                # compute id
                id = np.log((n.mean() - 1) / (data.kstar.mean() - 1)) / np.log(r_eff)
                # compute id error
                id_err = ut._compute_binomial_cramerrao(id, data.kstar-1, r_eff, data.N)
                # compute likelihood
                log_lik = ut.binomial_loglik(id, data.kstar - 1, n - 1, r_eff)
                # model validation through KS test
                n_model = rng.binomial(data.kstar-1, r_eff**id, size=len(n))
                ks, pv = ks_2samp(n-1, n_model)
                # set new id
                data.set_id(id)

                ids[i] = id
                ids_err[i] = id_err
                kstars[i] = data.kstar
                log_likelihoods[i] = log_lik
                ks_stats[i] = ks
                p_values[i] = pv

            data.intrinsic_dim = id
            data.intrinsic_dim_err = id_err
            data.intrinsic_dim_scale = 0.5 * (rn.mean() + rk.mean())

            return ids, kstars[(self.n_iter - 1), :]#, ids_err, log_likelihoods, ks_stats, p_values