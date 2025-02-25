#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on %(date)s

@author: Chance Fleeting

This code is formatted in accordance with PEP8
See: https://peps.python.org/pep-0008/

use %matplotlib qt to display images in pop out
use %matplotlib inline to display images inline
"""

from __future__ import annotations

__author__ = 'Chance Fleeting'
__version__ = '0.0'

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.stats import scoreatpercentile
import matplotlib.pyplot as plt

import fun_GetCov_SeqADMM_SelectTuningPar # Converted from the companion code.

#%% CONSTANTS
MUTE: bool = True

#%% CLASSES

#%% FUNCTIONS
def realsym(A: np.ndarray) -> np.ndarray:
    """
    Compute the real symmetric part of a square matrix A.

    Parameters:
    ----------
    A : np.ndarray
        A square matrix.

    Returns:
    -------
    np.ndarray
        The real symmetric part of the matrix.
    """
    B = np.real((A + A.T) / 2)
    return B

def get_smooth_D_0610(p: int, m: int) -> np.ndarray:
    """
    Generate the smoothing matrix D based on input parameters.

    Parameters:
    ----------
    p : int
        Total number of time points.
    m : int
        Number of variables.

    Returns:
    -------
    np.ndarray
        The smoothing matrix D.
    """
    Delta = np.zeros((p // m - 2, p // m))
    
    for i in range(p // m - 2):
        Delta[i, i] = 0.5
        Delta[i, i + 2] = 0.5
        Delta[i, i + 1] = -1
    
    D0 = Delta.T @ Delta
    
    D0_list = [D0] * m
    
    #print(D0_list)
    
    D = block_diag(*D0_list)
    
    return D

def multilevel_s_c(x_c: np.ndarray, L1: int, L2: int, option: dict) -> dict:
    """
    Perform multilevel covariance computation.

    Parameters:
    ----------
    x_c : np.ndarray
        Centered data.
    L1 : int
        First level size.
    L2 : int
        Second level size.
    option : dict
        Configuration options.

    Returns:
    -------
    dict
        Dictionary containing covariance matrices and other computed values.
    """
    # Wrapper for fun_GetCov_SeqADMM_SelectTuningPar.MultilevelS_c
    return fun_GetCov_SeqADMM_SelectTuningPar.MultilevelS_c(x_c, L1, L2, option)

def generate_gamma(S: np.ndarray, m: int, nsol: int, minv: float = None, maxv: float = None, skew: float = 1) -> np.ndarray:
    """
    Generate a sequence of gamma values based on the input parameters.

    Parameters:
    ----------
    S : np.ndarray
        The covariance matrix.
    m : int
        Number of variables.
    nsol : int
        Number of solutions or gamma values to generate.
    minv : float, optional
        Minimum value of gamma. Defaults to 0 if not provided.
    maxv : float, optional
        Maximum value of gamma. Defaults to the largest eigenvalue of S times (p/m) if not provided.
    skew : float, optional
        Skewness factor for the gamma sequence. Defaults to 1.

    Returns:
    -------
    np.ndarray
        A sequence of gamma values.
    """
    p = S.shape[1]  # Number of columns in S
    
    if minv is None:
        minv = 0
    
    if maxv is None:
        l1 = np.max(np.linalg.eigvalsh(realsym(S)))
        maxv = l1 * (p / m)
    
    if skew is None:
        skew = 1
    
    step = 1 / (nsol - 1)
    lx = (1 - np.exp(skew * np.arange(0, 1 + step, step))) / (1 - np.exp(skew))
    gamma_seq = (1 - lx) * minv + lx * maxv
    
    return gamma_seq

def cv_gamma_c(x_c: np.ndarray, option: dict) -> dict:
    """
    Cross-validation to select gamma parameters.

    Parameters:
    ----------
    x_c : np.ndarray
        Centered data.
    option : dict
        Configuration options.

    Returns:
    -------
    dict
        Dictionary containing selected gamma values.
    """
    # wrapper for the detailed implementation of this function
    return fun_GetCov_SeqADMM_SelectTuningPar.CV_Gamma_c(x_c, option)

def deflate(S: np.ndarray, PrevPi: np.ndarray = None) -> np.ndarray:
    """
    Perform matrix deflation to adjust the covariance matrix.

    Parameters:
    ----------
    S : np.ndarray
        The covariance matrix to be deflated.
        
    PrevPi : np.ndarray, optional
        The deflation matrix. If None, no deflation is performed.

    Returns:
    -------
    np.ndarray
        The deflated covariance matrix.
    """
    if PrevPi is not None:
        p = S.shape[0]
        I = np.eye(p)
        S = (I - PrevPi) @ S @ (I - PrevPi)
        # Alternate version listed in the R code:
        # S -= PrevPi @ S @ PrevPi

    return S

def generate_rho2(S: np.ndarray, nsol: int, minv: float = None, maxv: float = None, skew: float = None) -> np.ndarray:
    """
    Generate sequences of alpha and lambda values.

    Parameters:
    ----------
    S : np.ndarray
        Input matrix.
    nsol : int
        Number of solutions in the sequence.
    minv : float, optional
        Minimum value of the sequence. Defaults to 0.
    maxv : float, optional
        Maximum value of the sequence. Defaults to the 95th percentile of abs(S) excluding diagonal.
    skew : float, optional
        Skewness factor for the sequence generation. Defaults to 1.

    Returns:
    -------
    np.ndarray
        Generated rho2 sequence.
    """
    if minv is None:
        minv = 0

    if maxv is None:
        p = S.shape[0]
        v = np.abs(S) + np.diag(np.full(p, np.nan))
        maxv = np.nanpercentile(v.flatten(), 95)
    
    if maxv < 10:
        maxv *= 4   ## this was newly added in the R code, 4/26

    if skew is None:
        skew = 1

    step = 1 / (nsol - 1)
    lx = (1 - np.exp(skew * np.linspace(0, 1, step))) / (1 - np.exp(skew))
    rho2_seq = (1 - lx) * minv + lx * maxv

    return rho2_seq

def fve_alpha_lambda_c(K: np.ndarray, G: np.ndarray, alpha_seq: np.ndarray, lambda_seq: np.ndarray, totV: float,
                       Fantope_d: int, prevPi_d: int, option: dict, select: str, prevPi: np.ndarray) -> dict:
    """
    Function to select alpha and lambda based on FVE (Fraction of Variance Explained).

    Parameters:
    ----------
    K : np.ndarray
        Covariance matrix.
    G : np.ndarray
        The G matrix.
    alpha_seq : np.ndarray
        Sequence of alpha values.
    lambda_seq : np.ndarray
        Sequence of lambda values.
    totV : float
        Total variance.
    Fantope_d : int
        Fantope dimensionality.
    prevPi_d : int
        Previous Pi dimensionality.
    option : dict
        Configuration options.
    select : str
        Selection criteria ('w' or 'z').
    prevPi : np.ndarray
        Previous Pi values.

    Returns:
    -------
    dict
        Dictionary containing selected alpha and lambda.
    """
    # Wrapper for the detailed implementation of this function
    return fun_GetCov_SeqADMM_SelectTuningPar.FVE_AlphaLambda_c(K, G, alpha_seq, lambda_seq, totV,
                           Fantope_d, prevPi_d, option, select, prevPi)

def seq_admm_c(K: np.ndarray, case: int, prevPi_d: int, alpha: float, lambda_val: float, option: dict, prevPi: np.ndarray,
               verbose: bool) -> np.ndarray:
    """
    Solve the ADMM problem for a given case.

    Parameters:
    ----------
    K : np.ndarray
        The covariance matrix.
    case : int
        The case number.
    prevPi_d : int
        Previous Pi dimensionality.
    alpha : float
        The alpha parameter.
    lambda_val : float
        The lambda parameter.
    option : dict
        Configuration options.
    prevPi : np.ndarray
        Previous Pi values.
    verbose : bool
        Verbosity flag.

    Returns:
    -------
    np.ndarray
        The solution matrix.
    """
    # Wrapper for the detailed implementation of this function
    return fun_GetCov_SeqADMM_SelectTuningPar.seqADMM_c(K, case, prevPi_d, 
                alpha, lambda_val, option, prevPi, verbose)

def lvpca(data: np.ndarray, m: int, L1: int, L2: int = 1, model: str = "1Way", k: int = None,
          rFVEproportion: float = None, FVE_threshold: float = 0.85, correlation: bool = True,
          corr_rho: np.ndarray = None, corr_uncorrpct: float = 0.2, gammaSeq_list: list = None,
          gamma_list: list = None, lambda_list: list = None, alpha_list: list = None,
          lambdaSeq_list: list = None, alphaSeq_list: list = None, SmoothD: str = "2Diff", nsol: int = 10,
          nfold: int = 5, FVE_k: int = 10, maxiter: int = 100, maxiter_cv: int = 20, eps: float = 0.01,
          verbose: bool = False) -> dict:
    """
    Perform multilevel multivariate functional data analysis using the LVPCA method.

    Parameters:
    ----------
    data : np.ndarray
        The input data matrix.
    m : int
        The number of variables.
    L1 : int
        The number of observations for the first level.
    L2 : int
        The number of observations for the second level.
    model : str
        The model type ('1Way' or '2WayNested').
    k : int
        The number of components.
    rFVEproportion : float
        The proportion of FVE (Fraction of Variance Explained).
    FVE_threshold : float
        The threshold for FVE.
    correlation : bool
        Whether to consider correlation.
    corr_rho : np.ndarray
        Correlation values.
    corr_uncorrpct : float
        Uncorrelated percentage.
    gammaSeq_list : list
        List of gamma sequences.
    gamma_list : list
        List of gamma values.
    lambda_list : list
        List of lambda values.
    alpha_list : list
        List of alpha values.
    lambdaSeq_list : list
        List of lambda sequences.
    alphaSeq_list : list
        List of alpha sequences.
    SmoothD : str
        The smoothing method.
    nsol : int
        The number of solutions.
    nfold : int
        The number of folds for cross-validation.
    FVE_k : int
        The maximum number of iterations for FVE.
    maxiter : int
        The maximum number of iterations.
    maxiter_cv : int
        The maximum number of iterations for cross-validation.
    eps : float
        The convergence threshold.
    verbose : bool
        Whether to print progress.

    Returns:
    -------
    dict
        A dictionary with the LVPCA analysis results.
    """
    
    ### DATA INITIALIZATION
    
    x = np.array(data)
    n, p = x.shape # Numper of observations (n) vs the number of total time points (p)
    p_m = np.repeat(p // m, m) # vector of number of time points for each variable
    t_x = (np.arange(p)+1)/p

    option = {
                "m": m, 
                "L1": L1, 
                "L2": L2, 
                "model": model, 
                "k": k, 
                "rFVEproportion": rFVEproportion,
                "FVE_threshold": FVE_threshold, 
                "corr_rho": corr_rho, 
                "corr_uncorrpct": corr_uncorrpct,
                "gammaSeq_list": gammaSeq_list, 
                "gamma_list": gamma_list, 
                "lambda_list": lambda_list,
                "alpha_list": alpha_list, 
                "lambdaSeq_list": lambdaSeq_list, 
                "alphaSeq_list": alphaSeq_list,
                "SmoothD": SmoothD, 
                "nsol": nsol, 
                "nfold": nfold, 
                "FVE_k": FVE_k, 
                "maxiter": maxiter,
                "maxiter_cv": maxiter_cv, 
                "eps": eps, 
                "PrevPi": None, 
                "PrevPi_d": None
                }

    # If univariate, then only perform localized FPCA by restrict alpha=0
    if option["m"] == 1:
      if option["k"] is None:
        option["alpha_list"] = 0
      else:
          option["alpha_list"] = np.repeat(0,k)
    
    # If model is "1Way", transform tuning parameters  
    adjustments = {
                "gammaSeq_list": {
                    "gammaSeq_w": np.repeat(0, 10),
                    "gammaSeq_z": option["gammaSeq_list"]
                    },
                "gamma_list": {
                    "gamma_w": 0,
                    "gamma_z": option["gamma_list"]
                    },
                "lambda_list": {
                    "lambda_w": option["lambda_list"],
                    "lambda_z": option["lambda_list"]
                    },
                "alpha_list": {
                    "alpha_w": option["alpha_list"],
                    "alpha_z": option["alpha_list"]
                    },
                "lambdaSeq_list": {
                    "lambdaSeq_w": option["lambdaSeq_list"],
                    "lambdaSeq_z": option["lambdaSeq_list"]
                    },
                "alphaSeq_list": {
                    "alphaSeq_w": option["alphaSeq_list"],
                    "alphaSeq_z": option["alphaSeq_list"]
                    }
                }
    
    if model == "1Way":
        for v in adjustments.keys():
            if option.get(v) is not None:
                option[v] = adjustments[v]
                
    # If don't consider correlation, will force correlation matrix to be diagonal matrix
    if not correlation:
        option["corr_rho"] = np.diag(np.ones(L2))
  
    ### LVPCA: PREPROCESSING
    
    # get S = xcov - D*Rho_1
    if option["SmoothD"] == "2Diff":
        option["SmoothD"] = get_smooth_D_0610(p, m) # <----------------------------
    
    # center data
    eta = np.zeros((L2, p))
    for j in range(L2):
        eta[j, :] = np.mean(x[(np.arange(L1) * L2) + j, :], axis=0)
    
    x_c = np.zeros((n, p))
    for j in range(L2):
        x_c[(np.arange(L1) * L2) + j, :] = x[(np.arange(L1) * L2) + j, :] - eta[j, :].T
        
    # get level specific covariance
    G = multilevel_s_c(x_c, L1, L2, option)
    G_w = G["G_w"]
    G_z = G["G_z"]
    option["c"] = G["c"]
    if model=="2WayNested":
        option["corr_rho"] = G["h_w_sum"]
    option["F_hat"] = G["h_w_sum"]
    if option["gamma_list"] is none:
        if option["gammaSeq_list"] is none:
            option["gammaSeq_list"] = {"gammaSeq_w":generate_gamma(G_w,m,nsol,None,None,None),
                                   "gammaSeq_z":generate_gamma(G_z,m,nsol,None,None,None)} # get candidate gamma
        else:
            option["gamma_list"] = cv_gamma_c(x_c,option) # get gamma
    K_w = G_w-option["gamma_list"]["gamma_w"]*option["SmoothD"]                                     # get S=xcov-gamma*D
    K_z = G_z-option["gamma_list"]["gamma_z"]*option["SmoothD"]   
    
    # get totV
    d_w = np.linalg.eigvalsh(K_w)
    d_w = np.real(d_w)
    totV_w = np.sum(d_w[d_w > 0])
     
    d_z = np.linalg.eigvalsh(K_z)
    d_z = np.real(d_z)
    totV_z = np.sum(d_z[d_z > 0])
      
    # get alpha and lambda and enter into the algorithm
    vec_w = np.zeros((p, p))
    alpha_w = np.zeros(p)
    lambda_w = np.zeros(p)
    FVE_w = np.zeros(p)
    PrevPi_w = None
    alphaSeq_w = []
    lambdaSeq_w = []
      
    vec_z = np.zeros((p, p))
    alpha_z = np.zeros(p)
    lambda_z = np.zeros(p)
    FVE_z = np.zeros(p)
    PrevPi_z = None
    alphaSeq_z = []
    lambdaSeq_z = []
      
    ki = 0
    cont = 1
      
    while cont > 0 and ki < option["FVE_k"]:
        ki += 1
        # print(f"PC {ki} start")
        
        if option.get("alpha_list") is not None and option.get("lambda_list") is not None:
            if option.get("k") is not None:
                alpha_w[ki] = option["alpha_list"]["alpha_w"][ki]
                lambda_w[ki] = option["lambda_list"]["lambda_w"][ki]
                alpha_z[ki] = option["alpha_list"]["alpha_z"][ki]
                lambda_z[ki] = option["lambda_list"]["lambda_z"][ki]
            else:
                alpha_w[ki] = option["alpha_list"]["alpha_w"]
                lambda_w[ki] = option["lambda_list"]["lambda_w"]
                alpha_z[ki] = option["alpha_list"]["alpha_z"]
                lambda_z[ki] = option["lambda_list"]["lambda_z"]
          
        elif option.get("alpha_list") is None or option.get("lambda_list") is None:
            
            # get alpha sequence
            
            if option.get("alphaSeq_list") is not None:
                if option.get("k") is not None:
                    alphaSeq_w.append(option["alphaSeq_list"]["alphaSeq_w"][ki])
                    alphaSeq_z.append(option["alphaSeq_list"]["alphaSeq_z"][ki])
                else:
                    alphaSeq_w.append(option["alphaSeq_list"]["alphaSeq_w"])
                    alphaSeq_z.append(option["alphaSeq_list"]["alphaSeq_z"])
            else:
                Gnow_w = deflate(G_w, PrevPi_w)  # deflate, make the range narrower <<< CHECK THIS
                alphaSeq_w.append(GenerateRho2(Gnow_w, option["nsol"], None, None, None))
                Gnow_z = deflate(G_z, PrevPi_z)  # deflate, make the range narrower
                alphaSeq_z.append(GenerateRho2(Gnow_z, option["nsol"], None, None, None))
    
            alphaSeqnow_w = alphaSeq_w[ki]
            alphaSeqnow_z = alphaSeq_z[ki]
    
            # get lambda sequence
            
            if option.get("lambdaSeq_list") is not None:
                if option.get("k") is not None:
                    lambdaSeq_w.append(option["lambdaSeq_list"]["lambdaSeq_w"][ki])
                    lambdaSeq_z.append(option["lambdaSeq_list"]["lambdaSeq_z"][ki])
                else:
                    lambdaSeq_w.append(option["lambdaSeq_list"]["lambdaSeq_w"])
                    lambdaSeq_z.append(option["lambdaSeq_list"]["lambdaSeq_z"])
            else:
                lambdaSeq_w.append(alphaSeq_w[ki])
                lambdaSeq_z.append(alphaSeq_z[ki])
            
            lambdaSeqnow_w = lambdaSeq_w[ki]
            lambdaSeqnow_z = lambdaSeq_z[ki]
    
            # get alpha and lambda
            
            if option.get("rFVEproportion") is not None:
                if model == "1Way":
                    FVEchoice_w = {"alpha1": 0, "lambda1": 0}
                else:
                    FVEchoice_w = fve_alpha_lambda_c(K_w, G_w, alphaSeqnow_w, lambdaSeqnow_w, totV_w,
                                                    Fantope_d=1, PrevPi_d=(ki-1), option=option, select="w", PrevPi=PrevPi_w)
                FVEchoice_z = fve_alpha_lambda_c(K_z, G_z, alphaSeqnow_z, lambdaSeqnow_z, totV_z,
                                                Fantope_d=1, PrevPi_d=(ki-1), option=option, select="z", PrevPi=PrevPi_z)
                alpha_w[ki] = FVEchoice_w["alpha1"]
                lambda_w[ki] = FVEchoice_w["lambda1"]
                alpha_z[ki] = FVEchoice_z["alpha1"]
                lambda_z[ki] = FVEchoice_z["lambda1"]
                if not MUTE:
                    if model == "1Way":
                        print(f"Choice by FVE: alpha={alpha_z[ki]}, lambda={lambda_z[ki]} for PC {ki}")
                    else:
                        print(f"Z level choice by FVE: alpha={alpha_z[ki]}, lambda={lambda_z[ki]} \t"
                              f"W level choice by FVE: alpha={alpha_w[ki]}, lambda={lambda_w[ki]} for PC {ki}")
            else:
                CVchoice = CV_AlphaLambda_c(x_c, alphaSeqnow_w, lambdaSeqnow_w, alphaSeqnow_z, lambdaSeqnow_z,
                                            Fantope_d=1, PrevPi_d=(ki-1), option=option, PrevPi_w=PrevPi_w, PrevPi_z=PrevPi_z)
                alpha_w[ki] = CVchoice["alpha1_w"]
                lambda_w[ki] = CVchoice["lambda1_w"]
                alpha_z[ki] = CVchoice["alpha1_z"]
                lambda_z[ki] = CVchoice["lambda1_z"]
                if not MUTE:
                    if model == "1Way":
                        print(f"Choice by CV: alpha={alpha_z[ki]}, lambda={lambda_z[ki]} for PC {ki}")
                    else:
                        print(f"Z level choice by CV: alpha={alpha_z[ki]}, lambda={lambda_z[ki]} \t"
                              f"W level choice by CV: alpha={alpha_w[ki]}, lambda={lambda_w[ki]} for PC {ki}")

        if model == "1Way":
            # w
            projH_w = K_w  # get H matrix (case1)
            # z
            projH_z = seq_admm_c(K_z, 1, ki - 1, alpha_z[ki], lambda_z[ki], option, PrevPi_z, verbose)  # get H matrix (case1)
        elif model == "2WayNested":
            # w
            projH_w = seq_admm_c(K_w, 1, ki - 1, alpha_w[ki], lambda_w[ki], option, PrevPi_w, verbose)  # get H matrix (case1)
            # z
            projH_z = seq_admm_c(K_z, 1, ki - 1, alpha_z[ki], lambda_z[ki], option, PrevPi_z, verbose)  # get H matrix (case1)
        
        # update parameters
        vec_w = np.linalg.eigh(projH_w)[1][:, 0]  # update vec (eigenvector in the ki-th PC)
        if vec_w[np.argmax(np.abs(vec_w))] < 0:
            vec_w = -vec_w  # Decide the sign for eigenvector (rule: the largest abs needs to be positive)
        
        vec_z = np.linalg.eigh(projH_z)[1][:, 0]  # update vec (eigenvector in the ki-th PC)
        if vec_z[np.argmax(np.abs(vec_z))] < 0:
            vec_z = -vec_z  # Decide the sign for eigenvector (rule: the largest abs needs to be positive)
        
        if model == "1Way":
            FVE_w = 0
        else:
            FVE_w = np.sum(np.diag(vec_w.T @ G_w @ vec_w)) / totV_w  # update FVE (fraction of variance explained at ki-th PC)
        
        FVE_z = np.sum(np.diag(vec_z.T @ G_z @ vec_z)) / totV_z  # update FVE (fraction of variance explained at ki-th PC)
        
        if FVE_z < 0:
            ki_z_pos = max((i for i in range(1, ki+1) if FVE_z > 0), default=0)  # Since G.z is consistent up to Sigma-(1-1/c)*noise, it may not be positive definite
            FVE_z = 0  # which means eigenvalue/FVE[ki] can be<0. When that occurs, force FVE[ki]=0, and toss PCs from then.
        else:
            ki_z_pos = ki  # When the loop ends, instead of keep ki for z level, keep ki.z.pos for z level.
        
        if option.get('k') is None:  # update cont, stop criterion
            FVE_tot = (np.sum(FVE_w * totV_w + FVE_z * totV_z)) / (totV_w + totV_z)
            cont = 0 if FVE_tot > option.get('FVE_threshold', 0) else 1
        elif ki == option['k']:
            cont = 0
        else:
            cont = 1
        
        PrevPi_w = vec_w[:, np.newaxis] @ vec_w[np.newaxis, :]  # update PrevPi (projection matrix Pi)
        PrevPi_z = vec_z[:, np.newaxis] @ vec_z[np.newaxis, :]  # update PrevPi (projection matrix Pi)
                    
        if model == "2WayNested":
            # Prepare parameters
            k_w = ki
            k_z = ki_z_pos
            
            # Prepare small modules
            eigVec_w = vec_w[:, :k_w]  # T x 4
            eigVec_z = vec_z[:, :k_z]
            h = (np.ptp(t_x)) / (p - 1)
            phi_w = eigVec_w / np.sqrt(h)
            phi_z = eigVec_z / np.sqrt(h)
            eigValue_w = np.diag(eigVec_w.T @ G_w @ eigVec_w) * h
            eigValue_z = np.diag(eigVec_z.T @ G_z @ eigVec_z) * h
    
            npositive = np.sum(np.linalg.eigvals(K_w) > 0)
            noise = (np.sum(np.linalg.eigvals(G_w)) - np.sum(np.linalg.eigvals(K_w)[:npositive])) / p * option["c"]
    
            # Prepare large modules
            Gi1 = np.diag(eigValue_z)
            Gi2 = kron(np.diag(eigValue_w), option["corr_rho"])
            Gi = block_diag(Gi1, Gi2)
            
            Zi1 = kron(np.ones(L2), phi_z)
            Zi2 = np.hstack([kron(np.eye(L2), phi_w[:, x]) for x in range(k_w)])
            Zi = np.hstack([Zi1, Zi2])
            Ri = np.eye(L2 * p) * noise
            Vi = Zi @ Gi @ Zi.T + Ri
            GZVi = Gi @ Zi.T @ solve(Vi, np.eye(Vi.shape[0]))
    
            # Calculate PC scores and predict time series
            PCscore = np.full((x_c.shape[0], k_z + k_w), np.nan)
            predx = np.full((x_c.shape[0], p), np.nan)
            predx_margin = np.full((x_c.shape[0], p), np.nan)
            
            for i in range(L1):
                xi_c = x_c[i * L2:(i + 1) * L2, :].T.flatten()
                PCscorei_long = GZVi @ xi_c
                
                PCscorei = np.hstack([
                    np.kron(np.ones((L2, 1)), PCscorei_long[:k_z].T),
                    PCscorei_long[k_z:].reshape(L2, -1)
                ])
                
                PCscore[i * L2:(i + 1) * L2, :] = PCscorei
                predx[i * L2:(i + 1) * L2, :] = eta + (np.hstack([phi_z, phi_w]) @ PCscorei.T).T
                mu = np.mean(eta, axis=0)
                predx_margin[i * L2:(i + 1) * L2, :] = np.kron(np.ones((L2, 1)), mu.T).T + (phi_z @ PCscorei[:, :k_z].T).T
            
            PCscore_z = PCscore[:, :k_z]
            PCscore_w = PCscore[:, k_z:]
    
            # Prepare output
            FVE_w = option['FVE_w'][:k_w]
            FVE_z = option['FVE_z'][:k_z]
            totV_w = option['totV_w']
            totV_z = option['totV_z']
            totVprop_w = totV_w / (totV_w + totV_z)
            totVprop_z = totV_z / (totV_w + totV_z)
            FVE_tot = (np.sum(FVE_w * totV_w) + np.sum(FVE_z * totV_z)) / (totV_w + totV_z)
            
            option.update({
                "k": {"k_w": k_w, "k_z": k_z},
                "alpha_list": {"alpha_w": option['alpha_w'][:k_w], "alpha_z": option['alpha_z'][:k_z]},
                "lambda_list": {"lambda_w": option['lambda_w'][:k_w], "lambda_z": option['lambda_z'][:k_z]},
                "alphaSeq_list": {"alphaSeq_w": option['alphaSeq_w'], "alphaSeq_z": option['alphaSeq_z']},
                "lambdaSeq_list": {"lambdaSeq_w": option['lambdaSeq_w'], "lambdaSeq_z": option['lambdaSeq_z']},
                "noise": noise,
                "G_w": G_w,
                "G_z": G_z,
                "K_w": K_w,
                "K_z": K_z
            })
            
            return {
                'predx': predx,
                'predx_margin': predx_margin,
                'PCscore_w': PCscore_w,
                'PCscore_z': PCscore_z,
                'FVE_w': FVE_w,
                'eigValue_w': eigValue_w,
                'phi_w': phi_w,
                'totVprop_w': totVprop_w,
                'FVE_z': FVE_z,
                'eigValue_z': eigValue_z,
                'phi_z': phi_z,
                'totVprop_z': totVprop_z,
                'FVE_tot': FVE_tot,
                'option': option
            }
        
        elif model == "1Way":
            k = ki
            eigVec = vec_z[:, :k]
            projH = eigVec @ eigVec.T
            FVE = option['FVE_z'][:k]
            totV = option['totV_z']
            eigValue = FVE * totV
            option.update({
                "k": k,
                "alpha": option['alpha_z'][:k],
                "lambda": option['lambda_z'][:k],
                "alphaSeq": option['alphaSeq_z'],
                "lambdaSeq": option['lambdaSeq_z']
            })
            
            h = (np.ptp(t_x)) / (p - 1)
            phi = eigVec / np.sqrt(h)
            PCscore = np.zeros((x_c.shape[0], k))
            
            for i in range(k):
                prod = x_c * np.tile(phi[:, i], (x_c.shape[0], 1))
                PCscore[:, i] = np.array([trapz(t_x, row) for row in prod])
            
            predx = np.tile(eta, (x_c.shape[0], 1)) + PCscore @ phi.T
            
            return {
                'predx': predx,
                'PCscore': PCscore,
                'phi': phi,
                'FVE': FVE,
                'eigValue': eigValue,
                'option': option
            }

def main() -> None:
    """
    The main=.
    """
    # Example data
    data = np.random.randn(100, 20)

    # Perform LVPCA
    result = lvpca(data, m=2, L1=50, L2=2, model="1Way")
    
    if not MUTE:
        print("LVPCA analysis result:", result)
    
#%%##|    \###/    |#####/   \#####|_    _|#|    \##|  |################
#####|     \#/     |####/     \######|  |###|     \#|  |################
#####|  |\     /|  |###/  /#\  \#####|  |###|  |\  \|  |################
#####|  |#\   /#|  |##/   ___   \####|  |###|  |#\     |################
#####|__|##\ /##|__|#/__/#####\__\#|______|#|__|##\____|################

if __name__ == "__main__":
    """
    Main code idiom
    """
    main()
