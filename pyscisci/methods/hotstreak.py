# -*- coding: utf-8 -*-
"""
.. module:: hotstreak
    :synopsis: Calculate the career hotstreak.

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import sys
import pandas as pd
import numpy as np

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from pyscisci.utils import hard_rolling_window


def piecewise_step(x, a,b,c,d):
    return np.piecewise(x, [x<a, np.logical_and(x>=a,x<=b), x>b], [c,d,c])

def piecewise_step_err(y, loc1, loc2):
    y1 = y[loc1:(loc2+1)]
    y0 = np.hstack([y[:loc1], y[(loc2+1):]])
    return np.sqrt(np.sum([np.sum((a - a.mean())**2) for a in [y0,y1]]))

def brut_fit_piecewise_step(ys):
    locs = np.vstack(np.triu_indices(ys.shape[0]-1, k=1)).T + 1
    errs = [piecewise_step_err(ys, i, j) for i,j in locs]
    
    return np.min(errs), locs[np.argmin(errs)]


def piecewise_step_err2(y, loc1, loc2, loc3, loc4):
    y1 = y[loc1:(loc2+1)]
    y2 = y[loc3:(loc4+1)]
    y0 = np.hstack([y[:loc1], y[(loc2+1):loc3], y[(loc4+1):]])
    return np.sqrt(np.sum([np.sum((a - a.mean())**2) for a in [y0,y1,y2]]))

def brut_fit_piecewise_step2(ys):
    locs1 = np.vstack(np.triu_indices(ys.shape[0]-3, k=1)).T + 1
    locs2 = np.array([[i,j,j+k,j+m] for i,j in locs1 for k,m in np.vstack(np.triu_indices(ys.shape[0]-j-1, k=1)).T])
    errs = [piecewise_step_err2(ys, i, j,k,m) for i,j,k,m in locs2]
    
    return np.min(errs), locs2[np.argmin(errs)]

def career_hotstreak(author_career_df, citecol='c10', maxk=1, l1_lambda = 1.):
    """
    Identify hot streaks in author careers :cite:`liu2018hotstreak'.

    TODO: this is an interger programming problem.  Reimplement using an interger solver.
    Right now just using a brut force search (very inefficient)!
    
    Parameters
    ----------
    author_career_df : DataFrame
        The author publication history.

    citecol : str, default 'c10'
        The column with publication citation information.
    
    max_k : int, default 1
        The maximum number of hot streaks to search for in a career. Should be 1 or 2.
    
    l1_lambda : float, default 1.0
        The l1 regularization for the number of streaks.  
        Note, the authors never define the value they used for this in the SI.
        
    Returns
    ----------
    lsm_err : float
        The least square mean error of the model plus the l1-regularized term for the number of model coefficients.

    streak_loc : array
        The index locations for the hot streak start and end locations.
    """
    if maxk == 0 or maxk > 2:
        raise NotImplementedError("the career hotstreak is not implemented for this number of streaks.  set maxk = 1 or maxk=2 ")

    Delta_N = max(5, int(0.1*author_career_df.shape[0]))
    gamma_N = hard_rolling_window(np.log10(author_career_df[citecol].values), window=Delta_N, step_size = 1).mean(axis=1)
    gamma_N = gamma_N[int(Delta_N/2):-int(Delta_N/2)]
    
    nostreak_err = np.sqrt(np.sum((gamma_N - gamma_N.mean())**2)) + l1_lambda # no step functions uses 1 model coefficient
    streak_gammas = [gamma_N.mean(), None, None]

    streak_err, streak_loc1 = brut_fit_piecewise_step(gamma_N)
    streak_err += 3*l1_lambda # 1 step function = 3 model coefficients
    
    streak_loc = [None]*4

    if (nostreak_err <= streak_err): 
        streak_err = nostreak_err
        nstreak = 0
    else:
        streak_loc[:2] = streak_loc1 + int(Delta_N/2)
        streak_gammas[0] = np.hstack([gamma_N[:streak_loc1[0]], gamma_N[(streak_loc1[1]+1):]]).mean()
        streak_gammas[1] = gamma_N[streak_loc1[0]:(streak_loc1[1]+1)].mean()
        nstreak = 1

    if maxk == 2:
        streak_err2, streak_loc2 = brut_fit_piecewise_step2(gamma_N)
        streak_err2 += 6*l1_lambda # 2 step functions = 6 model coefficients
        if (streak_err > streak_err2): 
            streak_err, streak_loc = streak_err2, list(streak_loc2 + int(Delta_N/2))
            streak_gammas[0] = np.hstack([gamma_N[:streak_loc1[0]], gamma_N[streak_loc1[1]:(streak_loc1[2])], gamma_N[(streak_loc1[3]+1):]]).mean()
            streak_gammas[1] = gamma_N[streak_loc1[0]:(streak_loc1[1]+1)].mean()
            streak_gammas[2] = gamma_N[streak_loc1[2]:(streak_loc1[3]+1)].mean()
            nstreak = 2
    
    solution_df = [[streak_gammas[0], streak_gammas[1], streak_loc[0], streak_loc[1]]]
    if nstreak == 2:
        solution_df += [streak_gammas[0], streak_gammas[2], streak_loc[3], streak_loc[4]]
    return pd.DataFrame(solution_df, columns = ['Baseline', 'StreakGamma', 'StreakStart', 'StreakEnd'])
