
import datetime
import numpy as np

import matplotlib.pylab as plt
import matplotlib.colors as colors

from pyscisci.methods.productivitytrajectory import piecewise_linear

def career_impacttimeline(impact, datecol = 'Date', impactcol='Ctotal', fill_color='orange', hot_streak_info = None, 
    edge_color='k', streak_color='firebrick', ax=None):

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10,6))

    if isinstance(datecol, str):
        dates = impact[datecol].values
    elif isinstance(datecol, np.ndarray):
        dates = datecol

    if isinstance(impactcol, str):
        cites = impact[impactcol].values
    elif isinstance(impactcol, np.ndarray):
        cites = impactcol

    for d, c in zip(dates, cites):
        if isinstance(d, str):
            if 'T' in d:
                d= datetime.datetime.strptime(d.split('T')[0], '%Y-%m-%d')
            else:
                d= datetime.datetime.strptime(d.split(' ')[0], '%Y-%m-%d')

        ax.plot([d]*2, [0,c], c='k', lw = 0.5) 
        ax.scatter(d, c, color=fill_color, edgecolor=edge_color, linewidth=0.5, zorder=100)
    
    if not hot_streak_info is None:
        nstreaks = 2 - hot_streak_info[:3].isnull().sum()
        
        if nstreaks == 1:
            gamma0, gamma1 = 10**(hot_streak_info[:2])
            streak_start, streak_end = hot_streak_info[3:5].astype(int)
            ax.plot(datecol[:streak_start], [gamma0]*streak_start, c=streak_color)
            ax.plot(datecol[streak_start:(streak_end+1)], [gamma1]*(streak_end-streak_start+1), c=streak_color)
            ax.plot(datecol[(streak_end+1):], [gamma0]*(datecol.shape[0]-streak_end-1), c=streak_color)

    return ax


def career_productivitytimeline(yearlyprod, productivity_trajectory = None, datecol = 'Year', fill_color='blue', ax=None):

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10,6))

    ax.bar(yearlyprod[datecol].values, yearlyprod['YearlyProductivity'].values, color=fill_color) 
    
    if not productivity_trajectory is None:
        t_break, b, m1, m2 = productivity_trajectory[['t_break','b','m1','m2']].values[0]

        ts = np.arange(yearlyprod[datecol].min(), yearlyprod[datecol].max()+1)
        ax.plot(ts, piecewise_linear(ts, t_break, b, m1, m2), color='black')
        
    return ax

def hex2rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16)/255. for i in range(0, lv, lv // 3))

def hex2rgba(value, alpha = 1):
    return hex2rgb(value) + (alpha,)

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))