
import datetime
import numpy as np
import matplotlib.pylab as plt
from pyscisci.metrics.productivitytrajectory import piecewise_linear

def career_impacttimeline(impact_df, datecol = 'Date', impactcol='Ctotal', fill_color='orange', edge_color='k', ax=None):

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10,6))

    for d, c in zip(impact_df[datecol].values, impact_df[impactcol].values):
        if isinstance(d, str):
            if 'T' in d:
                d= datetime.datetime.strptime(d.split('T')[0], '%Y-%m-%d')
            else:
                d= datetime.datetime.strptime(d.split(' ')[0], '%Y-%m-%d')

        ax.plot([d]*2, [0,c], c='k', lw = 0.5) 
        ax.scatter(d, c, color=fill_color, edgecolor=edge_color, linewidth=0.5, zorder=100)
        
    return ax


def career_productivitytimeline(yearlyprod_df, productivity_trajectory = None, datecol = 'Year', fill_color='blue', ax=None):

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10,6))

    ax.bar(yearlyprod_df[datecol].values, yearlyprod_df['YearlyProductivity'].values, color=fill_color) 
    
    if not productivity_trajectory is None:
        t_break, b, m1, m2 = productivity_trajectory[['t_break','b','m1','m2']].values[0]

        ts = np.arange(yearlyprod_df[datecol].min(), yearlyprod_df[datecol].max()+1)
        ax.plot(ts, piecewise_linear(ts, t_break, b, m1, m2), color='black')
        
    return ax

def hex2rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16)/255. for i in range(0, lv, lv // 3))

def hex2rgba(value, alpha = 1):
    return hex2rgb(value) + (alpha,)