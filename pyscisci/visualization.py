
import datetime
import matplotlib.pylab as plt

def career_timeline(impact_df, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize(10,6))

    for d, c in zip(citedf['sortdate'].values, citedf['totalcite'].values):
        if 'T' in d:
            d= datetime.datetime.strptime(d.split('T')[0], '%Y-%m-%d')
        else:
            d= datetime.datetime.strptime(d.split(' ')[0], '%Y-%m-%d')

        ax.plot([d]*2, [0,totalcite], c = 'black', lw = 0.5) 
        ax.scatter(d, totalcite,  c = 'orange',edgecolor='black', linewidth='0.5', zorder=100)