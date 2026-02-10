import numpy as np 
import pandas as pd

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

def pvalue2stars(pvalue):
    if pvalue < 0.001:
        return "^{***}"
    elif pvalue < 0.01:
        return "^{**}"
    elif pvalue < 0.05:
        return "^{*}"
    else:
        return ""


def table_row(varname='const', offset=0, roun=2, modellist=[], namedict={},i=0,j=1):
    
    row_text = namedict[varname] + " & "*(offset+1) + " & ".join([ "$" + str(model.params[i][varname].round(roun)) + pvalue2stars(model.pvalues[i][varname]) + "$" for model in modellist[offset:]]) + "\\\\ \n"

    row_text += " & "*(offset+1) + " & ".join([ "$(" + str(model.conf_int().loc[str(j)].loc[varname]['lower'].round(roun)) + "," + str(model.conf_int().loc[str(j)].loc[varname]['upper'].round(roun)) + ")$" for model in modellist[offset:]]) + "\\\\ \n"
    row_text +=  " & "*(offset+1) + " & ".join([ "S.E. $" + str(model.bse[i][varname].round(roun)) + "$; p-v $" +str(model.pvalues[i][varname].round(4)) + "$"  for model in modellist[offset:]]) + "\\\\ [0.8ex]  \n"

      #row_text +=  " & "*(offset+1) + " & ".join([ "std. err. $" + str(model.std_errors[varname].round(roun)) + "$" for model in modellist[offset:]]) + "\\\\ [0.8ex] \n"
    return row_text

def make_multinomial_latex_table(fit_models, exog_var_sets = [], dep_var = "", namedict = {}, caption_text=""):
    
    Nmodels = len(exog_var_sets)
    table_text= """{\\tiny
    \\begin{longtable}{p{0.2\\linewidth}"""+"p{0.12\\linewidth}"*Nmodels+"}"+"""\\caption{\\textbf{Fixed-effect multinomial logit regression.} Model coefficients labelled by $p$-value. Standard errors in parentheses.} 
      \\label{table:multinomialfull} \\\\
      \\hline \\hline \\\\
    \\multicolumn{"""+str(Nmodels+1)+"""}{c}{\\textbf{Dependent variable: Citation preference}} \\\\ \\hline 
     & \\multicolumn{"""+str(Nmodels)+"""}{c}{Model}  \\\\"""
    
    table_text += "\cline{2-" + str(Nmodels + 1) + "}"

    for icol in range(Nmodels):
        table_text += "& (" + str(icol + 1) + ")"
    
    
    table_text += """\\\\[0.8ex]
    \\hline
    \\endfirsthead"""

    table_text += """\\multicolumn{2}{c}%
      {{\\tablename\\ \\thetable{} -- continued from previous page}} \\\\
      \\hline \\\\"""

    for icol in range(Nmodels):
        table_text += "& (" + str(icol + 1) + ")"

    table_text += """\\\\
      \\hline
      \\endhead """

    table_text += "\\hline"+"&"*(Nmodels+1-2)+ """\\multicolumn{2}{r}{{Continued on next page}} \\\\ \\endfoot
    \\hline
    \\caption*{} \\\\
    \\endlastfoot"""
    
    table_text += """$\\mathbf{Citation~Preference: Positive}$ & & & & & \\\\ [1.8ex]"""

    # add constant
    table_text += table_row(varname='const', offset=0, roun=2, modellist=fit_models, namedict=namedict,i=0,j=1)
    
    for offset, varlist in enumerate(exog_var_sets):
        for var in varlist:
            table_text += table_row(varname=var, offset=offset, roun=2, modellist=fit_models, namedict=namedict,i=0,j=1)

    table_text += """ \\hline \\\\ $\\mathbf{Citation~Preference: Negative}$ & & & & & \\\\ [1.8ex]"""

    table_text += table_row(varname='const', offset=0, roun=2, modellist=fit_models, namedict=namedict,i=1,j=2)

    for offset, varlist in enumerate(exog_var_sets):
        for var in varlist:
            if not 'year' in var:  #skip the year fixed effects
                table_text += table_row(varname=var, offset=offset, roun=2, modellist=fit_models, namedict=namedict,i=1,j=2)


    
    table_text += """\\hline 
\\hline \\\\[-1.8ex] 
\\textit{Note:} & \\multicolumn{2}{r}{$^{*}p<0.05$; $^{**}p<0.01$; $^{***}p<0.001$} \\\\ \n"""
    
    table_text += "Observations & " + " & ".join([str(model.nobs) for model in fit_models]) + " \\\\ \n"
    table_text += "Pseudo $R^2$ & " + " & ".join([str(np.round(model.prsquared, 4)) for model in fit_models]) + " \\\\ \n"
    table_text += "Log Likelihood & " + " & ".join([str(model.llf.round(2)) for model in fit_models]) + " \\\\ \n"
    #table_text += "F statistic & " + " & ".join(["$" + str(np.round(model.f_statistic.stat, 2)) + pvalue2stars(model.f_statistic.pval) + "$ (d.f.=" + str(model.f_statistic.df) + ")" for model in fit_models]) + " \\\\ \n"

    # m8.llr,m8.df_model,m8.llr_pvalue

    fstat_text = []
    for model in fit_models:
        fstat_text.append("$" + str(np.round(model.llr, 2)) + pvalue2stars(model.llr_pvalue) + "$ (d.f.=" + str(model.df_model) + ")")
        
    table_text += "LLR $\\chi^2$  & " + " & ".join(fstat_text) + " \\\\ \n"
    table_text += "Year FE  & " + " & ".join(['Yes' for model in fit_models]) + " \\\\ \n"
    
    table_text += "\\hline \n\\end{longtable} } }"
    
    print(table_text)


def logistic_coefficient_plot(df1, df2, colours, Model_features, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,12))
    
    
    df1s=df1[df1['pvalues']<=0.05]
    df1ns=df1[df1['pvalues']>0.05]

    df2s=df2[df2['pvalues']<=0.05]
    df2ns=df2[df2['pvalues']>0.05]

    x1s = df1s['risk']
    y1s = np.array(df1s.index.to_list())-0.2
    xerror1s = df1s['xerror']

    x1ns = df1ns['risk']
    y1ns = np.array(df1ns.index.to_list())-0.2
    xerror1ns = df1ns['xerror']

    x2s = df2s['risk']
    y2s = np.array(df2s.index.to_list())+0.2
    xerror2s = df2s['xerror']

    x2ns = df2ns['risk']
    y2ns = np.array(df2ns.index.to_list())+0.2
    xerror2ns = df2ns['xerror']
    
    y=df1.index.to_list()
    
    ax.errorbar(x1s, y1s, xerr=xerror1s, fmt='o',color=colours[1],label="Positive recognition",ms=4)
    ax.errorbar(x2s, y2s, xerr=xerror2s, fmt='o',color=colours[2],label="Negative recognition",ms=4)
    
    ax.errorbar(x1ns, y1ns, xerr=xerror1ns, fmt='o',color=colours[1],mfc='w',mew=1,label="Positive recognition(not sig)",ms=4)
    ax.errorbar(x2ns, y2ns, xerr=xerror2ns, fmt='o',color=colours[2],mfc='w',mew=1,label="Negative recognition(not sig)",ms=4)

    #ax.errorbar(x1ns, y1ns, xerr=xerror1ns, fmt='o',color=colours[1],label="Positive recognition(not sig)",ms=4, alpha=0.5)
    #ax.errorbar(x2ns, y2ns, xerr=xerror2ns, fmt='o',color=colours[2],label="Negative recognition(not sig)",ms=4, alpha=0.5)

    ax.invert_yaxis()
    
    ax.axvline(x=0, color=colours[0],  linestyle='--',ymin=0,ymax=0.98)

    sns.despine(top=True, right=True, left=True, bottom=False)

    ylabels=df1['item']
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xticks([-3,-2,-1,0,1,2,3,4])
    ax.tick_params(axis="both", which="both", bottom=True, top=False,    
                    labelbottom=True, left=False, right=False, labelleft=True,labelsize=9) 

    ax.set_ylim([len(Model_features)+2.5,0])
    ax.set_xlabel("Regression coefficient",fontsize=10)

    hlines=y
    counter=0
    for ix in hlines: 
        if counter % 2 == 0:
            ax.axhspan(ix - 0.5, ix + 0.5, color=colours[0],alpha=0.3, zorder=0,lw=0)
        counter += 1


    plt.legend(bbox_to_anchor=(-0.3, 0.55, 0.5, 0.5),
        fontsize=10,
               loc="upper left",
               ncol=2,
               markerscale=1,
               frameon=False,
               handletextpad=.1,
               columnspacing=.2)
    
    return ax