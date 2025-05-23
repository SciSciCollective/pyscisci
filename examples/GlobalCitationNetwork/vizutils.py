import numpy as np 
import pandas as pd

import matplotlib.gridspec as gridspec


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