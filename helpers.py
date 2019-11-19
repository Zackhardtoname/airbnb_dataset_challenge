import seaborn as sns
import numpy as np

def convert_string_to_dollar (str):
    return float(str.replace('$', '').replace(',', ''))

def percent_to_float(x):
    try:
        return float(x.strip('%'))/100
    except:
        return np.nan

def corr_heatmap (corr):
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    return ax