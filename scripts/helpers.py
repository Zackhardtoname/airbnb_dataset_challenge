import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def convert_string_to_dollar (str):
    return float(str.replace('$', '').replace(',', ''))

def percent_to_float(x):
    # convert a percent based string to a float between 0 and 1
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

def draw(y, x=None, filename=None, title=None, xlabel=None, ylabel=None):
    if not title:
        title = filename

    if not x:
        x = list(range(len(y)))

    plt.rcParams["figure.figsize"] = (20, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, color='lightblue', linewidth=3)
    plt.xticks(x)
    plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def drop_attribs(df, attribs, list_to_drop):
    # drop the list_to_drop columns from the dataframe and remove attribs from attribs
    df.drop(list_to_drop, axis=1, inplace=True)
    attribs = [col_name for col_name in attribs if col_name not in list_to_drop]

    return df, attribs
