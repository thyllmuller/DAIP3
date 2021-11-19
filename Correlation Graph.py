import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import D_C_T

pd.set_option('display.max_columns', None)

'''LOADING ALL DATA AND MERGING'''
# my data import
cb = D_C_T.c_Bload()



# dont forget to adjust the vmin and vmax, +- 0.5 for A and +-0.25 for B.
# and change output name
def corr_graph(x):
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    corr = x.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(corr, vmin=-0.5, vmax=.5, center=0, mask=mask,
                cmap=cmap, linewidths=1, linecolor="white",
                square=True, annot=True, cbar_kws={"orientation": "horizontal", "shrink": .60})
    plt.xticks(rotation=45, ha='right')
    ax.set_title("Correlation of all Variables in the Dataset", fontsize=30, y=1, pad=10)
    output_name = str(f"Correlation Graph c_B_LN")
    plt.tight_layout()
    fig.savefig(output_name)
    plt.show()


corr_graph(cb)
