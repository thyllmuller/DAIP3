from matplotlib import pyplot as plt
import pandas as pd
import D_C_T
import timeit


tic = timeit.default_timer()
pd.set_option('display.max_columns', None)
cb = D_C_T.c_Bload()

# library & dataset
import seaborn as sns

# use the function regplot to make a scatterplot

sns.jointplot(x=cb["Account_Length"], y=cb["Total_Charge"], kind='scatter')
sns.jointplot(x=cb["Account_Length"], y=cb["Total_Charge"], kind='hex')
sns.jointplot(x=cb["Account_Length"], y=cb["Total_Charge"], kind='kde')

plt.show()
