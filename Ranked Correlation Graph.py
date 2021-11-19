import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


def c_Aload():
    ca = pd.read_csv("Company_A.csv", low_memory=False)
    ca['Churn'] = ca['Churn'].map({'Yes': 1, 'No': 0})
    ca['PaperlessBilling'] = ca['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    ca['Partner'] = ca['Partner'].map({'Yes': 1, 'No': 0})
    ca['Dependents'] = ca['Dependents'].map({'Yes': 1, 'No': 0})
    ca['PhoneService'] = ca['PhoneService'].map({'Yes': 1, 'No': 0})
    ca['MultipleLines'] = ca['MultipleLines'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    ca['InternetService'] = ca['InternetService'].map({'Fiber optic': 2, 'DSL': 1, 'No': 0})
    ca['OnlineSecurity'] = ca['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    ca['OnlineBackup'] = ca['OnlineBackup'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
    ca['DeviceProtection'] = ca['DeviceProtection'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
    ca['TechSupport'] = ca['TechSupport'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
    ca['StreamingTV'] = ca['StreamingTV'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
    ca['StreamingMovies'] = ca['StreamingMovies'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
    ca['Contract'] = ca['Contract'].map({'Two year': 2, 'One year': 1, 'Month-to-month': 0})
    ca['PaymentMethod'] = ca['PaymentMethod'].map(
        {'Credit card (automatic)': 3, 'Bank transfer (automatic)': 2, 'Electronic check': 1, 'Mailed check': 0})
    ca.drop(['index', 'customerID'], axis=1)
    return ca


ca = c_Aload()

print(ca)


def c_Bload():
    cb = pd.read_csv("Company_B.csv", low_memory=False)
    cb['International plan'] = cb['International plan'].map({'Yes': 1, 'No': 0})
    cb['Voice mail plan'] = cb['Voice mail plan'].map({'Yes': 1, 'No': 0})

    return cb


cb = c_Bload()

print(cb)

# dont forget to adjust the vmin and vmax, +- 0.5 for A and +-0.25 for B.
# and change output name
def corr_graph(x):
    fig, ax = plt.subplots(figsize=(8, 10))
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    sns.heatmap(x.corr()[['Churn']].sort_values(by='Churn', ascending=False),
                vmin=-0.25, vmax=0.25, center=0, linewidths=0.5, linecolor='white',
                cmap=cmap, square=True, annot=True, cbar=False)
    ax.tick_params(right=True, top=False, labelright=True, labeltop=False, left=False, labelleft=False, bottom=False,
                   labelbottom=True)
    plt.yticks(rotation=0, ha='left')
    ax.set_xlabel('')
    ax.set(xticklabels=[])
    output_name = str("Ranked Correlation Heatmap c_B")
    plt.tight_layout()
    fig.savefig(output_name)
    plt.show()


corr_graph(cb)
