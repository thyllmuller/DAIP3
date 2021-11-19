import pandas as pd


def c_Bload():
    cdb = pd.read_csv("Company_B_LN.csv", low_memory=False)
    return cdb


if __name__ == '__main__':
    c_Bload()