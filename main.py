import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def print_hi(name):
    # Use pandas to read loan_data.csv as a dataframe called loans.
    loans = pd.read_csv('loan_data.csv')
    print(loans)
    # Check out the info(), head(), and describe() methods on loans.
    print(loans.info())
    print(loans.head())
    print(loans.describe())
    # Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.
    sns.heatmap(loans.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    x = loans[loans['credit.policy'] == 1]['fico']
    y = loans[loans['credit.policy'] == 0]['fico']
    plt.figure(figsize=(10, 6))
    x.hist(alpha=0.5, color='blue', bins=30, label='Credit.Policy=1')
    y.hist(alpha=0.5, color='red', bins=30, label='Credit.Policy=0')
    plt.legend()
    plt.xlabel('FICO')
    plt.show()
    #  Create a similar figure, except this time select by the not.fully.paid column
    x = loans[loans['not.fully.paid'] == 1]['fico']
    y = loans[loans['not.fully.paid'] == 0]['fico']
    plt.figure(figsize=(10, 6))
    x.hist(alpha=0.5, color='blue', bins=30, label='not.fully.paid=1')
    y.hist(alpha=0.5, color='red', bins=30, label='not.fully.paid=0')
    plt.legend()
    plt.xlabel('FICO')
    plt.show()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
