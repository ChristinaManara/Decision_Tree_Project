import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix


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
    #  Create a similar figure, except this time select by the not.fully.paid column
    x = loans[loans['not.fully.paid'] == 1]['fico']
    y = loans[loans['not.fully.paid'] == 0]['fico']
    plt.figure(figsize=(10, 6))
    x.hist(alpha=0.5, color='blue', bins=30, label='not.fully.paid=1')
    y.hist(alpha=0.5, color='red', bins=30, label='not.fully.paid=0')
    plt.legend()
    plt.xlabel('FICO')
    # Create a countplot using seaborn showing the counts of loans by purpose,
    # with the color hue defined by not.fully.paid.
    sns.countplot('purpose', hue='not.fully.paid', data=loans, palette='Set1')
    sns.jointplot(x='fico', y='int.rate', data=loans, color='purple')
    plt.figure(figsize=(11, 7))
    sns.lmplot(y='int.rate', x='fico', data=loans, hue='credit.policy', col='not.fully.paid', palette='Set1')
    plt.show()
    cat_feats = ['purpose']
    final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)
    X = final_data.drop('not.fully.paid', axis=1)
    y = final_data['not.fully.paid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    predictions = dtree.predict(X_test)
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test, predictions))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
