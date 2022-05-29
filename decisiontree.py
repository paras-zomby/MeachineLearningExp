import sklearn as sk
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train_data = pd.read_csv('dataset/titanic/train.csv')
    label = train_data.loc['Survived']
    train_data = train_data.drop('Survived')
    test_data = pd.read_csv('dataset/titanic/test.csv')
    dtc = sk.tree.DecisionTreeClassifier(criterion="entropy")
    dtc.fit(train_data, label)
    out = dtc.predict(test_data)
