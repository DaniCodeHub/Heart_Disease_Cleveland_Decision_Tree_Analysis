import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd


dataset = pd.read_csv('/Users/marinamudrova/Downloads/heart-disease-cleveland-cleaned-final - heart-disease-cleveland-cleaned (1).csv')
df = pd.DataFrame(dataset)
print(df)

unique_count = df['diagnosis'].nunique()
print(f'Number of Unique Diagnoses: ', unique_count)

y = df['diagnosis']
#print(y)

X = df.drop('diagnosis', axis=1)
#print(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100, shuffle=True)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=5, ccp_alpha=0.001)
dtc = dtc.fit(X_train, y_train)

print(dtc.get_params())


print(dtc.predict_proba(X_test))

train_predictions = dtc.predict(X_train)
print(train_predictions)

predictions = dtc.predict(X_test)
print(predictions)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
print(f'Accuracy Score: ', accuracy_score(y_test, predictions))
print(f'Confusion Matrix Score: ', confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3, 4]))
print(f'Precision Score: ', precision_score(y_test, predictions, labels=[0, 1, 2, 3 ,4 ], average='weighted'))
print(f'Recall Score: ', recall_score(y_test, predictions,  labels=[0, 1, 2, 3 ,4 ], average='weighted'))

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, target_names=['T# 0', 'T# 1', 'T# 2', 'T# 3', 'T# 4']))

feature_names = X.columns
print(feature_names)

feature_importance = pd.DataFrame(dtc.feature_importances_, index=feature_names).sort_values(0, ascending=False)
print(feature_importance)

features = list(feature_importance[feature_importance[0]>0].index)
print(features)

feature_importance.head(10).plot(kind='bar')
plt.show()

from sklearn import tree

fig = plt.figure(figsize=(50, 50))
_ = tree.plot_tree(dtc,
                   feature_names=feature_names,
                   class_names={0: 'T# 0', 1: 'T# 1', 2: 'T# 2', 3: 'T# 3', 4: 'T# 4' },
                   filled=True,
                   fontsize=3)
plt.show()
 


