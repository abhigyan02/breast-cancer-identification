import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from matplotlib import pyplot as plt

cancer_data = datasets.load_breast_cancer()

features = cancer_data.data
labels = cancer_data.target

# with grid search we can find an optimal parameters
param_grid = {'max_depth': np.arange(1, 10)}
features_train, features_test, target_train, target_test = train_test_split(features, labels, test_size=0.25)

tree = GridSearchCV(DecisionTreeClassifier(), param_grid)

tree.fit(features_train, target_train)

print('Best parameter with Grid Search', tree.best_params_)

grid_prediction = tree.predict(features_test)

confusionMatrix = confusion_matrix(target_test, grid_prediction, labels=tree.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=tree.classes_)
display.plot()
plt.show()

print('Accuracy', accuracy_score(target_test, grid_prediction)*100)
