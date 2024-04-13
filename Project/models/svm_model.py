from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_svm(X_train, y_train):
    svm_params = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
    svm = SVC()
    grid_svm = GridSearchCV(svm, svm_params, refit=True, verbose=0)
    grid_svm.fit(X_train, y_train)
    return grid_svm.best_estimator_, grid_svm.best_score_
