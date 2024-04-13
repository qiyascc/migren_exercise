from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def train_knn(X_train, y_train):
    knn_params = {'n_neighbors': range(1, 10)}
    knn = KNeighborsClassifier()
    grid_knn = GridSearchCV(knn, knn_params, refit=True, verbose=0)
    grid_knn.fit(X_train, y_train)
    return grid_knn.best_estimator_, grid_knn.best_score_
