from utils.preprocessing import load_data, preprocess_data, split_data
from models.knn_model import train_knn
from models.svm_model import train_svm
from models.logistic_regression import train_lr
from sklearn.metrics import accuracy_score

data_path = 'data/MIGREN.csv'
data = load_data(data_path)
X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(X, y)

knn_best, knn_score = train_knn(X_train, y_train)
print(f'Best KNN Score: {knn_score}')

svm_best, svm_score = train_svm(X_train, y_train)
print(f'Best SVM Score: {svm_score}')

lr_best = train_lr(X_train, y_train)
accuracy_knn = accuracy_score(y_test, knn_best.predict(X_test))
accuracy_svm = accuracy_score(y_test, svm_best.predict(X_test))
accuracy_lr = accuracy_score(y_test, lr_best.predict(X_test))

print(f'KNN Accuracy: {accuracy_knn}')
print(f'SVM Accuracy: {accuracy_svm}')
print(f'LR Accuracy: {accuracy_lr}')
