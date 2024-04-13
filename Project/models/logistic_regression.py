from sklearn.linear_model import LogisticRegression

def train_lr(X_train, y_train):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    return lr
