class Trainer:
    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def predict_model(self, model, X_test):
        return model.predict(X_test)
