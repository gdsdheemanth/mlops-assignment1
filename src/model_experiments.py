from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from preprocess import Preprocessor
from train import Trainer
from evaluate import Evaluator


class ModelBuilder:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=20, 
                                                   random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'KNeighbors': KNeighborsClassifier(n_neighbors=8)
        }

    def get_models(self):
        return self.models


if __name__ == '__main__':
    # Initialize classes
    preprocessor = Preprocessor('../data/winequality-white.csv')
    model_builder = ModelBuilder()
    trainer = Trainer()
    evaluator = Evaluator()

    # Load and preprocess data
    preprocessor.load_data()
    preprocessor.preprocess_data()

    print("Attributes with high correlation")
    print(preprocessor.get_to_drop())

    X_train, X_test, y_train, y_test = preprocessor.split_data('quality')

    models = model_builder.get_models()
    for model_name, model in models.items():
        # Train and predict
        trained_model = trainer.train_model(model, X_train, y_train)
        y_pred = trainer.predict_model(trained_model, X_test)

        # Evaluate
        metrics = evaluator.evaluate_model(y_test, y_pred)
        evaluator.print_metrics(model_name, metrics)

