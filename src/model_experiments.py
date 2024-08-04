from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from preprocess import Preprocessor
from train import Trainer
from evaluate import Evaluator
import mlflow
import mlflow.sklearn


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
    # Set tracking URI (optional)
    mlflow.set_tracking_uri("file:../mlruns")
    experiment_name = "ML Experiments"
    mlflow.set_experiment(experiment_name)
    # Verify if experiment is created
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print("Experiment not created!")
    else:
        print(f"Experiment ID: {experiment.experiment_id}")
    for model_name, model in models.items():
        with mlflow.start_run():
            mlflow.log_param("model_name", model_name)
            # Train and predict
            trained_model = trainer.train_model(model, X_train, y_train)
            y_pred = trainer.predict_model(trained_model, X_test)
            mlflow.sklearn.log_model(trained_model, "model")
            # Evaluate
            metrics = evaluator.evaluate_model(y_test, y_pred)
            accuracy, precision, recall, f1 = metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
