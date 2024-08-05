import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from evaluate import Evaluator
from preprocess import Preprocessor
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=[i for i in range(cm.shape[0])],
                         columns=[i for i in range(cm.shape[1])])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(y_true, y_proba, output_path, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                 label=f'ROC curve (area = %0.2f) for class {i}' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    plt.close()


def plot_precision_recall_curve(y_true, y_proba, output_path, n_classes):
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                            y_proba[:, i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], marker='.', label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(output_path)
    plt.close()


def plot_learning_curve(estimator, X, y, output_path):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                            cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-',
             label="Cross-validation score")
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.savefig(output_path)
    plt.close()


def plot_validation_curve(estimator, X, y, param_name, param_range,
                          output_path):
    train_scores, test_scores = validation_curve(estimator, X, y,
                                                 param_name=param_name,
                                                 param_range=param_range, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(param_range, train_scores_mean, 'o-', label="Training score")
    plt.plot(param_range, test_scores_mean, 'o-',
             label="Cross-validation score")
    plt.xlabel('Parameter value')
    plt.ylabel('Score')
    plt.title(f'Validation Curve ({param_name})')
    plt.legend(loc='best')
    plt.savefig(output_path)
    plt.close()


class Model:
    def __init__(self, path) -> None:
        self.path = path

    def main(self):
        try:
            # Set tracking URI (optional)
            mlflow.set_tracking_uri("file:../mlruns")
            experiment_name = "RandomForestClassifier"
            mlflow.set_experiment(experiment_name)
            # Verify if experiment is created
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                print("Experiment not created!")
                return
            else:
                print(f"Experiment ID: {experiment.experiment_id}")
            with mlflow.start_run():
                preprocessor_initial = Preprocessor(self.path)
                evaluator = Evaluator()
                preprocessor_initial.load_data()
                df = preprocessor_initial.get_dataFrame()
                df_major = df[(df.quality == 6) | (df.quality == 5) | (
                    df.quality == 7)]
                df_8 = df[df.quality == 8]
                df_4 = df[df.quality == 4]
                df_5 = df[df.quality == 5]
                df_9 = df[df.quality == 9]
                df_8_sample = resample(df_8, replace=True, n_samples=880,
                                       random_state=42)
                df_4_sample = resample(df_4, replace=True, n_samples=880,
                                       random_state=42)
                df_5_sample = resample(df_5, replace=True, n_samples=880,
                                       random_state=42)
                df_9_sample = resample(df_9, replace=True, n_samples=880,
                                       random_state=42)
                df = pd.concat([df_major, df_9_sample, df_8_sample,
                                df_5_sample, df_4_sample])
                preprocessed_data = Preprocessor(dataframe=df)
                X_train, _, y_train, _ = preprocessed_data.split_data(
                    'quality')
                _, X_test, _, y_test = preprocessor_initial.split_data(
                    'quality')
                n_classes = len(np.unique(y_train))

                mlflow.log_param("dataset_size", len(X_train) + len(X_test))
                mlflow.log_param("test_size", len(X_test))

                parameters = {'n_estimators': (30, 60, 100, 130, 160),
                              'criterion': ('gini', 'entropy', 'log_loss')}
                Randomforest = RandomForestClassifier(random_state=42)
                clf = GridSearchCV(Randomforest, parameters)
                clf.fit(X_train, y_train)
                # print(clf.best_params_)
                # print(clf.best_score_)
                model = clf.best_estimator_
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)

                mlflow.log_param("model_type", "RandomForestClassifier")
                mlflow.log_param("parameters", clf.best_params_)
                mlflow.log_param("best score", clf.best_score_)
                # Log model
                mlflow.sklearn.log_model(model, "model")

                # Evaluate
                metrics = evaluator.evaluate_model(y_test, y_pred)
                accuracy, precision, recall, f1 = metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)

                # Log confusion matrix as artifact
                output_path = "../results/confusion_matrix.png"
                plot_confusion_matrix(y_test, y_pred, output_path)
                mlflow.log_artifact(output_path)

                y_test_bin = label_binarize(y_test, classes=np.arange(
                    n_classes))

                # Log ROC curve as artifact
                roc_path = "../results/roc_curve.png"
                plot_roc_curve(y_test_bin, y_proba, roc_path, n_classes)
                mlflow.log_artifact(roc_path)

                # Log Precision-Recall curve as artifact
                prc_path = "../results/precision_recall_curve.png"
                plot_precision_recall_curve(y_test_bin, y_proba, prc_path,
                                            n_classes)
                mlflow.log_artifact(prc_path)

                # Log Learning Curve as artifact
                lc_path = "../results/learning_curve.png"
                plot_learning_curve(model, X_train, y_train, lc_path)
                mlflow.log_artifact(lc_path)

                # Log Validation Curve as artifact
                vc_path = "../results/validation_curve.png"
                plot_validation_curve(model, X_train, y_train,
                                      param_name="n_estimators",
                                      param_range=[10, 50, 100, 200],
                                      output_path=vc_path)
                mlflow.log_artifact(vc_path)

                print("Save model")
                joblib.dump(model, '../models/random_forest_model.joblib')
                print("MLflow run completed successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == '__main__':
    model = Model('../data/winequality-white.csv')
    model.main()
