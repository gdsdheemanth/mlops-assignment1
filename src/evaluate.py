
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score


class Evaluator:
    def evaluate_model(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='positive',
                                    average='micro')
        recall = recall_score(y_test, y_pred, pos_label='positive',
                              average='micro')
        f1 = f1_score(y_test, y_pred, pos_label='positive', average='micro')

        return accuracy, precision, recall, f1

    def print_metrics(self, model_name, metrics):
        accuracy, precision, recall, f1 = metrics
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print('---')
