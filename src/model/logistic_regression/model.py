from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from src.services.dataset.dataset_service import dataset_service

def train_churn_model(X_train, y_train):
    preprocessor = dataset_service.get_processor()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)

    return pipeline

def get_metrics(pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return accuracy, f1

def train_and_measure_model():
    X_train, X_test, y_train, y_test = dataset_service.get_train_test()

    pipeline = train_churn_model(X_train, y_train)

    accuracy, f1 = get_metrics(pipeline, X_test, y_test)

    return pipeline, accuracy, f1
