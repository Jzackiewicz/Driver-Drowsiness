import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
import pickle

if __name__ == '__main__':
    df = pd.read_csv('coordsnew.csv', delimiter=";")
    X = df.drop('class', axis=1)  # features
    Y = df['class']  # target value
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

    pipelines = {
        'lr': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }

    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train.values, Y_train)
        fit_models[algo] = model

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, accuracy_score(Y_test, yhat))

    with open('ML/body_language.pkl', 'wb') as f:
        pickle.dump(fit_models['lr'], f)
