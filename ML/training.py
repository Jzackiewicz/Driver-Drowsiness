import pickle

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
# from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

if __name__ == '__main__':
    df = pd.read_csv('new_dataset.csv', delimiter=";")
    X = df.drop('class', axis=1)  # features
    Y = df['class']  # target value
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1234)

    # pipelines = {
    #     'MLP': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=1000, alpha=0, solver='adam'))
    # }

    pipelines = {
        'LDA': make_pipeline(LinearDiscriminantAnalysis())
    }

    fit_models = {}
    for algo, pipeline in tqdm(pipelines.items()):
        model = pipeline.fit(X_train.values, Y_train)
        fit_models[algo] = model

    pl_accuracies = {}
    for algo, model in tqdm(fit_models.items()):
        yhat = model.predict(X_test)
        pl_accuracies[algo] = accuracy_score(Y_test, yhat)
        print(algo, accuracy_score(Y_test, yhat))

    max_acc = max(pl_accuracies, key=pl_accuracies.get)
    print(max_acc)
    with open('new_model.pkl', mode='wb') as f:
        pickle.dump(fit_models['LDA'], f)
