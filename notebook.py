import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,accuracy_score
import pickle

def preprocess_data(train_data, test_data):
    label_encoder = LabelEncoder()

    for column in train_data.columns:
        train_data[column] = label_encoder.fit_transform(train_data[column])
        test_data[column] = label_encoder.transform(test_data[column])

    train_data["type"] = train_data["type"].apply(lambda x: 1 if x == "p" else 0)
    test_data["type"] = test_data["type"].apply(lambda x: 1 if x == "p" else 0)

    return train_data, test_data

def modelling(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    train_data, test_data = preprocess_data(train_data, test_data)

    X_train = train_data.drop('type', axis=1)
    y_train = train_data['type']
    X_test = test_data.drop('type', axis=1)
    y_test = test_data['type']

    # Entrenar el modelo
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    #print("Classification Report:")
    #print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    #print(f'Accuracy: {accuracy:.2f}')

    # Guardar el modelo
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

modelling('train_dataset.csv', 'test_dataset.csv')