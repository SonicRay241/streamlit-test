from collections import defaultdict
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

class DataLoader:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.x = None
        self.y = None
        self.enc = None
    
    def separate_y(self, y_column):
        self.y = self.data[y_column]
        self.x = self.data.drop(columns=y_column, axis=1)
    
    def remove_column(self, column):
        self.data.drop(columns=column, inplace=True)

class RFModel:
    def __init__(self, dataloader: DataLoader):
        self.data = dataloader
        self.model = RandomForestClassifier()
        self.train_data = { "x": None, "y": None }
        self.test_data = { "x": None, "y": None }
        self.predictor: RandomForestClassifier = None
    
    def encode_y(self):
        self.enc = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        self.train_data["y"] = self.enc.fit_transform(self.train_data["y"].to_numpy().reshape(-1, 1))
        self.test_data["y"] = self.enc.transform(self.test_data["y"].to_numpy().reshape(-1, 1))

    def decode(self, input_data):
        return self.enc.inverse_transform(input_data)
    
    def split_data(self, test_size= 0.2):
        self.train_data["x"], self.test_data["x"], self.train_data["y"], self.test_data["y"] = train_test_split(
            self.data.x, 
            self.data.y,
            test_size = test_size
        )
    
    def search(self, params):
        grid = GridSearchCV(
            self.model,
            params,
            cv=3,
            scoring='f1_macro',
            verbose = 3,
            n_jobs=-1 #all cores
        )
        self.best = grid.fit(self.train_data["x"], self.train_data["y"]).best_estimator_

    def predict(self, data):
        return self.best.predict(data)
    
    def evaluate(self):
        predictions = self.predict(self.test_data["x"])
        print(
            "Accuracy:",
            accuracy_score(self.test_data["y"], predictions),
            "Classification Report:",
            classification_report(self.test_data["y"], predictions),
            sep="\n"
        )

    def save(self, filename="model.pkl"):
        joblib.dump(self.best, filename)
    
    def load(self, path):
        self.best = joblib.load(path)

if __name__ == "__main__":
    data = DataLoader("./dataset/Iris.csv")
    data.remove_column("Id")
    data.separate_y("Species")

    model = RFModel(data)
    model.split_data()
    model.encode_y()

    grid_params = {
        'max_depth': [3,5,7,10,20,30],
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': [10, 20, 30 , 40],
        'min_samples_leaf': [1, 2, 4]
    }
    model.search(grid_params)
    model.evaluate()
    model.save("model.pkl")