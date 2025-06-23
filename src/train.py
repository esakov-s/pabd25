"""
This is the train ml model script.
Linear regression with 5 features:
- total_meters
- floors_count
- first_floor
- last_floor
- n_rooms (One Hot Encoded)
"""

import argparse
import os

import logging
import joblib
import pandas as pd
from models_functions.catboost import CatBoostModel
from sklearn.metrics import mean_squared_error


MODEL_NAME = "catboost_model.pkl"

logging.basicConfig(
    filename="train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def train_model(model_path):
    """Train model and save with MODEL_NAME"""
    train_df = pd.read_csv("data/processed/train.csv")
    X_train = train_df[
        [
            "total_meters",
            "floors_count",
            "rooms_1",
            "rooms_2",
            "rooms_3",
            "first_floor",
            "last_floor",
        ]
    ]
    y_train = train_df["price"]
   
    cat_features = ["rooms_1", "rooms_2", "rooms_3", "first_floor", "last_floor"]
    model = CatBoostModel(model_path=model_path, cat_features=cat_features, verbose=500)
    model.train(X_train, y_train)

    logging.info(f"Train {model} and save to {model_path}")

    joblib.dump(model.best_model_, model_path)


if __name__ == "__main__":
    """Parse arguments and train model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    args = parser.parse_args()
    model_path = os.path.join("models", args.model)

    train_model(model_path)
