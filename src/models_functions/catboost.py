from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib


class CatBoostModel:
    def __init__(self, model_path, random_state=42, **kwargs):
        """
        Инициализация CatBoost модели.

        Параметры:
        ----------
        **kwargs : dict
        """
        self.model = CatBoostRegressor(**kwargs)
        self.best_params_ = None
        self.best_score_ = None
        self.model_path = model_path
        self.random_state = random_state

    def train(self, X_train, y_train, cv=5):
        """
        Обучение CatBoost с подбором гиперпараметров через GridSearch.

        Параметры:
        ----------
        X_train : array-like или pd.DataFrame, форма (n_samples, n_features)
        y_train : array-like, форма (n_samples,)
        param_grid : dict
        cv : int, optional (default=5)
        """
        param_grid = {
            "iterations": [500, 1000],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5, 7],
            "border_count": [32, 64],
        }

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=KFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring="r2",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.best_model_ = grid_search.best_estimator_

        print(f"Лучшие параметры: {self.best_params_}")
        print(f"Лучший R2: {self.best_score_}")

        joblib.dump(self.best_model_, self.model_path)

    def test(self, X_test, y_test):
        """
        Оценка модели на тестовых данных.

        Параметры:
        ----------
        X_test : array-like или pd.DataFrame, форма (n_samples, n_features)
        y_test : array-like, форма (n_samples,)

        Возвращает:
        ----------
        metrics : dict
            Словарь с метриками: MSE, RMSE, MAE, R².
        """
        y_pred = self.best_model_.predict(X_test)

        metrics = {
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }

        print("Метрики на тестовых данных:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        return metrics
