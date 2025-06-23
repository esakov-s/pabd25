from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib


class LinearRegressionModel:
    def __init__(self, model_path, model_type='ridge', random_state=42):
        """
        Инициализация модели линейной регрессии.

        Параметры:
        ----------
        model_type : str, optional (default='ridge')
            Тип модели: 'ridge', 'lasso', 'elasticnet'.
        """
        self.model_type = model_type
        self.model = None
        self.best_params_ = None
        self.best_score_ = None
        self.model_path = model_path
        self.random_state = random_state

    def _get_model(self):
        """Возвращает модель в зависимости от типа."""
        if self.model_type == 'ridge':
            return Ridge()
        elif self.model_type == 'lasso':
            return Lasso()
        elif self.model_type == 'elasticnet':
            return ElasticNet()
        else:
            raise ValueError("Неизвестный тип модели. Допустимые значения: 'ridge', 'lasso', 'elasticnet'")

    def _get_grid_param_dict(self):
        """Возвращает модель в зависимости от типа."""
        if self.model_type == 'ridge':
            return {
                    "alpha": [0.01, 0.1, 1, 10, 100],
                    "fit_intercept": [True, False],
                }
        elif self.model_type == 'lasso':
            return {
                    "alpha": [0.01, 0.1, 1, 10, 100],
                    "fit_intercept": [True, False],
                }
        elif self.model_type == 'elasticnet':
            return {
                    "alpha": [0.01, 0.1, 1, 10],
                    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "fit_intercept": [True, False],
                }
        else:
            raise ValueError("Неизвестный тип модели. Допустимые значения: 'ridge', 'lasso', 'elasticnet'")

    def train(self, X_train, y_train, cv=5):
        """
        Обучение модели с подбором гиперпараметров через GridSearchCV.

        Параметры:
        ----------
        X_train : array-like, форма (n_samples, n_features)
            Тренировочные данные.
        y_train : array-like, форма (n_samples,)
            Целевые значения.
        param_grid : dict
            Сетка гиперпараметров для GridSearch.
        cv : int, optional (default=5)
            Количество фолдов кросс-валидации.
        """
        model = self._get_model()
        param_grid = self._get_grid_param_dict()
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=KFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_  # neg MSE -> MSE

        print(f"Лучшие параметры: {self.best_params_}")
        print(f"Лучший R2: {self.best_score_:.4f}")

        joblib.dump(self.model, self.model_path)

    def test(self, X_test, y_test):
        """
        Оценка модели на тестовых данных.

        Параметры:
        ----------
        X_test : array-like, форма (n_samples, n_features)
            Тестовые данные.
        y_test : array-like, форма (n_samples,)
            Истинные целевые значения.

        Возвращает:
        ----------
        metrics : dict
            Словарь с метриками: MSE, RMSE, MAE, R².
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train_with_grid_search().")

        y_pred = self.model.predict(X_test)

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
