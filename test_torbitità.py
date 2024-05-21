import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, delimiter=';')
    
    if 'Turbidity' not in data.columns or 'Cloud' not in data.columns:
        print("Warning: 'Turbidity' or 'Cloud' not found in the CSV file headers.")
        return None, None, None, None
    
    data = data[data['Cloud'] < 0.2]
    X = data.drop(['Turbidity', 'Cloud'], axis=1).values
    y = data['Turbidity'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, data.columns.drop(['Turbidity', 'Cloud'])
def analyze_distribution(data, feature_names):
    for feature in feature_names:
        plt.figure()
        sns.histplot(data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()
def find_outliers(data, feature_names):
    for feature in feature_names:
        plt.figure()
        sns.boxplot(x=data[feature])
        plt.title(f'Outliers in {feature}')
        plt.show()
def try_regression_models(X_train, y_train, X_test, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = mse
        print(f'{name}: MSE = {mse}')
    
    return models, results
def evaluate_error(results):
    for name, mse in results.items():
        print(f'{name}: MSE = {mse}')
def select_important_features(X_train, y_train, feature_names):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(len(feature_names)):
        print(f'{i + 1}. feature {feature_names[indices[i]]} ({importances[indices[i]]})')
    
    sfm = SelectFromModel(model, threshold=0.1)
    sfm.fit(X_train, y_train)
    
    selected_features = sfm.transform(X_train)
    
    return selected_features, sfm.get_support(indices=True)
def retry_with_selected_features(X_train, X_test, y_train, y_test, selected_indices):
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    models, results = try_regression_models(X_train_selected, y_train, X_test_selected, y_test)
    
    return models, results
def main():
    filepath = 'dati.csv'
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(filepath)
    
    if X_train is None or X_test is None or y_train is None or y_test is None:
        return
    
    data = pd.read_csv(filepath, delimiter=';')
    analyze_distribution(data, feature_names)
    find_outliers(data, feature_names)
    
    models, results = try_regression_models(X_train, y_train, X_test, y_test)
    evaluate_error(results)
    
    X_train_selected, selected_indices = select_important_features(X_train, y_train, feature_names)
    models, results = retry_with_selected_features(X_train, X_test, y_train, y_test, selected_indices)
    evaluate_error(results)

if __name__ == "__main__":
    main()
