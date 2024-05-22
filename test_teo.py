import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.model_selection import LearningCurve

# Carica i dati dal file CSV
data = pd.read_csv("dati.csv", delimiter=';')

# Esplora e pulisci i dati
# Gestione dei dati mancanti
data.dropna(inplace=True)

# Selezione di un campione di dati validi (senza copertura nuvolosa)
data_valid = data[data['Cloud'] < 0.1]

# Analisi della distribuzione dei valori nelle bande spettrali
# e individuazione di eventuali outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_valid.drop(['Turbidity', 'Cloud'], axis=1))
plt.title('Distribuzione delle bande spettrali')
plt.xlabel('Bande spettrali')
plt.ylabel('Valore')
plt.show()

# Dividi i dati in set di addestramento e test
X = data_valid.drop(['Turbidity', 'Cloud'], axis=1) # Features
y = data_valid['Turbidity'] # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Costruisci un pipeline per la gestione delle features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

# Definisci i modelli
models = {'Linear Regression': LinearRegression(),
          'Ridge Regression': Ridge(),
          'Lasso Regression': Lasso(),
          'Support Vector Regression': SVR(),
          'Gradient Boosting Regression': GradientBoostingRegressor()}

# Addestra e valuta i modelli
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)

    print(f"\n{name}:")
    
    # Calcola e visualizza la curva di apprendimento
    plt.figure(figsize=(8, 6))
    visualizer = LearningCurve(pipeline, scoring='r2')
    visualizer.fit(X, y)
    visualizer.show()

    # Valutazione R2 score
    print("Training R2 score:", pipeline.score(X_train, y_train))
    print("Test R2 score:", pipeline.score(X_test, y_test))

    # Valuta le prestazioni del modello
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)  # Calcolo del MAE
    mse = mean_squared_error(y_test, y_pred)   # Calcolo del MSE
    print(f"Test MAE for {name}: {mae:.4f}")
    print(f"Test MSE for {name}: {mse:.4f}")

    # Visualizza previsioni rispetto ai valori effettivi
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
    plt.title(f"{name} - Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

    # Visualizza l'importanza delle features (per modelli non lineari)
    if name in ['Support Vector Regression', 'Gradient Boosting Regression']:
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            feature_importance = pipeline.named_steps['model'].feature_importances_
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_importance, y=X.columns)
            plt.title(f"{name} - Feature Importance")
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.show()

    # Visualizza i coefficienti (solo per modelli lineari)
    if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        if hasattr(pipeline.named_steps['model'], 'coef_'):
            coef = pipeline.named_steps['model'].coef_
            plt.figure(figsize=(10, 6))
            sns.barplot(x=coef, y=X.columns)
            plt.title(f"{name} - Coefficients")
            plt.xlabel('Coefficient Value')
            plt.ylabel('Features')
            plt.show()
# Calcola le metriche per ogni modello
metrics = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)
    
    # Calcola le metriche
    train_r2 = pipeline.score(X_train, y_train)
    test_r2 = pipeline.score(X_test, y_test)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Salva le metriche
    metrics[name] = {'Train R2': train_r2, 'Test R2': test_r2, 'MAE': mae, 'MSE': mse}

# Confronta le metriche per selezionare il miglior modello
metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
best_model = metrics_df['Test R2'].idxmax()

# Visualizza il grafico del miglior modello
plt.figure(figsize=(10, 6))
sns.barplot(x=metrics_df.index, y='Test R2', data=metrics_df, palette='Blues')
plt.title('Test R2 score per ogni modello')
plt.xlabel('Modello')
plt.ylabel('Test R2 score')
plt.xticks(rotation=45)
plt.axhline(y=metrics_df.loc[best_model, 'Test R2'], color='red', linestyle='--', label='Miglior modello')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Il miglior modello per la previsione della torbidità è: {best_model}")
print(f"Test R2 score del miglior modello: {metrics_df.loc[best_model, 'Test R2']}")

