import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Funzione per rimuovere tutti i file in una cartella
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

# Crea la cartella img se non esiste e rimuovi i file se esistono
img_folder = 'img'
if not os.path.exists(img_folder):
    os.makedirs(img_folder)
else:
    clear_folder(img_folder)

# Carica i dati
data = pd.read_csv("dati.csv", delimiter=';')

# Filtra i dati in base alla copertura nuvolosa
data = data[data['Cloud'] < 0.01]

# Definisci le feature e il target
X = data.iloc[:, 1:14]  # Le colonne delle bande spettrali
y = data['Turbidity']

# Normalizzazione delle feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisione in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Esplorazione dei Dati e Identificazione degli Outlier

# Calcolo delle statistiche descrittive
data_description = X.describe()
print("Statistiche Descrittive delle Feature:")
print(data_description)

# Identificazione degli Outlier con Local Outlier Factor (LOF)
lof = LocalOutlierFactor()
outlier_labels_lof = lof.fit_predict(X_scaled)
outlier_indices_lof = np.where(outlier_labels_lof == -1)[0]

# Funzione per valutare i modelli
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if name == "Lasso Regression":  # Analisi dei Coefficienti del Modello Lineare (Lasso)
        # Ottieni i coefficienti del modello Lasso addestrato
        coefficients = model.coef_

        # Creazione di un DataFrame per visualizzare i coefficienti
        coefficients_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})

        # Ordina i coefficienti per valore assoluto
        coefficients_df['AbsoluteCoefficient'] = np.abs(coefficients_df['Coefficient'])
        coefficients_df = coefficients_df.sort_values(by='AbsoluteCoefficient', ascending=False)

        # Plot dei coefficienti
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(coefficients_df)), coefficients_df['Coefficient'], tick_label=coefficients_df['Feature'])
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.title('Coefficients of Lasso Regression Model')
        plt.savefig(os.path.join(img_folder, 'lasso_coefficients.png'))
        plt.close()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - MSE: {mse}, MAE: {mae}, R2: {r2}")

    # Grafico dei risultati della regressione
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Valori Reali')
    plt.ylabel('Valori Predetti')
    plt.title(f'Real vs Predicted Values ({name})')
    plt.savefig(os.path.join(img_folder, f'real_vs_pred_{name.replace(" ", "_").lower()}.png'))
    plt.close()

    return mse, mae, r2

# Modelli da provare
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Valutazione dell'Impatto della Rimozione degli Outlier e Addestramento e Valutazione dei Modelli di Regressione

# Calcola le performance del modello prima della rimozione degli outlier
print("Performance del modello prima della rimozione degli outlier:")
results = []
for name, model in models.items():
    mse, mae, r2 = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    results.append((name, mse, mae, r2, 'Before Outlier Removal'))

# Rimuovi gli outlier dal dataset X_scaled e y corrispondenti
X_scaled_clean = np.delete(X_scaled, outlier_indices_lof, axis=0)
y_clean = np.delete(y, outlier_indices_lof, axis=0)

# Divisione in set di addestramento e di test dopo la rimozione degli outlier
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_scaled_clean, y_clean, test_size=0.2, random_state=42)

# Calcola le performance del modello dopo la rimozione degli outlier
print("\nPerformance del modello dopo la rimozione degli outlier:")
for name, model in models.items():
    # Gestione dei parametri dei modelli e cross-validation
    if name == "Linear Regression":
        # Definizione dei parametri per la regressione lineare
        linear_params = {'fit_intercept': [True, False]}

        # Ricerca dei parametri ottimali per la regressione lineare
        linear_grid_search = GridSearchCV(model, linear_params, cv=5)
        linear_grid_search.fit(X_train_clean, y_train_clean)
        
        # Parametri ottimali per la regressione lineare
        best_linear_params = linear_grid_search.best_params_
        
        # Addestramento del modello con i parametri ottimali
        best_linear_model = LinearRegression(**best_linear_params)
        best_linear_model.fit(X_train_clean, y_train_clean)
        
        # Valutazione del modello con cross-validation
        linear_cv_scores = cross_val_score(best_linear_model, X_train_clean, y_train_clean, cv=5)
        print("Cross-Validation Scores (Linear Regression):", linear_cv_scores)
        print("Mean CV Score (Linear Regression):", np.mean(linear_cv_scores))
        
        # Valutazione delle prestazioni sul set di test
        mse, mae, r2 = evaluate_model(name, best_linear_model, X_train_clean, X_test_clean, y_train_clean, y_test_clean)
        results.append((name, mse, mae, r2, 'After Outlier Removal'))
        
    elif name == "Lasso Regression":
        # Definizione dei parametri per Lasso Regression
        lasso_params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}

        # Ricerca dei parametri ottimali per Lasso Regression
        lasso_grid_search = GridSearchCV(model, lasso_params, cv=5)
        lasso_grid_search.fit(X_train_clean, y_train_clean)
        
        # Parametri ottimali per Lasso Regression
        best_lasso_params = lasso_grid_search.best_params_
        
        # Addestramento del modello con i parametri ottimali
        best_lasso_model = Lasso(**best_lasso_params)
        best_lasso_model.fit(X_train_clean, y_train_clean)
        
        # Valutazione del modello con cross-validation
        lasso_cv_scores = cross_val_score(best_lasso_model, X_train_clean, y_train_clean, cv=5)
        print("Cross-Validation Scores (Lasso Regression):", lasso_cv_scores)
        print("Mean CV Score (Lasso Regression):", np.mean(lasso_cv_scores))
        
        # Valutazione delle prestazioni sul set di test
        mse, mae, r2 = evaluate_model(name, best_lasso_model, X_train_clean, X_test_clean, y_train_clean, y_test_clean)
        results.append((name, mse, mae, r2, 'After Outlier Removal'))
        
    elif name == "Random Forest":
        # Definizione dei parametri per il Random Forest
        rf_params = {'n_estimators': [100, 200, 300],
                     'max_depth': [None, 10, 20, 30]}

        # Ricerca dei parametri ottimali per il Random Forest
        rf_grid_search = GridSearchCV(model, rf_params, cv=5)
        rf_grid_search.fit(X_train_clean, y_train_clean)
        
        # Parametri ottimali per il Random Forest
        best_rf_params = rf_grid_search.best_params_
        
        # Addestramento del modello con i parametri ottimali
        best_rf_model = RandomForestRegressor(**best_rf_params)
        best_rf_model.fit(X_train_clean, y_train_clean)
        
        # Valutazione del modello con cross-validation
        rf_cv_scores = cross_val_score(best_rf_model, X_train_clean, y_train_clean, cv=5)
        print("Cross-Validation Scores (Random Forest):", rf_cv_scores)
        print("Mean CV Score (Random Forest):", np.mean(rf_cv_scores))
        
        # Valutazione delle prestazioni sul set di test
        mse, mae, r2 = evaluate_model(name, best_rf_model, X_train_clean, X_test_clean, y_train_clean, y_test_clean)
        results.append((name, mse, mae, r2, 'After Outlier Removal'))

# Creazione di un DataFrame per i risultati
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'MAE', 'R2', 'Type'])

# Visualizzazione e salvataggio dei risultati
sns.barplot(x='Model', y='MSE', hue='Type', data=results_df)
plt.title('MSE dei Modelli di Regressione')
plt.savefig(os.path.join(img_folder, 'mse_comparison.png'))
plt.close()

sns.barplot(x='Model', y='MAE', hue='Type', data=results_df)
plt.title('MAE dei Modelli di Regressione')
plt.savefig(os.path.join(img_folder, 'mae_comparison.png'))
plt.close()

sns.barplot(x='Model', y='R2', hue='Type', data=results_df)
plt.title('R2 dei Modelli di Regressione')
plt.savefig(os.path.join(img_folder, 'r2_comparison.png'))
plt.close()

# Funzione per tracciare le curve di apprendimento
def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

print("\nCurva di apprendimento dei modelli:")
for name, model in models.items():
    title = f"Learning Curves ({name})"
    plot_learning_curve(model, title, X_scaled, y, cv=5, n_jobs=-1)
    plt.savefig(os.path.join(img_folder, f'learning_curve_{name.replace(" ", "_").lower()}.png'))
    plt.close()
