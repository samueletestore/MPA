import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import os

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

# Identificazione dei Cluster Anomali con K-Means
kmeans = KMeans(n_clusters=2)  # Consideriamo 2 cluster per l'esempio
cluster_labels = kmeans.fit_predict(X_scaled)
cluster_centers = kmeans.cluster_centers_

# Visualizzazione dei Cluster
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='red', s=100)
plt.title('Clustering dei Dati')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig(os.path.join(img_folder, 'clustering.png'))
plt.close()

# Funzione per valutare i modelli
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - MSE: {mse}, MAE: {mae}, R2: {r2}")
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
    mse, mae, r2 = evaluate_model(name, model, X_train_clean, X_test_clean, y_train_clean, y_test_clean)
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
