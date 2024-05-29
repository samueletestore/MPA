import pandas as pd
import numpy as np
import os
import dataframe_image as dfi
import shutil
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Caricare i dati
data = pd.read_csv("dati.csv", delimiter=';')

# Filtrare i dati per la copertura nuvolosa
data = data[data['Cloud'] <= 0.2]

# Analisi statistica delle bande spettrali
stats = data.describe().T[['mean', 'std', 'min', 'max']]
stats = stats.round(2)
print(stats)

# Rimuovere gli outlier utilizzando Isolation Forest
iso = IsolationForest(contamination=0.1)
data['outliers'] = iso.fit_predict(data.iloc[:, 1:-1])
data = data[data['outliers'] == 1].drop('outliers', axis=1)

# Creazione della cartella img
if os.path.exists('img'):
    shutil.rmtree('img')
os.makedirs('img')

dfi.export(stats, 'img/stats1.png')

# Boxplot per le bande spettrali
plt.figure(figsize=(15, 10))
sns.boxplot(data=data.iloc[:, 1:-1])
plt.title('Boxplot delle bande spettrali')
plt.savefig('img/boxplot_bande_spettrali.png')
plt.close()

# Istogrammi per le bande spettrali
data.iloc[:, 1:-1].hist(bins=20, figsize=(15, 10))
plt.suptitle('Istogrammi delle bande spettrali')
plt.savefig('img/istogrammi_bande_spettrali.png')
plt.close()

# Istogramma della correlazione tra bande spettrali e torbidità
correlations = data.corr()['Turbidity'].iloc[1:-1]
correlations.plot(kind='bar', figsize=(12, 6))
plt.title('Correlazione tra bande spettrali e torbidità')
plt.savefig('img/correlazione_bande_torbidita.png')
plt.close()

# Normalizzazione dei dati
scaler = StandardScaler()
X = data.iloc[:, 1:-1]
y = data['Turbidity']
X_scaled = scaler.fit_transform(X)

# Selezione delle feature più importanti
# In questo caso usiamo tutte le bande, ma si potrebbe usare una tecnica di selezione delle feature come l'analisi di importanza delle feature
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Addestramento del modello KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Addestramento del modello SVM
svm = SVR(kernel='rbf')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Valutazione dei modelli
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

mae_knn, mse_knn, r2_knn = evaluate_model(y_test, y_pred_knn)
mae_svm, mse_svm, r2_svm = evaluate_model(y_test, y_pred_svm)

# Confronto delle performance dei modelli
performance = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'R2'],
    'KNN': [mae_knn, mse_knn, r2_knn],
    'SVM': [mae_svm, mse_svm, r2_svm]
})
performance.set_index('Metric', inplace=True)
performance.plot(kind='bar', figsize=(10, 6))
plt.title('Confronto delle performance dei modelli')
plt.savefig('img/confronto_performance_modelli.png')
plt.close()

# Visualizzazione della torbidità predetta vs osservata per KNN
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_knn, alpha=0.5, label='KNN')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Torbidità osservata')
plt.ylabel('Torbidità predetta')
plt.title('Torbidità predetta vs osservata (KNN)')
plt.legend()
plt.savefig('img/torbidita_predetta_vs_osservata_knn.png')
plt.close()

# Visualizzazione della torbidità predetta vs osservata per SVM
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_svm, alpha=0.5, label='SVM', color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Torbidità osservata')
plt.ylabel('Torbidità predetta')
plt.title('Torbidità predetta vs osservata (SVM)')
plt.legend()
plt.savefig('img/torbidita_predetta_vs_osservata_svm.png')
plt.close()

# Ottimizzazione dei parametri per SVM utilizzando Grid Search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=2, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# Migliori parametri trovati
print("Migliori parametri trovati dalla Grid Search: ", grid.best_params_)

# Predizioni con il modello ottimizzato
y_pred_svm_opt = grid.predict(X_test)

# Valutazione del modello ottimizzato
mae_svm_opt, mse_svm_opt, r2_svm_opt = evaluate_model(y_test, y_pred_svm_opt)

# Visualizzazione della torbidità predetta vs osservata per SVM ottimizzato
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_svm_opt, alpha=0.5, label='SVM Ottimizzato', color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Torbidità osservata')
plt.ylabel('Torbidità predetta')
plt.title('Torbidità predetta vs osservata (SVM Ottimizzato)')
plt.legend()
plt.savefig('img/torbidita_predetta_vs_osservata_svm_ottimizzato.png')
plt.close()

# Curva di apprendimento per KNN
train_sizes, train_scores, test_scores = learning_curve(knn, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', label='Training error')
plt.plot(train_sizes, test_scores_mean, 'o-', label='Cross-validation error')
plt.title('Curva di apprendimento per KNN')
plt.xlabel('Dimensione del training set')
plt.ylabel('Errore MSE')
plt.legend(loc='best')
plt.savefig('img/curva_apprendimento_knn.png')
plt.close()

# Curva di apprendimento per SVM
train_sizes, train_scores, test_scores = learning_curve(svm, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', label='Training error')
plt.plot(train_sizes, test_scores_mean, 'o-', label='Cross-validation error')
plt.title('Curva di apprendimento per SVM')
plt.xlabel('Dimensione del training set')
plt.ylabel('Errore MSE')
plt.legend(loc='best')
plt.savefig('img/curva_apprendimento_svm.png')
plt.close()

# Curva di apprendimento per SVM ottimizzato
train_sizes, train_scores, test_scores = learning_curve(grid.best_estimator_, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', label='Training error')
plt.plot(train_sizes, test_scores_mean, 'o-', label='Cross-validation error')
plt.title('Curva di apprendimento per SVM Ottimizzato')
plt.xlabel('Dimensione del training set')
plt.ylabel('Errore MSE')
plt.legend(loc='best')
plt.savefig('img/curva_apprendimento_svm_ottimizzato.png')
plt.close()

# Calcolo delle statistiche della colonna 'Turbidity'
turbidity_stats = data['Turbidity'].describe()[['mean', 'std', 'min', 'max']]
turbidity_stats_df = pd.DataFrame(turbidity_stats).T
turbidity_stats_df.index = ['Turbidity']
turbidity_stats_df = turbidity_stats_df.round(2)
print(turbidity_stats_df)
dfi.export(turbidity_stats_df, 'img/turbidity_stats1.png')

# Salvataggio delle statistiche della torbidità come immagine
fig, ax = plt.subplots(figsize=(8, 2)) # Dimensione della figura
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=turbidity_stats_df.values,
                 colLabels=turbidity_stats_df.columns,
                 rowLabels=turbidity_stats_df.index,
                 cellLoc='center',
                 loc='center')
plt.title('Statistiche della Torbidità')
plt.savefig('img/turbidity_stats.png')
plt.close()