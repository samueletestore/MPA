import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt

from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os

# Funzione per rimuovere tutti i file in una cartella
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

# Crea la cartella img-indiano se non esiste e rimuovi i file se esistono
img_folder = 'img-indiano'
if not os.path.exists(img_folder):
    os.makedirs(img_folder)
else:
    clear_folder(img_folder)

# Carica il dataset
df = read_csv("dati.csv", delimiter=';')

# Nomina delle colonne
feature_names = ['Turbidity', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12', 'Cloud']
df.columns = feature_names

# Filtra i dati basati sulla copertura nuvolosa
df = df[df['Cloud'] < 0.2]  # Soglia per la copertura nuvolosa

# Rilevazione degli outlier con Z-score
def detect_outliers_zscore(df, feature, threshold=3):
    z_scores = (df[feature] - df[feature].mean()) / df[feature].std()
    outliers = df[np.abs(z_scores) > threshold]
    return outliers

outliers = []
outlier_indices = set()
for feature in df.columns:
    if feature != 'Cloud' and feature != 'Turbidity':
        feature_outliers = detect_outliers_zscore(df, feature)
        outliers.append(feature_outliers)

        # Visualizzazione degli outlier rilevati per ogni caratteristica
        plt.scatter(feature_outliers.index, feature_outliers[feature], c='r', label='Outlier')
        plt.scatter(df.index, df[feature], c='b', label='Non-outlier', alpha=0.3)
        plt.title(f'Outliers in {feature}')
        plt.xlabel('Index')
        plt.ylabel(feature)
        plt.legend()
        plt.savefig(f'img-indiano/outliers_{feature}.png')
        plt.close()

        outlier_indices.update(feature_outliers.index)

# Rimozione degli outlier
df_no_outliers = df.drop(outlier_indices)

# Analisi esplorativa dei dati
sns.pairplot(df_no_outliers, diag_kind='kde')
plt.savefig('img-indiano/pairplot.png')
plt.close()

# Analisi della distribuzione dei valori nelle bande e ricerca di outliers
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_no_outliers.iloc[:, 1:-1])
plt.title('Boxplot delle bande spettrali')
plt.savefig('img-indiano/boxplot_bande_spettrali.png')
plt.close()

# Separazione in caratteristiche e target
X = df_no_outliers.drop(['Turbidity', 'Cloud'], axis=1)
y = df_no_outliers['Turbidity']

# Divisione dei dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Standardizzazione dei dati
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definizione del modello di rete neurale
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()

# Addestramento del modello
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50)

# Visualizzazione delle perdite di addestramento e validazione
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('img-indiano/training_validation_loss.png')
plt.close()

acc = history.history['mae']
val_acc = history.history['val_mae']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.savefig('img-indiano/training_validation_mae.png')
plt.close()

# Previsione sui dati di test
predictions = model.predict(X_test_scaled[:])
print("Predicted values are: ", predictions)
print("Real values are: ", y_test[:].values)

# Creazione del grafico a punti con il colore dei marcatori desiderato
plt.scatter(y_test, predictions, alpha=0.5, color='#6CE5E8')  # Grafico a punti

# Linea di best fitting
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z[0])  
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

plt.gca().set_facecolor('#DAE9FF')
plt.title('Predicted vs Real Values with Best Fit Line')
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.savefig('img-indiano/predicted_vs_real_best_fit.png')
plt.close()



# Valutazione del modello di rete neurale
mse_neural, mae_neural = model.evaluate(X_test_scaled, y_test)
r2_neural = r2_score(y_test, model.predict(X_test_scaled))
print('Mean squared error from neural net: ', mse_neural)
print('Mean absolute error from neural net: ', mae_neural)
print('R2 score from neural net: ', r2_neural)

# Confronto con altri modelli

# Regressione Lineare
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print('Mean squared error from linear regression: ', mse_lr)
print('Mean absolute error from linear regression: ', mae_lr)
print('R2 score from linear regression: ', r2_lr)

# Decision Tree
tree = DecisionTreeRegressor()
tree.fit(X_train_scaled, y_train)
y_pred_tree = tree.predict(X_test_scaled)
mse_dt = mean_squared_error(y_test, y_pred_tree)
mae_dt = mean_absolute_error(y_test, y_pred_tree)
r2_dt = r2_score(y_test, y_pred_tree)
print('Mean squared error using decision tree: ', mse_dt)
print('Mean absolute error using decision tree: ', mae_dt)
print('R2 score using decision tree: ', r2_dt)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=30, random_state=30)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print('Mean squared error using Random Forest: ', mse_rf)
print('Mean absolute error Using Random Forest: ', mae_rf)
print('R2 score Using Random Forest: ', r2_rf)

# Istogrammi per il confronto delle metriche per ogni modello
models = ['Linear Regression', 'Decision Tree', 'Random Forest']
mse_values = [mse_neural, mse_lr, mse_dt, mse_rf]
mae_values = [mae_neural, mae_lr, mae_dt, mae_rf]
r2_values = [r2_neural, r2_lr, r2_dt, r2_rf]

model_color = '#41B8D5'
nn_color = '#6CE5E8'
bg_color = '#DAE9FF'  # Colore di sfondo

for i, model in enumerate(models):
    fig, axs = plt.subplots(1, 3, figsize=(9, 10))

    # MSE
    axs[0].bar([model, 'Neural Net'], [mse_values[i + 1], mse_values[0]], color=[model_color, nn_color])
    axs[0].set_title(f'Mean Squared Error \n {model} \nvs\n Neural Net')
    axs[0].set_ylabel('MSE')
    axs[0].set_facecolor(bg_color)  # Imposta il colore di sfondo

    # MAE
    axs[1].bar([model, 'Neural Net'], [mae_values[i + 1], mae_values[0]], color=[model_color, nn_color])
    axs[1].set_title(f'Mean Absolute Error \n {model} \nvs\n Neural Net')
    axs[1].set_ylabel('MAE')
    axs[1].set_facecolor(bg_color)  # Imposta il colore di sfondo

    # R2
    axs[2].bar([model, 'Neural Net'], [r2_values[i + 1], r2_values[0]], color=[model_color, nn_color])
    axs[2].set_title(f'R2 Score \n {model} \nvs\n Neural Net')
    axs[2].set_ylabel('R2')
    axs[2].set_facecolor(bg_color)  # Imposta il colore di sfondo

    plt.tight_layout()
    plt.savefig(f'img-indiano/model_comparison_{model}.png')
    plt.close()





# Importanza delle caratteristiche con Regressione Lineare
feature_list = list(X.columns)
coef = lr_model.coef_
coef_abs = np.abs(coef)  # Calcola il valore assoluto dei coefficienti
feature_imp_lr = pd.Series(coef_abs, index=feature_list).sort_values(ascending=False)  # Ordina in base al valore assoluto
print(feature_imp_lr)

# Visualizzazione dell'importanza delle caratteristiche con Regressione Lineare
feature_imp_lr.plot(kind='bar')
plt.title('Feature Importance with Linear Regression (Absolute Values)')
plt.savefig('img-indiano/feature_importance_lr_abs.png')
plt.close()

# Modelli semplici con singole bande spettrali
for band in feature_names[1:-1]:  # Escludiamo 'Turbidity' e 'Cloud'
    X_single_band = df_no_outliers[[band]]
    X_train_sb, X_test_sb, y_train_sb, y_test_sb = train_test_split(X_single_band, y, test_size=0.2, random_state=20)
    scaler_sb = StandardScaler()
    X_train_sb_scaled = scaler_sb.fit_transform(X_train_sb)
    X_test_sb_scaled = scaler_sb.transform(X_test_sb)

    lr_model_sb = LinearRegression()
    lr_model_sb.fit(X_train_sb_scaled, y_train_sb)
    y_pred_lr_sb = lr_model_sb.predict(X_test_sb_scaled)
    mse_lr_sb = mean_squared_error(y_test_sb, y_pred_lr_sb)
    mae_lr_sb = mean_absolute_error(y_test_sb, y_pred_lr_sb)
    r2_lr_sb = r2_score(y_test_sb, y_pred_lr_sb)
    print(f'Mean squared error from linear regression using {band}: ', mse_lr_sb)
    print(f'Mean absolute error from linear regression using {band}: ', mae_lr_sb)
    print(f'R2 score from linear regression using {band}: ', r2_lr_sb)

# Selezione delle bande spettrali più importanti
top_bands = feature_imp_lr.index[:5]  # Selezioniamo le prime 5 bande più importanti
X_top_bands = df_no_outliers[top_bands]

print('\nTop bands: ', top_bands, '\n')

X_train_tb, X_test_tb, y_train_tb, y_test_tb = train_test_split(X_top_bands, y, test_size=0.2, random_state=20)

scaler_tb = StandardScaler()
X_train_tb_scaled = scaler_tb.fit_transform(X_train_tb)
X_test_tb_scaled = scaler_tb.transform(X_test_tb)

# Modello di rete neurale con bande spettrali selezionate
model_top_bands = Sequential()
model_top_bands.add(Dense(64, input_dim=X_train_tb.shape[1], activation='relu'))
model_top_bands.add(Dense(32, activation='relu'))
model_top_bands.add(Dense(1, activation='linear'))

model_top_bands.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model_top_bands.summary()

history_top_bands = model_top_bands.fit(X_train_tb_scaled, y_train_tb, validation_split=0.2, epochs=50)

mse_neural_tb, mae_neural_tb = model_top_bands.evaluate(X_test_tb_scaled, y_test_tb)
r2_neural_tb = r2_score(y_test_tb, model_top_bands.predict(X_test_tb_scaled))
print('Mean squared error from neural net with top bands: ', mse_neural_tb)
print('Mean absolute error from neural net with top bands: ', mae_neural_tb)
print('R2 score from neural net with top bands: ', r2_neural_tb)



# Previsione sui dati di test utilizzando le migliori bande
predictions_top_bands = model_top_bands.predict(X_test_tb_scaled)
print("Predicted values with top bands are: ", predictions_top_bands)
print("Real values are: ", y_test_tb.values)

# Creazione del grafico a punti con il colore dei marcatori desiderato
plt.scatter(y_test_tb, predictions_top_bands, alpha=0.5, color='#6CE5E8')  # Grafico a punti

# Linea di best fitting
z = np.polyfit(y_test_tb, predictions_top_bands, 1)
p = np.poly1d(z[0])
plt.plot([min(y_test_tb), max(y_test_tb)], [min(y_test_tb), max(y_test_tb)], color='red')

plt.gca().set_facecolor('#DAE9FF')
plt.title('Predicted vs Real Values with Best Fit Line (Top Bands)')
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.savefig('img-indiano/predicted_vs_real_best_fit_top_bands.png')
plt.close()