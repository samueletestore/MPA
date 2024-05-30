import pandas as pd
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt

# Carica il dataset
df = pd.read_csv("dati.csv", delimiter=';')

# Nomina delle colonne
feature_names = ['Turbidity', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 
                 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12', 'Cloud']
df.columns = feature_names

# Filtra i dati basati sulla copertura nuvolosa
df = df[df['Cloud'] < 0.01]  # Soglia per la copertura nuvolosa

# Separazione in caratteristiche e target
X = df.drop(['Turbidity', 'Cloud'], axis=1)
y = df['Turbidity']

# Divisione dei dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Standardizzazione dei dati
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definizione del modello di rete neurale
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()

# Addestramento del modello
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100)

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
plt.show()

acc = history.history['mae']
val_acc = history.history['val_mae']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Previsione sui dati di test
predictions = model.predict(X_test_scaled[:5])
print("Predicted values are: ", predictions)
print("Real values are: ", y_test[:5].values)

# Valutazione del modello di rete neurale
mse_neural, mae_neural = model.evaluate(X_test_scaled, y_test)
print('Mean squared error from neural net: ', mse_neural)
print('Mean absolute error from neural net: ', mae_neural)

# Confronto con altri modelli

# Regressione Lineare
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print('Mean squared error from linear regression: ', mse_lr)
print('Mean absolute error from linear regression: ', mae_lr)

# Decision Tree
tree = DecisionTreeRegressor()
tree.fit(X_train_scaled, y_train)
y_pred_tree = tree.predict(X_test_scaled)
mse_dt = mean_squared_error(y_test, y_pred_tree)
mae_dt = mean_absolute_error(y_test, y_pred_tree)
print('Mean squared error using decision tree: ', mse_dt)
print('Mean absolute error using decision tree: ', mae_dt)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=30, random_state=30)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print('Mean squared error using Random Forest: ', mse_rf)
print('Mean absolute error Using Random Forest: ', mae_rf)

# Importanza delle caratteristiche
feature_list = list(X.columns)
feature_imp = pd.Series(rf_model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)

# Visualizzazione dell'importanza delle caratteristiche
feature_imp.plot(kind='bar')
plt.title('Feature Importance')
plt.show()
