import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, delimiter=';')
    if 'Turbidity' not in data.columns or 'Cloud' not in data.columns:
        #Controlla se le colonne 'Turbidity' o 'Cloud' sono presenti nei dati
        print("Warning: 'Turbidity' or 'Cloud' not found in the CSV file headers.")
        return None, None, None, None  # Restituisce valori nulli se le colonne non sono presenti

    # Filtra i dati per mantenere solo le righe con 'Cloud' < 0.2
    data = data[data['Cloud'] < 0.2]
    
    X = data.drop(['Turbidity', 'Cloud'], axis=1).values
    y = data['Turbidity'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, data.columns.drop(['Turbidity', 'Cloud'])

def try_regression_models(X_train, y_train, X_test, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    # Crea un dizionario di modelli di regressione da testare

    results = {}  # Inizializza un dizionario per i risultati

    for name, model in models.items():
        model.fit(X_train, y_train)  # Adatta il modello ai dati di addestramento
        y_pred = model.predict(X_test)  # Predice i valori del set di test
        mse = mean_squared_error(y_test, y_pred)  # Calcola l'errore quadratico medio
        results[name] = mse  # Salva l'errore quadratico medio nei risultati
        print(f'{name}: MSE = {mse}')  # Stampa il risultato del modello

    return models, results  # Restituisce i modelli e i risultati
def evaluate_error(results):
    for name, mse in results.items():
        print(f'{name}: MSE = {mse}')  # Stampa l'errore quadratico medio per ciascun modello
def select_important_features(X_train, y_train, feature_names):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)  # Adatta un modello di Random Forest ai dati di addestramento

    importances = model.feature_importances_  # Ottiene le importanze delle caratteristiche
    indices = np.argsort(importances)[::-1]  # Ordina le importanze in ordine decrescente

    for i in range(len(feature_names)):
        print(f'{i + 1}. feature {feature_names[indices[i]]} ({importances[indices[i]]})')
        # Stampa le caratteristiche ordinate per importanza

    sfm = SelectFromModel(model, threshold=0.1)  # Inizializza un selezionatore di caratteristiche con una soglia
    sfm.fit(X_train, y_train)  # Adatta il selezionatore ai dati di addestramento

    selected_features = sfm.transform(X_train)  # Trasforma i dati di addestramento selezionando solo le caratteristiche importanti

    return selected_features, sfm.get_support(indices=True)  # Restituisce le caratteristiche selezionate e gli indici di quelle selezionate
def retry_with_selected_features(X_train, X_test, y_train, y_test, selected_indices):
    X_train_selected = X_train[:, selected_indices]  # Seleziona le caratteristiche importanti nel set di addestramento
    X_test_selected = X_test[:, selected_indices]  # Seleziona le caratteristiche importanti nel set di test

    models, results = try_regression_models(X_train_selected, y_train, X_test_selected, y_test)
    # Prova i modelli di regressione con le caratteristiche selezionate

    return models, results  # Restituisce i modelli e i risultati
def train_autoencoder(X_train, input_dim, encoding_dim=16):
    # Definisce lo strato di input con la dimensione specificata
    input_layer = Input(shape=(input_dim,))
    
    # Definisce lo strato codificato (bottleneck) con la dimensione di codifica specificata
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    
    # Definisce lo strato decodificato (output ricostruito) con la dimensione dell'input
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # Crea il modello autoencoder che mappa l'input ricostruito dall'input originale
    autoencoder = Model(input_layer, decoded)
    
    # Compila il modello con l'ottimizzatore Adam e la perdita MSE (mean squared error)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Addestra l'autoencoder sui dati di addestramento per 50 epoche con un batch size di 32, shufflando i dati e utilizzando il 20% dei dati per la validazione
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)
    
    # Crea un modello solo per l'encoder che mappa l'input nello spazio codificato
    encoder = Model(input_layer, encoded)
    
    return encoder  # Restituisce l'encoder addestrato


# 3/4. Ottenimento degli embeddings con l'encoder
def get_embeddings(encoder, X):
    # Utilizza l'encoder per ottenere gli embeddings dai dati
    return encoder.predict(X)  # Passa i dati attraverso l'encoder per ottenere le rappresentazioni codificate
def main():
    filepath = 'dati.csv'
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(filepath)

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return

    models, results = try_regression_models(X_train, y_train, X_test, y_test)
    evaluate_error(results)

    X_train_selected, selected_indices = select_important_features(X_train, y_train, feature_names)
    models, results = retry_with_selected_features(X_train, X_test, y_train, y_test, selected_indices)
    evaluate_error(results)

    input_dim = X_train.shape[1]
    encoder = train_autoencoder(X_train, input_dim)
    X_embedded = get_embeddings(encoder, X_train)

    models, results = retry_with_selected_features(X_train[:, selected_indices], X_test[:, selected_indices], y_train, y_test, selected_indices)
    evaluate_error(results)

if __name__ == "__main__":
    main()
