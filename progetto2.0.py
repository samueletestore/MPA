import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, delimiter=';')
    if 'Turbidity' not in data.columns or 'Cloud' not in data.columns:
        print("Warning: 'Turbidity' or 'Cloud' not found in the CSV file headers.")
        return None, None, None, None, None

    data = data[data['Cloud'] < 0.2]
    
    X = data.drop(['Turbidity', 'Cloud'], axis=1).values
    y = data['Turbidity'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, data.columns.drop(['Turbidity', 'Cloud'])

def select_important_features(X_train, y_train, feature_names):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(len(feature_names)):
        print(f'{i + 1}. feature {feature_names[indices[i]]} ({importances[indices[i]]})')

    sfm = SelectFromModel(model, threshold='mean')
    sfm.fit(X_train, y_train)

    selected_features = sfm.transform(X_train)
    selected_indices = sfm.get_support(indices=True)
    
    print(f"Selected feature indices: {selected_indices}")
    print(f"Number of features selected: {selected_features.shape[1]}")
    
    return selected_features, selected_indices

def retry_with_selected_features(X_train, X_test, y_train, y_test, selected_indices):
    print(f"Shape of X_train before selection: {X_train.shape}")
    print(f"Shape of X_test before selection: {X_test.shape}")

    selected_indices_in_embedded = [i for i in selected_indices if i < X_train.shape[1]]
    if not selected_indices_in_embedded:
        print("No valid indices in embedded space. Exiting.")
        return None, None

    X_train_selected = X_train[:, selected_indices_in_embedded]
    X_test_selected = X_test[:, selected_indices_in_embedded]

    print(f"Shape of X_train after selection: {X_train_selected.shape}")
    print(f"Shape of X_test after selection: {X_test_selected.shape}")

    models, results = try_regression_models(X_train_selected, y_train, X_test_selected, y_test)

    return models, results

def try_regression_models(X_train, y_train, X_test, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        results[name] = mse
        print(f'{name}: MSE = {mse}')
        
    return models, results

def evaluate_error(results):
    for name, mse in results.items():
        print(f'{name}: MSE = {mse}')

def train_autoencoder(X_train, input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2)
    
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    encoder = Model(input_layer, encoded)
    
    return encoder

def get_embeddings(encoder, X):
    return encoder.predict(X)

def main():
    filepath = "dati.csv"
    
    # Carica e preelabora i dati
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(filepath)
    
    if X_train is None:
        return
    
    # Seleziona le caratteristiche importanti
    selected_features, selected_indices = select_important_features(X_train, y_train, feature_names)
    
    # Analizza le bande spettrali per identificare quelle importanti
    important_bands = [feature_names[i] for i in selected_indices]
    print("Bande spettrali importanti:", important_bands)
    
    # Riprova i modelli di regressione con le caratteristiche selezionate
    models, results = retry_with_selected_features(X_train, X_test, y_train, y_test, selected_indices)
    
    # Valuta l'errore atteso
    print("Valutazione dell'errore atteso:")
    evaluate_error(results)
    
    # Addestra un autoencoder
    input_dim = X_train.shape[1]
    encoder = train_autoencoder(X_train, input_dim)
    
    # Ottieni le rappresentazioni codificate
    encoded_train = get_embeddings(encoder, X_train)
    encoded_test = get_embeddings(encoder, X_test)

if __name__ == "__main__":
    main()


