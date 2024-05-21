import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor

# 1. Caricamento e pre-elaborazione dei dati
def load_and_preprocess_data(filepath):
    # Read the first line of the file to get the headers
    with open(filepath, 'r') as f:
        headers = f.readline().strip().split(';')

    # Check if 'Turbidity' and 'Cloud' are present in headers
    turbidity_present = 'Turbidity' in headers
    cloud_present = 'Cloud' in headers

    if not turbidity_present:
        print("Warning: 'Turbidity' not found in the CSV file headers.")
        return None, None, None, None

    if not cloud_present:
        print("Warning: 'Cloud' not found in the CSV file headers.")
        return None, None, None, None

    # Now load the rest of the data, skipping the first row (header)
    data = pd.read_csv(filepath, delimiter=';')
    print(data.columns)
    # Modifica qui se i nomi delle colonne sono diversi nel tuo CSV
    X = data.drop(['Turbidity', 'Cloud'], axis=1).values
    y = data['Turbidity'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# 2. Definizione e addestramento dell'autoencoder
def train_autoencoder(X_train, input_dim, encoding_dim=16):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)
    encoder = Model(input_layer, encoded)
    return encoder

# 3/4. Embedding con l'encoder
def get_embeddings(encoder, X):
    return encoder.predict(X)

# 5. Produzione del dataset Iris (esempio base)
def load_iris_dataset():
    iris = load_iris()
    return iris.data, iris.target

# 6. Clusterizzazione e riduzione delle feature
def perform_clustering(X_embedded):
    kmeans = KMeans(n_clusters=6, random_state=42)
    clusters = kmeans.fit_predict(X_embedded)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_embedded)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
    plt.title('Clusterizzazione con K-Means su Autoencoder Embeddings')
    plt.show()
    return clusters

# 7. Confronto con un albero decisionale
def compare_with_decision_tree(X, y, X_embedded):
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X, y)
    y_pred_tree = tree.predict(X)
    mse_tree = mean_squared_error(y, y_pred_tree)
    
    tree_embedded = DecisionTreeRegressor(random_state=42)
    tree_embedded.fit(X_embedded, y)
    y_pred_tree_embedded = tree_embedded.predict(X_embedded)
    mse_tree_embedded = mean_squared_error(y, y_pred_tree_embedded)
    
    print(f'MSE su dati originali: {mse_tree}')
    print(f'MSE su embeddings: {mse_tree_embedded}')

# Funzione principale
def main():
    # Sostituisci 'path_to_dataset.csv' con il percorso reale del tuo file CSV
    X_train, X_test, y_train, y_test = load_and_preprocess_data('dati.csv')
    
    # Addestramento dell'autoencoder
    input_dim = X_train.shape[1]
    encoder = train_autoencoder(X_train, input_dim)
    
    # Ottenimento delle embeddings
    X_embedded = get_embeddings(encoder, X_train)
    
    # Esempio con dataset Iris
    X_iris, y_iris = load_iris_dataset()
    
    # Clusterizzazione e visualizzazione
    perform_clustering(X_embedded)
    
    # Confronto con un albero decisionale
    compare_with_decision_tree(X_train, y_train, X_embedded)

if __name__ == "__main__":
    main()
