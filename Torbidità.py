# Importazione delle librerie necessarie
import pandas as pd  # Per la manipolazione dei dati
import numpy as np  # Per operazioni numeriche
import matplotlib.pyplot as plt  # Per visualizzazioni
from sklearn.model_selection import train_test_split  # Per dividere i dati in set di addestramento e test
from sklearn.preprocessing import StandardScaler  # Per normalizzare i dati
from sklearn.metrics import mean_squared_error  # Per calcolare l'errore quadratico medio
import tensorflow as tf  # Per costruire e addestrare reti neurali
from tensorflow.keras.models import Model  # Per definire modelli Keras
from tensorflow.keras.layers import Input, Dense  # Per definire strati di input e densamente connessi
from sklearn.datasets import load_iris  # Per caricare il dataset Iris
from sklearn.cluster import KMeans  # Per eseguire la clusterizzazione K-Means
from sklearn.decomposition import PCA  # Per la riduzione dimensionale con PCA
from sklearn.tree import DecisionTreeRegressor  # Per costruire un albero decisionale regressore

# 1. Caricamento e pre-elaborazione dei dati
def load_and_preprocess_data(filepath):
    # Legge la prima riga del file per ottenere gli header
    with open(filepath, 'r') as f:
        headers = f.readline().strip().split(';')
    # Controlla se 'Turbidity' e 'Cloud' sono presenti negli header
    turbidity_present = 'Turbidity' in headers
    cloud_present = 'Cloud' in headers
    if not turbidity_present:
        print("Warning: 'Turbidity' not found in the CSV file headers.")
        return None, None, None, None
    if not cloud_present:
        print("Warning: 'Cloud' not found in the CSV file headers.")
        return None, None, None, None
    # Ora carica il resto dei dati, saltando la prima riga (header)
    data = pd.read_csv(filepath, delimiter=';')
    print(data.columns)
    # Modifica qui se i nomi delle colonne sono diversi nel tuo CSV
    X = data.drop(['Turbidity', 'Cloud'], axis=1).values
    y = data['Turbidity'].values
    # Dividi i dati in set di addestramento e di test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Normalizza i dati
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# 2. Definizione e addestramento dell'autoencoder
def train_autoencoder(X_train, input_dim, encoding_dim=16):
    # Definisce lo strato di input con dimensione di input specificata
    input_layer = Input(shape=(input_dim,))
    # Definisce lo strato codificato (bottleneck)
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    # Definisce lo strato decodificato (output ricostruito)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    # Crea il modello autoencoder
    autoencoder = Model(input_layer, decoded)
    # Compila il modello con l'ottimizzatore Adam e la perdita MSE
    autoencoder.compile(optimizer='adam', loss='mse')
    # Addestra l'autoencoder sui dati di addestramento
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)
    # Crea un modello solo per l'encoder
    encoder = Model(input_layer, encoded)
    return encoder

# 3/4. Ottenimento degli embeddings con l'encoder
def get_embeddings(encoder, X):
    # Utilizza l'encoder per ottenere gli embeddings dai dati
    return encoder.predict(X)

# 5. Produzione del dataset Iris (esempio base)
def load_iris_dataset():
    # Carica il dataset Iris
    iris = load_iris()
    return iris.data, iris.target

# 6. Clusterizzazione e riduzione delle feature
def perform_clustering(X_embedded):
    # Esegue la clusterizzazione K-Means su embeddings con 6 cluster
    kmeans = KMeans(n_clusters=6, random_state=42)
    clusters = kmeans.fit_predict(X_embedded)
    # Riduce le dimensioni a 2 componenti principali per visualizzazione
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_embedded)
    # Visualizza i cluster in un grafico a dispersione
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
    plt.title('Clusterizzazione con K-Means su Autoencoder Embeddings')
    plt.show()
    return clusters

# 7. Confronto con un albero decisionale
def compare_with_decision_tree(X, y, X_embedded):
    # Addestra un albero decisionale sui dati originali
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X, y)
    y_pred_tree = tree.predict(X)
    mse_tree = mean_squared_error(y, y_pred_tree)
    
    # Addestra un albero decisionale sui embeddings
    tree_embedded = DecisionTreeRegressor(random_state=42)
    tree_embedded.fit(X_embedded, y)
    y_pred_tree_embedded = tree_embedded.predict(X_embedded)
    mse_tree_embedded = mean_squared_error(y, y_pred_tree_embedded)
    
    # Stampa i risultati degli MSE per confronto
    print(f'MSE su dati originali: {mse_tree}')
    print(f'MSE su embeddings: {mse_tree_embedded}')

# Funzione principale
def main():
    # Caricamento e pre-elaborazione dei dati
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

# Esegue la funzione principale se lo script è eseguito direttamente
if __name__ == "__main__":
    main()
