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
    # Legge il file CSV con pandas, specificando il delimitatore come ';'
    data = pd.read_csv(filepath, delimiter=';')  # Usa il delimitatore appropriato (; in questo caso)
    
    # Controlla se le colonne 'Turbidity' e 'Cloud' sono presenti nel DataFrame
    turbidity_present = 'Turbidity' in data.columns
    cloud_present = 'Cloud' in data.columns
    
    # Se la colonna 'Turbidity' non è presente, stampa un avviso e termina la funzione
    if not turbidity_present:
        print("Warning: 'Turbidity' not found in the CSV file headers.")
        return None, None, None, None
    
    # Se la colonna 'Cloud' non è presente, stampa un avviso e termina la funzione
    if not cloud_present:
        print("Warning: 'Cloud' not found in the CSV file headers.")
        return None, None, None, None
    
    # Filtra i dati per rimuovere le righe con alta copertura nuvolosa (Cloud >= 0.2)
    data = data[data['Cloud'] < 0.2]  # Modifica il valore soglia di 'Cloud' in base alla tua definizione di alta copertura
    
    # Seleziona le feature (tutte le colonne eccetto 'Turbidity' e 'Cloud') e la target ('Turbidity')
    X = data.drop(['Turbidity', 'Cloud'], axis=1).values
    y = data['Turbidity'].values
    
    # Dividi i dati in set di addestramento e di test (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizza i dati usando StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Calcola la media e la deviazione standard sui dati di addestramento e trasforma i dati
    X_test = scaler.transform(X_test)  # Trasforma i dati di test usando i parametri calcolati sui dati di addestramento
    
    return X_train, X_test, y_train, y_test  # Restituisce i set di addestramento e test, normalizzati


# 2. Definizione e addestramento dell'autoencoder
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


# 5. Produzione del dataset Iris (esempio base)
def load_iris_dataset():
    # Carica il dataset Iris utilizzando una funzione predefinita di scikit-learn
    iris = load_iris()
    return iris.data, iris.target  # Restituisce i dati e i target del dataset Iris


# 6. Clusterizzazione e riduzione delle feature
def perform_clustering(X_embedded):
    # Esegue la clusterizzazione K-Means sugli embeddings con 6 cluster
    kmeans = KMeans(n_clusters=6, random_state=42)
    clusters = kmeans.fit_predict(X_embedded)  # Assegna ciascun punto al cluster più vicino
    
    # Riduce le dimensioni a 2 componenti principali per visualizzazione con PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_embedded)  # Trasforma gli embeddings riducendo le dimensioni a 2 componenti
    
    # Visualizza i cluster in un grafico a dispersione
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)  # Crea uno scatter plot delle componenti principali colorate per cluster
    plt.title('Clusterizzazione con K-Means su Autoencoder Embeddings')  # Aggiunge il titolo al grafico
    plt.show()  # Mostra il grafico
    
    return clusters  # Restituisce i cluster assegnati ai punti


# 7. Confronto con un albero decisionale
def compare_with_decision_tree(X, y, X_embedded):
    # Addestra un albero decisionale sui dati originali
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X, y)  # Addestra l'albero decisionale sui dati originali
    y_pred_tree = tree.predict(X)  # Predice i valori target sui dati originali
    mse_tree = mean_squared_error(y, y_pred_tree)  # Calcola l'errore quadratico medio sulle predizioni
    
    # Addestra un albero decisionale sui embeddings
    tree_embedded = DecisionTreeRegressor(random_state=42)
    tree_embedded.fit(X_embedded, y)  # Addestra l'albero decisionale sugli embeddings
    y_pred_tree_embedded = tree_embedded.predict(X_embedded)  # Predice i valori target sugli embeddings
    mse_tree_embedded = mean_squared_error(y, y_pred_tree_embedded)  # Calcola l'errore quadratico medio sulle predizioni
    
    # Stampa i risultati degli MSE per confronto
    print(f'MSE su dati originali: {mse_tree}')  # Stampa l'errore quadratico medio sui dati originali
    print(f'MSE su embeddings: {mse_tree_embedded}')  # Stampa l'errore quadratico medio sugli embeddings


# Funzione principale
def main():
    # Sostituisci 'path_to_dataset.csv' con il percorso reale del tuo file CSV
    X_train, X_test, y_train, y_test = load_and_preprocess_data('dati.csv')
    
    # Verifica se i dati sono stati caricati correttamente
    if X_train is None or X_test is None or y_train is None or y_test is None:
        return
    
    # Addestramento dell'autoencoder
    input_dim = X_train.shape[1]  # Ottiene la dimensione dell'input dai dati di addestramento
    encoder = train_autoencoder(X_train, input_dim)  # Addestra l'autoencoder e ottiene l'encoder
    
    # Ottenimento delle embeddings
    X_embedded = get_embeddings(encoder, X_train)  # Genera gli embeddings dai dati di addestramento
    
    # Esempio con dataset Iris
    X_iris, y_iris = load_iris_dataset()  # Carica il dataset Iris
    
    # Clusterizzazione e visualizzazione
    perform_clustering(X_embedded)  # Esegue la clusterizzazione e visualizza i risultati
    
    # Confronto con un albero decisionale
    compare_with_decision_tree(X_train, y_train, X_embedded)  # Confronta l'errore di un albero decisionale sui dati originali e sugli embeddings

# Esegue la funzione principale se lo script è eseguito direttamente
if __name__ == "__main__":
    main()  # Chiamata alla funzione principale