import pandas as pd  # Importa la libreria pandas per la manipolazione dei dati
import numpy as np  # Importa la libreria numpy per le operazioni numeriche
import matplotlib.pyplot as plt  # Importa la libreria matplotlib per la creazione di grafici
import seaborn as sns  # Importa la libreria seaborn per la visualizzazione dei dati
from sklearn.model_selection import train_test_split, cross_val_score  # Importa metodi per la suddivisione del dataset e la valutazione incrociata
from sklearn.preprocessing import StandardScaler  # Importa il metodo per la standardizzazione delle caratteristiche
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Importa i modelli di regressione lineare, Ridge e Lasso
from sklearn.ensemble import RandomForestRegressor  # Importa il modello di regressione Random Forest
from sklearn.metrics import mean_squared_error  # Importa il metodo per il calcolo dell'errore quadratico medio
from sklearn.feature_selection import SelectFromModel  # Importa il metodo per la selezione delle caratteristiche basato su un modello
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, delimiter=';')  # Carica i dati da un file CSV con delimitatore ';'

    if 'Turbidity' not in data.columns or 'Cloud' not in data.columns:
        # Controlla se le colonne 'Turbidity' o 'Cloud' sono presenti nei dati
        print("Warning: 'Turbidity' or 'Cloud' not found in the CSV file headers.")
        return None, None, None, None  # Restituisce valori nulli se le colonne non sono presenti

    data = data[data['Cloud'] < 0.2]  # Filtra i dati per mantenere solo le righe con 'Cloud' < 0.2
    X = data.drop(['Turbidity', 'Cloud'], axis=1).values  # Rimuove le colonne 'Turbidity' e 'Cloud' e salva le altre come array di valori
    y = data['Turbidity'].values  # Salva i valori della colonna 'Turbidity' come target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Divide i dati in set di addestramento e test

    scaler = StandardScaler()  # Inizializza uno scaler per standardizzare le caratteristiche
    X_train = scaler.fit_transform(X_train)  # Adatta e trasforma il set di addestramento
    X_test = scaler.transform(X_test)  # Trasforma il set di test con lo stesso scaler

    return X_train, X_test, y_train, y_test, data.columns.drop(['Turbidity', 'Cloud'])  # Restituisce i dati pre-elaborati e i nomi delle caratteristiche

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
def main():
    filepath = 'dati.csv'  # Definisce il percorso del file CSV
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(filepath)
    # Carica e pre-elabora i dati

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return  # Esce dalla funzione se i dati non sono stati caricati correttamente

    data = pd.read_csv(filepath, delimiter=';')  # Ricarica i dati per l'analisi

    models, results = try_regression_models(X_train, y_train, X_test, y_test)  # Prova diversi modelli di regressione
    evaluate_error(results)  # Valuta l'errore dei modelli

    X_train_selected, selected_indices = select_important_features(X_train, y_train, feature_names)
    # Seleziona le caratteristiche importanti
    models, results = retry_with_selected_features(X_train, X_test, y_train, y_test, selected_indices)
    # Ripete il processo con le caratteristiche selezionate
    evaluate_error(results)  # Valuta di nuovo l'errore dei modelli

if __name__ == "__main__":
    main()  # Esegue la funzione principale se lo script è eseguito direttamente
