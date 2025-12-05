"""
==========================================================
ROZWIĄZANIE ZADANIA: KLASYFIKACJA DRZEWEM DECYZYJNYM I SVM
==========================================================

OPIS PROBLEMU:
Celem skryptu jest wykonanie kompleksowej analizy klasyfikacyjnej
na dwóch zbiorach danych: Wheat Seeds (wieloklasowa) i Breast Cancer
(binarna). Analiza obejmuje:
1. Trening i ocenę modeli Drzewa Decyzyjnego (DT) i SVM dla obu zbiorów.
2. Wizualizację danych i wytrenowanych modeli (4 wykresy).
3. Demonstrację klasyfikacji dla przykładowych danych wejściowych.
4. Eksperymentowanie z różnymi funkcjami jądra (kernel functions) w SVM
   na zbiorze Breast Cancer (linear, poly, rbf) oraz podsumowanie ich wpływu.

AUTORZY:
 -Oskar Skomra
 -Marek Jenczyk

INSTRUKCJA UŻYCIA:
1. Upewnij się, że masz zainstalowane biblioteki: pandas, numpy, matplotlib, scikit-learn.
   (pip install pandas numpy matplotlib scikit-learn)
2. Uruchom skrypt w środowisku Python.
3. Wyniki (metryki, podsumowanie kerneli i przykładowe predykcje) zostaną
   wyświetlone w konsoli.
4. Cztery wykresy wizualizacyjne pojawią się w oddzielnych oknach.
==========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import warnings

warnings.filterwarnings("ignore")

URL_SEEDS = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
COLUMNS_SEEDS = ["Area", "Perimeter", "Compactness", "Length_of_kernel", "Width_of_kernel", "Asymmetry_coefficient",
                 "Length_of_kernel_groove", "Class"]
RANDOM_STATE = 42


def wheat_seeds_classification():
    """
    Przeprowadza klasyfikację zbioru Wheat Seeds przy użyciu drzewa decyzyjnego i SVM.
    Wykonuje:
    - Wczytanie danych
    - Podział na zbiór treningowy i testowy
    - Standaryzację cech (dla SVM)
    - Trening modeli i obliczenie dokładności
    - Wizualizację danych i drzewa decyzyjnego
    - Przykładową predykcję dla nowych danych
    """
    print("\n--- ZBIÓR 1: WHEAT SEEDS ---")

    # Wczytanie danych
    df_seeds = pd.read_csv(URL_SEEDS, sep=r'\s+', header=None, names=COLUMNS_SEEDS)
    X_s = df_seeds.drop('Class', axis=1).values
    y_s = df_seeds['Class'].values

    # Podział na zbiór treningowy i testowy
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_s, y_s, test_size=0.3, random_state=RANDOM_STATE, stratify=y_s
    )

    # Standaryzacja cech dla SVM
    scaler_s = StandardScaler()
    X_train_s_scaled = scaler_s.fit_transform(X_train_s)
    X_test_s_scaled = scaler_s.transform(X_test_s)

    # Trening drzewa decyzyjnego
    dt_s = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_s.fit(X_train_s, y_train_s)
    dt_acc_s = accuracy_score(y_test_s, dt_s.predict(X_test_s))

    # Trening SVM
    svm_s = SVC(random_state=RANDOM_STATE)
    svm_s.fit(X_train_s_scaled, y_train_s)
    svm_acc_s = accuracy_score(y_test_s, svm_s.predict(X_test_s_scaled))

    # Wyświetlenie metryk
    print("METRYKI:")
    print(f"Dokładność DT: {dt_acc_s:.4f}")
    print(f"Dokładność SVM: {svm_acc_s:.4f}")

    # Wizualizacja danych
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_s[:, 3], X_s[:, 4], c=y_s, cmap='viridis', alpha=0.7)
    plt.xlabel(COLUMNS_SEEDS[3])
    plt.ylabel(COLUMNS_SEEDS[4])
    plt.title('1/4 Wizualizacja Danych Wheat Seeds (Długość vs Szerokość)')
    plt.colorbar(scatter, ticks=[1, 2, 3], label='Klasa Nasiona')
    plt.show()

    # Wizualizacja drzewa decyzyjnego
    plt.figure(figsize=(18, 12))
    plot_tree(
        dt_s, feature_names=COLUMNS_SEEDS[:-1],
        class_names=[str(c) for c in sorted(np.unique(y_s))],
        filled=True, rounded=True, fontsize=9
    )
    plt.title('2/4 Wizualizacja Drzewa Decyzyjnego dla Wheat Seeds')
    plt.show()

    # Przykładowa predykcja
    new_seed_data = np.array([[15.26, 14.84, 0.8710, 5.763, 3.312, 2.221, 5.220]])
    new_seed_data_scaled = scaler_s.transform(new_seed_data)
    print("PRZYKŁADOWE PREDYKCJE:")
    print(f"DT przewiduje: {dt_s.predict(new_seed_data)}")
    print(f"SVM przewiduje: {svm_s.predict(new_seed_data_scaled)}")
    print("-" * 50)


def breast_cancer_classification():
    """
    Przeprowadza klasyfikację zbioru Breast Cancer przy użyciu drzewa decyzyjnego i SVM.
    Wykonuje:
    - Wczytanie danych
    - Podział na zbiór treningowy i testowy
    - Standaryzację cech (dla SVM)
    - Trening modeli i obliczenie dokładności
    - Wizualizację danych i drzewa decyzyjnego
    - Przykładową predykcję dla nowych danych
    """
    print("\n--- ZBIÓR 2: BREAST CANCER ---")
    print("Link do opisu: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)")

    # Wczytanie danych
    data_c = load_breast_cancer()
    X_c = data_c.data
    y_c = data_c.target
    feature_names_c = data_c.feature_names.tolist()

    # Podział danych
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_c, y_c, test_size=0.3, random_state=RANDOM_STATE, stratify=y_c
    )

    # Standaryzacja
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)

    # Trening drzewa decyzyjnego
    dt_c = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_c.fit(X_train_c, y_train_c)
    dt_acc_c = accuracy_score(y_test_c, dt_c.predict(X_test_c))

    # Trening SVM
    svm_c = SVC(random_state=RANDOM_STATE)
    svm_c.fit(X_train_c_scaled, y_train_c)
    svm_acc_c = accuracy_score(y_test_c, svm_c.predict(X_test_c_scaled))

    # Wyświetlenie metryk
    print("METRYKI:")
    print(f"Dokładność DT: {dt_acc_c:.4f}")
    print(f"Dokładność SVM: {svm_acc_c:.4f}")

    # Wizualizacja danych
    plt.figure(figsize=(8, 6))
    scatter_c = plt.scatter(X_c[:, 0], X_c[:, 1], c=y_c, cmap='coolwarm', alpha=0.7)
    plt.xlabel(feature_names_c[0])
    plt.ylabel(feature_names_c[1])
    plt.title('3/4 Wizualizacja Danych Breast Cancer (Radius vs Texture)')
    plt.colorbar(scatter_c, ticks=[0, 1], label='Klasa (0: Złośliwy, 1: Łagodny)')
    plt.show()

    # Wizualizacja drzewa decyzyjnego
    plt.figure(figsize=(18, 12))
    plot_tree(
        dt_c, feature_names=feature_names_c,
        class_names=[data_c.target_names[i] for i in sorted(np.unique(y_c))],
        filled=True, rounded=True, fontsize=9, max_depth=3
    )
    plt.title('4/4 Wizualizacja Drzewa Decyzyjnego dla Breast Cancer (Max Głębokość 3)')
    plt.show()

    # Przykładowa predykcja
    new_cancer_data = X_c[:2]
    new_cancer_data_scaled = scaler_c.transform(new_cancer_data)
    print("PRZYKŁADOWE PREDYKCJE:")
    print(f"DT przewiduje: {dt_c.predict(new_cancer_data)}")
    print(f"SVM przewiduje: {svm_c.predict(new_cancer_data_scaled)}")
    print("-" * 50)


def svm_kernel_experiments():
    """
    Eksperymentuje z różnymi funkcjami jądra SVM (linear, poly, rbf) dla zbioru Breast Cancer.
    - Trenuje SVM z różnymi parametrami dla każdego kernela
    - Oblicza dokładność dla zbioru testowego
    - Wyświetla porównanie wyników
    """
    print("\n--- EKSPERYMENTY Z JĄDRAMI SVM ---")
    kernels = ['linear', 'poly', 'rbf']
    results = {}

    data_c = load_breast_cancer()
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        data_c.data, data_c.target, test_size=0.3, random_state=RANDOM_STATE, stratify=data_c.target
    )
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)

    for kernel in kernels:
        if kernel == 'poly':
            svc = SVC(kernel=kernel, degree=3, random_state=RANDOM_STATE)
        elif kernel == 'linear':
            svc = SVC(kernel=kernel, C=0.1, random_state=RANDOM_STATE)
        else:
            svc = SVC(kernel=kernel, gamma=0.01, random_state=RANDOM_STATE)

        svc.fit(X_train_c_scaled, y_train_c)
        accuracy = accuracy_score(y_test_c, svc.predict(X_test_c_scaled))
        results[kernel] = accuracy

    print("\nPORÓWNANIE DOKŁADNOŚCI DLA RÓŻNYCH JĄDER SVM:")
    for kernel, acc in results.items():
        print(f"- Jądro {kernel} (zmienione parametry): {acc:.4f}")

    print("\nPODSUMOWANIE (DO REPOZYTORIUM):")
    print(
        "Różne jądra SVM znacząco wpływają na wyniki, ponieważ zmieniają sposób,\n"
        " w jaki dane są mapowane do wyższego wymiaru w celu ich separacji.\n"
        " Jądro linear tworzy prostą granicę decyzyjną.\n"
        " Jądra poly i rbf są przeznaczone do tworzenia bardziej złożonych,\n"
        " nieliniowych granic. Jądro rbf (Radial Basis Function) często daje najlepsze wyniki dla nieliniowych,\n"
        " złożonych zbiorów danych, takich jak ten.\n"
        " Eksperymentowanie z parametrami (np. C, degree, gamma) jest kluczowe,\n"
        " aby zoptymalizować separację i zapobiec przetrenowaniu.\n"
    )


if __name__ == "__main__":
    wheat_seeds_classification()
    breast_cancer_classification()
    svm_kernel_experiments()
