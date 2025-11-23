## 1. Wprowadzenie

Celem zadania była klasyfikacja win na trzy kategorie na podstawie 13 parametrów chemicznych. Dane pochodziły ze zbioru **Wine** (UCI Machine Learning Repository). Przed trenowaniem dane zostały potasowane, znormalizowane (**StandardScaler**), a klasy zakodowane metodą **one-hot**.

## 2. Przygotowanie danych

- Wczytano CSV z katalogu `public_resources/`.
- Potasowano dane (`frac=1, random_state=42`).
- Wydzielono cechy i etykiety.
- Klasy zakodowano metodą one-hot (`pd.get_dummies` → 3 kolumny).
- Podzielono dane na zbiór treningowy i testowy (80/20) z zachowaniem proporcji klas (`stratify`).
- Skalowanie cech: **StandardScaler** dopasowany wyłącznie na danych treningowych.

# Skalowanie cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Modele

### Model FIRST (simple)
**Architektura:**
Input → Dense(32, activation='relu') → Dense(3, activation='softmax')
- Zalety: mała liczba parametrów, szybki trening, niskie ryzyko overfittingu.
- Funkcja straty: `categorical_crossentropy`
- Optymalizator: `Adam(lr=0.001)`

### Model SECOND (deep)
**Architektura:**
Input → Dense(256, relu) → Dense(128, relu) → Dense(64, relu) → Dense(32, relu) → Dense(16, relu) → Dropout(0.2) → Dense(3, softmax)
- Zalety: duża moc modelowania
- Wady: podatny na przeuczenie przy małych zbiorach danych
- Inicjalizacja: `HeNormal` (dobra dla ReLU)


---

### 4. Trening i wyniki
- Oba modele trenowano z określonymi hiperparametrami (`epochs`, `batch_size`, `learning_rate`).
- Zapisano wykresy uczenia (accuracy i loss) oraz najlepsze modele w katalogu `saved/`.

**Porównanie wyników:**

| Model       | Wynik na treningu | Wynik na teście | Uwagi |
|------------|-----------------|----------------|-------|
| FIRST      | stabilne, wysokie accuracy | stabilne, wysokie accuracy | Większa stabilność przy niewielkiej liczbie próbek |
| SECOND     | bardzo niska strata | wyższa strata niż trening | Przejaw overfittingu |

> **Wniosek:** Model FIRST sprawdził się lepiej jako finalny model.

## 5. Predykcja użytkownika

- Na podstawie najlepszego modelu umożliwiono predykcję jednej próbki przez **CLI** (`argparse`).
- Użytkownik podaje 13 parametrów wina, które są skalowane tym samym **StandardScalerem**.
- Model zwraca przewidywaną klasę oraz prawdopodobieństwa.

## Wnioski

- Model prosty okazał się bardziej odporny na przeuczenie i dał lepsze wyniki generalizacji.
- Model głęboki wymagałby większego zbioru danych lub mocniejszej regularizacji.
- Znormalizowanie danych i utrzymanie spójności nazw kolumn było kluczowe do poprawnego działania predykcji.