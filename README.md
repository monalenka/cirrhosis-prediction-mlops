# Cirrhosis Patient Survival Prediction & MLOps

**Авторы:** 
- [Доржиева Пурбо-Суруна](https://github.com/poeeeri) (группа 972401)
- [Норец Елена](https://github.com/monalenka) (группа 972402)

## Аналитическое исследование (EDA)

Перед построением моделей проведён анализ данных пациентов с циррозом печени.  
Основные выводы (подробнее – в [`notebooks/eda.ipynb`](notebooks/eda.ipynb)):

- **Данные:** 15 000 записей в train, 10 000 в test. Признаки и числовые, и категориальные
- **Целевая переменная** несбалансирована: `C` (жив) – 67%, `D` (смерть) – 31%, `CL` (трансплантация) – 2%. Учтено при выборе метрики (log loss).
- **Пропуски** в некоторых признаках до 57%. Для числовых заполнены медианой, для категориальных – отдельной категорией `"Missing"`.
- **Асимметрия:** `Bilirubin`, `Cholesterol`, `Copper`, `Alk_Phos`, `SGOT`, `Tryglicerides`, `Prothrombin` сильно скошены (skewness > 1.5). Применено логарифмическое преобразование (так, `Bilirubin_log` с 4.19 до 1.91)
- **Корреляции:** мультиколлинеарности нет. `Bilirubin` отрицательно коррелирует с `Albumin` (-0.43), их отношение добавлено как признак `Bili_Alb`
- **Боксплоты:** распределения показателей по группам `C`, `CL`, `D` в основном перекрываются. Наиболее заметные различия: повышенный `Prothrombin` и `Bilirubin` у умерших, пониженный `Albumin`. Новые признаки (`Alk_SGOT`, `Platelets_Prothrombin`, `Bili_Alb`) существенного улучшения не дали.

**Итоговый набор признаков:** лог-трансформированные (`Bilirubin_log`, `Cholesterol_log`, `Copper_log`, `Alk_Phos_log`, `SGOT_log`, `Tryglicerides_log`, `Prothrombin_log`), числовые без трансформации (`Albumin`, `Platelets`, `Age_years`, `N_Days`, `Stage`) и категориальные (`Drug`, `Ascites`, `Hepatomegaly`, `Spiders`, `Edema`).

## MLOps: Training, Deployment & Experiment Tracking
### 1. Train model и Evaluate

**Локально** установить зависимости.
```
poetry install
```
По умолчанию при запуске будет использоваться ClearML, если это не нужно, можно использовать флаг:
```
poetry run python model.py train data/train.csv data/test.csv clearml=disabled
```
Получить предсказания:
```
poetry run python model.py predict data/test.csv data/results.csv
```

### 2. Deploy

**В Docker** запустить сборку модели:
```
docker build -t cirrhosis-model .
```

Тренировка модели и получение предсказаний:
1. на Linux
```
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/model:/app/model cirrhosis-model train data/train.csv data/test.csv clearml=disabled

docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/model:/app/model cirrhosis-model predict data/test.csv data/results.csv
```
2. на Windows
```
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/model:/app/model cirrhosis-model train data/train.csv data/test.csv clearml=disabled

docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/model:/app/model cirrhosis-model predict data/test.csv data/results.csv
```

### 3. Использование ClearML
Также возможно запустить тренировку модели с использованием ClearML для отслеживания параметров.
Локально:
```
poetry run python model.py train data/train.csv data/test.csv
```
Через Docker:
1. на Linux
```
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/model:/app/model -v $(pwd)/clearml.conf:/app/clearml.conf -e CLEARML_CONFIG_FILE=/app/clearml.conf cirrhosis-model train data/train.csv data/test.csv
```
2. на Windows
```
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/model:/app/model -v ${PWD}/clearml.conf:/app/clearml.conf -e CLEARML_CONFIG_FILE=/app/clearml.conf cirrhosis-model train data/train.csv data/test.csv
```
## Использованные ресурсы
- Kaggle
- CatBoost
- XGBoost
- Optuna
- ClearML
- Hydra
