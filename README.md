# Cirrhosis Patient Survival Prediction & MLOps

**Авторы:** 
- [Доржиева Пурбо-Суруна](https://github.com/poeeeri) (группа 972401)
- [Норец Елена](https://github.com/monalenka) (группа 972402)

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