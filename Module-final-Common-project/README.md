# Описание проекта и задача
Из данных пользователей телеком компании необходимо построить модель машинного обучения, которая будет предсказывать перестанет пользоваться клиент услугами или нет (для дальнейшего предложения скидок/промокодов). *главная метрика - это AUC-ROC, также учитывать Accuracy*


# Данные:
Таблицы со следующими данными:
- `contract.csv` — информация о договоре;
- `personal.csv` — персональные данные клиента;
- `internet.csv` — информация об интернет-услугах;
- `phone.csv` — информация об услугах телефонии.

# Используемые библиотеки
- pandas
- numpy
- matplotlib
- seaborn
- phik
- sklearn
  - model_selection
  - preprocessing
  - KMeans (clusters)
  - metrics
  - RandomForestClassifier
  - LogisticRegression
  - SGDClassifier
  - DummyClassifier
- catboost
- lightgbm

# Результат
Обучена модель LGBMClassifier для нашего случая несбалансированной бинарной классификации. Итоговый  результат (на тестовой выборке): 0.9047 AUC-ROC и 0.8358 Accuracy
