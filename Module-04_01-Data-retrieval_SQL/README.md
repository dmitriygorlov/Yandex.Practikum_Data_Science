# Описание проекта и задача
По данным авиаперелетов, сгрупированных по городам и моделям самолетов, нужно провести анализ на наиболее распространенные. *Нужно для начала сформировать запросы к базе данных на SQL*


# Данные:
**query_1.csv** — результат первого запроса. В нём содержится такая информация:
- *model* — **модель самолета;
- *flights_amount* — количество рейсов для каждой модели самолетов *model* в сентябре 2018 года.

**query_3.csv** — результат третьего запроса. В нём содержится такая информация:
- *city* — город;
- *average_flights* — среднее количество рейсов, прибывающих в город (*city*) за день в сентябре 2018 года.

# Используемые библиотеки
- SQL
- numpy
- pandas
- matplotlib

# Результат
Главным городом по количеству прибывающих рейсов с отрывом в 4 раза от ближайшего соперника является Москва, при этом самой часто летающей моделью самолета по стране является Cessna
