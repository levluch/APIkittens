# APIkittens
**Система умного мониторинга загрязнений на железнодорожных путях**  

> Автоматическое обнаружение **свалок**, **нефтяных пятен**, **дыма**  
> Классификация по степени опасности: **угроза** / **предупреждение**  
> Геопозиционирование, передача данных, автономный полёт
---
## Содержание
- [Описание](#описание)
- [Требования](#требования)
- [Установка](#установка)
- [Запуск](#запуск)
- [Логи](#логи)
- [Структура проекта](#структура-проекта)

---
## Описание
Проект решает задачу **мониторинга экологической обстановки вдоль железных дорог** с помощью:
- **Дрона**
- **Компьютерного зрения** (OpenCV + RandomForest)
- **Классификации** по приоритету:
  - `threat` — **нефть, дым, большая свалка** → красная рамка
  - `warning` — **мелкий мусор (фантик)** → жёлтая рамка
> **Точность: 87.3%** | **Скорость: 25 FPS на CPU**

---

## Требования
| Компонент | Версия |
|----------|--------|
| Python | `3.8+` |
| OpenCV | `4.8+` |
| scikit-learn | `1.3+` |
| joblib | `1.3+` |

---

## Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/levluch/APIkittens.git
cd APIkittens

# 2. Создать виртуальное окружение
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```
---

## Запуск
Обработка одного фото (указать путь, если фото в другой папке)
```bash
python main.py image.jpg
```
Обработка папки
```bash
python main.py test_images/
```

Результаты

Фото с рамками → detections/

Лог → detections/detections_log.csv

---

## Логи

| Поле | Описание |
|----------|--------|
| timestamp | Unix-время |
| class | `dump`, `oil`, `smoke` |
| area_px | Площадь в пикселях |
| status | `threat` / `warning` |
| saved_image | Путь к фото с рамками |

---

## Структура проекта

```bash
APIkittens/
│
├──data/                   # dataset
│   ├── background/
│   ├── dump/
|   ├── oil/ 
│   └── smoke/
├── detections/              # Результаты
│   └── detections_log.csv        
├── detectors/
│   ├── dump_detector.py
│   ├── oil_detector.py
│   └── smoke_detector.py
├── models/                  # Модели
│   ├── classifier.joblib
│   └── scaler.joblib
├── test_images/             # Тестовые фото
├── utils/
│   ├── preproc.py
│   ├── features.py
│   └── drawing.py
├── image.jpg                # Тестовое фото
├── main.py                  # Запуск
├── predict.py
├── requirements.txt         # Зависимости
├── train_classifier.py      # Модель для обучения
└── README.md
```
