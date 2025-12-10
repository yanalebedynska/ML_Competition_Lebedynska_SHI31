# ML_Competiotion_Lebedynska_SHI31
Author: Yana Lebedynska (SHI-31)

**Опис проєкту**

Проєкт створений у рамках змагання NULP Military Experience Classification.

Завдання - побудувати модель бінарної класифікації, яка визначає, чи містить повідомлення згадку про:
наявність військового досвіду (так / ні).

Модель побудована на основі попередньо натренованої трансформерної архітектури xlm-roberta-base, яка добре працює з українською та російською мовами.


**Структура репозиторію:**

```
README.md                - опис проєкту
project/
│-- train.py                 - основний скрипт навчання моделі
│-- evaluation.py            - швидка перевірка пайплайну (без повного тренування)
│-- how-to-evaluate.ipynb    - докладна інструкція для викладача
│-- requirements.txt         - список залежностей
│-- Звіт до роботи
│-- data/
│      └── (порожньо - користувач додає train.csv і to_answer.csv вручну)
```

**Використані бібліотеки та моделі**
_Pretrained модель_

xlm-roberta-base 
https://huggingface.co/xlm-roberta-base

Ліцензія: Apache 2.0


_Основні бібліотеки_

PyTorch
Transformers
scikit-learn
pandas
numpy
tqdm

Усі залежності у **_requirements.txt_**.
