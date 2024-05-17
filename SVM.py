import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# Пути к папкам с изображениями
healthy_paths = ["data/healthy", "data/healthy90", "data/healthy180", "data/healthy270"]
rust_paths = ["data/Rust", "data/Rust90", "data/Rust180", "data/Rust270"]


def extract_features(image):
    # В этом примере используем гистограммы цветов как признаки
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def load_data(paths):
    X = []
    y = []
    for i, path in enumerate(paths):
        label = 0 if 'healthy' in path else 1
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                img = cv2.resize(img, (500, 500))
                features = extract_features(img)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)


# Загрузка данных
X_healthy, y_healthy = load_data(healthy_paths)
X_rust, y_rust = load_data(rust_paths)

# Объединение данных и меток
X = np.concatenate((X_healthy, X_rust))
y = np.concatenate((y_healthy, y_rust))

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели SVM
clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
clf.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = clf.predict(X_test)

# Запись результатов предсказания в файл
with open("predict.txt", "w") as file:
    for original, predicted in zip(y_test, y_pred):
        file.write(f"origin: {original} predicted: {predicted}\n")

print("The prediction results are saved in a file: predict.txt")

# Оценка модели
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



from joblib import dump

# Указываем путь и имя файла для сохранения модели
model_filename = "svm_model.joblib"

# Сохраняем модель в файл
dump(clf, model_filename)

print(f"Model is saved as: {model_filename}")














