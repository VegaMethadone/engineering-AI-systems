import os
import cv2
import numpy as np
from joblib import load

# Загрузка модели
loaded_model = load("svm_model.joblib")

# Функция извлечения признаков из изображения
def extract_features(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Загрузка и подготовка тестовых данных
def load_data(paths):
    X, y, images = [], [], []
    for path in paths:
        label = 0 if 'healthy' in path else 1
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                img = cv2.resize(img, (500, 500))
                features = extract_features(img)
                X.append(features)
                y.append(label)
                images.append(img)
    return np.array(X), np.array(y), np.array(images)

# Пути к папкам с тестовыми данными
healthy_path = "test/healthy"
rust_path = "test/rust"

# Загрузка тестовых данных
X_test, y_test, test_images = load_data([healthy_path, rust_path])

# Предсказание на тестовых данных
y_pred = loaded_model.predict(X_test)

# Вывод сравнения между фактическими метками и предсказанными метками
passed, failed = 0, 0
for i, (original, predicted) in enumerate(zip(y_test, y_pred)):
    print(f"Sample {i + 1}: Original: {original}, Predicted: {predicted}")
    if original == predicted:
        passed += 1
    else:
        failed += 1

print(f"Passed: {passed}, Failed: {failed}")
print(f"Accuracy: {passed / (passed+failed)}")