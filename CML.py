import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
        label = 0 if 'test/fake' in path else 1
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
nonfake_path = "test/nonfake"
fake_path = "test/fake"

# Загрузка данных
X_test, y_test, images_test = load_data([nonfake_path, fake_path])

# Предсказания модели
y_pred = loaded_model.predict(X_test)

# Подсчет метрик
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

# Функция для создания и сохранения сетки изображений
def create_grid_image(images, y_true, y_pred, correct=True, base_dir="results", filename="grid.png"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    indices = np.where(y_true == y_pred if correct else y_true != y_pred)[0]
    n_images = len(indices)
    
    if n_images > 0:
        cols = 4
        rows = (n_images // cols) + 1 if n_images % cols != 0 else n_images // cols
        plt.figure(figsize=(15, rows * 5))
        
        for i, idx in enumerate(indices):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
            plt.title(f"True: {'Healthy' if y_true[idx] == 0 else 'Diseased'}, Pred: {'Healthy' if y_pred[idx] == 0 else 'Diseased'}")
            plt.axis('off')
        
        plt.tight_layout()
        grid_path = os.path.join(base_dir, filename)
        plt.savefig(grid_path)
        plt.close()
        
        return grid_path
    else:
        print("No examples found for the given condition.")
        return None

correct_grid_path = create_grid_image(images_test, y_test, y_pred, correct=True, filename="correct_grid.png")
incorrect_grid_path = create_grid_image(images_test, y_test, y_pred, correct=False, filename="incorrect_grid.png")

def update_readme(accuracy, f1, correct_grid_path, incorrect_grid_path, readme_path="README.md"):
    with open(readme_path, 'w') as f:
        f.write(f"# Model Evaluation Report\n\n")
        f.write(f"## Metrics\n\n")
        f.write(f"- **Accuracy**: {accuracy}\n")
        f.write(f"- **F1 Score**: {f1}\n\n")
        
        if correct_grid_path:
            f.write(f"## Correctly Classified Examples\n\n")
            f.write(f"![Correctly Classified Examples]({correct_grid_path})\n\n")
        
        if incorrect_grid_path:
            f.write(f"## Incorrectly Classified Examples\n\n")
            f.write(f"![Incorrectly Classified Examples]({incorrect_grid_path})\n\n")

update_readme(accuracy, f1, correct_grid_path, incorrect_grid_path)