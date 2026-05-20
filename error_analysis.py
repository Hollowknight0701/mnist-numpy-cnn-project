import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import mynn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols).astype(np.float32) / 255.0
    return data

def load_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

test_images_path = './dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = './dataset/MNIST/t10k-labels-idx1-ubyte.gz'

X_test = load_images(test_images_path)
y_test = load_labels(test_labels_path)

model = nn.models.Model_CNN()
model.load_model('./best_models/best_model.pickle')

# predict
logits = model(X_test)
y_pred = np.argmax(logits, axis=1)

acc = np.mean(y_pred == y_test)
print(f"Test accuracy: {acc:.4f}")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(values_format='d', cmap=None)
plt.title(f'CNN Confusion Matrix, Test Acc = {acc:.4f}')
plt.tight_layout()
plt.savefig('./figs/cnn_confusion_matrix.png', dpi=200)
print("Saved confusion matrix to ./figs/cnn_confusion_matrix.png")

# misclassified examples
wrong_idx = np.where(y_pred != y_test)[0]
print(f"Number of misclassified examples: {len(wrong_idx)}")

num_show = min(16, len(wrong_idx))
plt.figure(figsize=(8, 8))
for i in range(num_show):
    idx = wrong_idx[i]
    plt.subplot(4, 4, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
    plt.axis('off')

plt.suptitle('Misclassified Examples: T=True, P=Predicted')
plt.tight_layout()
plt.savefig('./figs/cnn_misclassified_examples.png', dpi=200)
print("Saved misclassified examples to ./figs/cnn_misclassified_examples.png")
