import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import mynn as nn
from copy import deepcopy

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

train_images_path = './dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = './dataset/MNIST/train-labels-idx1-ubyte.gz'
test_images_path = './dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = './dataset/MNIST/t10k-labels-idx1-ubyte.gz'

train_imgs = load_images(train_images_path)
train_labs = load_labels(train_labels_path)
test_imgs = load_images(test_images_path)
test_labs = load_labels(test_labels_path)

# quick split: use part of data to save time
valid_imgs = test_imgs[:1000]
valid_labs = test_labs[:1000]
train_imgs = train_imgs[:5000]
train_labs = train_labs[:5000]

def accuracy(logits, labels):
    pred = np.argmax(logits, axis=1)
    return np.mean(pred == labels)

def train_one_lr(lr, num_epochs=5, batch_size=64):
    model = nn.models.Model_MLP(
        size_list=[784, 256, 128, 10],
        act_func='ReLU',
        lambda_list=[1e-4, 1e-4, 1e-4]
    )
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    optimizer = nn.optimizer.SGD(init_lr=lr, model=model)

    train_losses = []
    valid_scores = []

    num_samples = train_imgs.shape[0]
    for epoch in range(num_epochs):
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            idx = indices[start:start+batch_size]
            X = train_imgs[idx]
            y = train_labs[idx]

            logits = model(X)
            loss = loss_fn(logits, y)
            loss_fn.backward()
            optimizer.step()

        train_logits = model(train_imgs[:1000])
        train_loss = loss_fn(train_logits, train_labs[:1000])
        valid_logits = model(valid_imgs)
        valid_acc = accuracy(valid_logits, valid_labs)

        train_losses.append(train_loss)
        valid_scores.append(valid_acc)

        print(f"lr={lr}, epoch={epoch+1}, train_loss={train_loss:.4f}, valid_acc={valid_acc:.4f}")

    return train_losses, valid_scores

learning_rates = [0.01, 0.005, 0.001]
results = {}

for lr in learning_rates:
    print("=" * 50)
    print(f"Training MLP with learning rate = {lr}")
    losses, scores = train_one_lr(lr)
    results[lr] = {
        "losses": losses,
        "scores": scores,
    }

plt.figure(figsize=(8, 5))
for lr in learning_rates:
    plt.plot(results[lr]["scores"], marker='o', label=f"lr={lr}")
plt.xlabel("epoch")
plt.ylabel("validation accuracy")
plt.title("Learning Rate Comparison on MLP")
plt.legend()
plt.tight_layout()
plt.savefig("./figs/lr_comparison.png", dpi=200)
print("Saved learning-rate comparison figure to ./figs/lr_comparison.png")

print("\nFinal validation accuracy:")
for lr in learning_rates:
    print(f"lr={lr}: {results[lr]['scores'][-1]:.4f}")
