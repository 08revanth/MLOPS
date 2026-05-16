import matplotlib.pyplot as plt

# Using the numbers from your terminal logs
epochs = [1, 2, 3, 4, 5]
train_acc = [0.22, 0.15, 0.15, 0.19, 0.28]
val_acc = [0.05, 0.05, 0.05, 0.08, 0.45]

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label='Training Accuracy', marker='o', color='blue')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o', color='orange')
plt.title('Model Training Progress (EfficientNetB3)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('training_plot.png')
print("Plot saved as training_plot.png")