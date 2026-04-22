import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model, test_loader, device, model_name, save_dir='assets'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            all_preds.extend(predicted.squeeze(1).cpu().numpy())
            all_labels.extend(labels.squeeze(1).cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=['REAL', 'FAKE'], digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['REAL', 'FAKE'],
                yticklabels=['REAL', 'FAKE'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=150)
    plt.show()


def plot_training_curves(history, model_name, save_dir='assets'):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.title(f'{model_name} - Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train')
    plt.plot(epochs, history['val_acc'], label='Val')
    plt.xlabel('Epoch')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name.lower().replace(" ", "_")}_training_curves.png', dpi=150)
    plt.show()
