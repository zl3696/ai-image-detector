import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += ((outputs > 0.5).float() == labels).sum().item()
            train_total += labels.size(0)
        avg_train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total

        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        average_loss = val_loss / total
        accuracy = 100 * correct / total

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(average_loss)
        history['val_acc'].append(accuracy)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_acc:.2f}% | Val Loss: {average_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

    print(f"Best validation accuracy: {max(history['val_acc']):.1f}%")
    return history
