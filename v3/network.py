import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.preprocessing import LabelEncoder

def use_nn(X_train, y_train, X_test, y_test):

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out
    
    # Define model, loss function, and optimizer
    input_size = X_train.shape[1]
    hidden_size = 50  # You can adjust this
    output_size = len(np.unique(y_train))  # Number of classes
    model = SimpleNN(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10  # You can adjust this
    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in trainloader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in testloader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    print(f'Accuracy of the model on the test set: {100 * correct / total}%')

    return 0

def main():

    #Fake data to test
    X_train = np.random.rand(800, 20)  # 800 samples, 20 features each for training
    y_train = np.random.choice(['cat', 'dog', 'mouse'], 800)  # 800 labels for training
    X_test = np.random.rand(200, 20)  # 200 samples, 20 features each for testing
    y_test = np.random.choice(['cat', 'dog', 'mouse'], 200)  # 200 labels for testing
    
    do_nn(X_train, y_train, X_test, y_test)


    return 0

if __name__ == '__main__':
    main()
