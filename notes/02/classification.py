from helper_functions import plot_predictions, plot_decision_boundary
from pathlib import Path
import requests
from torch import nn
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles



# Binary classification

# Make 1000 samples

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# print(f"First 5 X features:\n{X[:5]}")
# print(f"\nFirst 5 y labels:\n{y[:5]}")

# Examine the data
# Notice, y takes on only 0 and 1 and we have an even split of both
# circles = pd.DataFrame({
#     "X1": X[:, 0],
#     "X2": X[:, 1],
#     "label": y
# })

# print(circles.head(10))
# print(circles.label.value_counts())

# Plotting usually helps understand the data better if the data is simple enough
# plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# Keep a good understanding of the shape of your data to avoid shape errors
# print(X.shape, y.shape)

# View the first example of features and labels
# X_sample = X[0]
# y_sample = y[0]
# print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
# print(
#     f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

# x is a vector, not a scalar, so let's convert them into tensors

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# print(X[:5], y[:5])

# Now let's create our training sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Show the sizes of the sets
# print(" ".join([repr(len(i)) for i in [X_train, X_test, y_train, y_test]]))

# Now let's build a model for the classifier

# Device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

# The model


class CircleModelV0(nn.Module):
    def __init__(self, hidden_features=5):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=hidden_features)
        self.layer_2 = nn.Linear(in_features=hidden_features, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))
    
class CircleModelV1(nn.Module):
    def __init__(self, hidden_features=10):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=hidden_features)
        self.layer_2 = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.layer_3 = nn.Linear(in_features=hidden_features, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))


model_0 = CircleModelV0().to(device)
print(model_0)

# You can also recreate CircleModelV0 with nn.Sequential
# model_0 = nn.Sequential(
#     nn.Linear(2, 5),
#     nn.Linear(5, 1)
# ).to(device)

print(model_0)

untrained_preds = model_0(X_test.to(device))

print(
    f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")

# Define loss and optimizer

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# evaluation metric


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# View the frist 5 outputs of the forward pass on the test data
y_logits = model_0(X_test.to(device))[:5]
print(y_logits)
# Use sigmoid on model logits to get numbers comparable to our truth labels
# y_pred_probs are the probabilities that the model will predict that value to be 1.
# So, if y_pred_probs == 0.9, then y has a 90% chance to be predicted to be 1
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)

y_preds = torch.round(y_pred_probs)
print(y_preds)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# Get rid of extra dimension
print(y_preds.squeeze())
print(y_test[:5])


def train(model=model_0, loss_fn=loss_fn, optimizer=optimizer, epochs=10 ** 2, test_cadence=10):
    torch.manual_seed(42)
    for e in range(epochs):
        model.train()
        # Forward
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        # Loss/Accuracy
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Test
        model.eval()
        with torch.inference_mode():
            # Forward
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            # Calculate loss/accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        if e % test_cadence == 0:
            print(
                f"Epoch: {e} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# Plot decision boundaries for training and test sets
def plot(model=model_0, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X_test, y_test)
    plt.show()

train()
# plot()

# Note that model_0 is bad.  Can adding a layer and a longer training period fix this?

model_1 = CircleModelV1().to(device)
train(model=model_1, epochs=10 ** 3)
# plot(model_1)

# Nope...
# Is this model useless? Let's see if it can plot a line

# Create some data (same as notebook 01)
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

# Create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias # linear regression formula

# Check the data
print(len(X_regression))
print(X_regression[:5], y_regression[:5])

# Create train and test splits
train_split = int(0.8 * len(X_regression)) # 80% of data used for training set
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# Check the lengths of each split
print(len(X_train_regression), 
    len(y_train_regression), 
    len(X_test_regression), 
    len(y_test_regression))

plot_predictions(train_data=X_train_regression,
    train_labels=y_train_regression,
    test_data=X_test_regression,
    test_labels=y_test_regression
);

# plt.show()

model_2 = nn.Sequential(
    nn.Linear(1, 10),
    nn.Linear(10, 10),
    nn.Linear(10, 1)
).to(device)

model=model_2
loss_fn=nn.L1Loss()
optimizer=torch.optim.SGD(model_2.parameters(), lr=0.1)
epochs=1000

for e in range(epochs):
    model_2.train()
    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred, y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred, y_test_regression)
        
    
    if e % 100 == 0: 
        print(f"Epoch: {e} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")

# Turn on evaluation mode
model_2.eval()

# Make predictions (inference)
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

# Plot data and predictions with data on the CPU (matplotlib can't handle data on the GPU)
# (try removing .cpu() from one of the below and see what happens)
plot_predictions(train_data=X_train_regression.cpu(),
                 train_labels=y_train_regression.cpu(),
                 test_data=X_test_regression.cpu(),
                 test_labels=y_test_regression.cpu(),
                 predictions=y_preds.cpu());
plt.show()