# pprint = pretty print, see: https://docs.python.org/3/library/pprint.html
from pathlib import Path
from pprint import pprint
import torch
from torch import nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

# Generate some data
weight = 0.7
bias = 0.3

start = 0
stop = 1
step = 0.02

X = torch.arange(start, stop, step).unsqueeze(dim=1)
y = weight * X + bias

# print(X[:10], y[:10])

# Split it
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Visualize


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


# plot_predictions()

# Model

class LinearRegressionV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(
            in_features=1,
            out_features=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(42)
model_1 = LinearRegressionV2()
print(model_1, model_1.state_dict())

model_1.to(device=device)
# Show device params
# print(next(model_1.parameters()).device)

# Training

# Loss & optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(),  # optimize newly created model's parameters
                            lr=0.01)
# Training loop
torch.manual_seed(42)
epochs = 10 ** 3

for i in [X_train, X_test, y_train, y_test]:
    i = i.to(device)

for epoch in range(epochs):
    # TRAIN
    model_1.train()
    # Forward
    y_pred = model_1(X_train)
    # Loss
    loss = loss_fn(y_pred, y_train)
    # Zero grad optimizer
    optimizer.zero_grad()
    # Loss back
    loss.backward()
    # Step
    optimizer.step()

    # TEST
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)

        # Loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")


# plot_predictions(predictions=test_pred)

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
pprint(model_1.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# PREDICTIONS

print("\n6.4 Predictions\n")

model_1.eval()

with torch.inference_mode():
    y_pred = model_1(X_test)
print(y_pred)

# plot_predictions(predictions=y_pred.cpu())

# SAVING AND LOADING THE MODEL

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),  # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)
