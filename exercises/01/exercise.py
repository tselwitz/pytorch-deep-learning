from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt

# 1. Create a straight line dataset using the linear regression formula (`weight * X + bias`).
#   * Set `weight=0.3` and `bias=0.9` there should be at least 100 datapoints total.
#   * Split the data into 80% training, 20% testing.
#   * Plot the training and testing data so it becomes visual.

weight = 0.3
bias = 0.9

start, stop, step = 0, 10, 0.1
X = torch.arange(start, stop, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


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

# Uncomment to plot
# plot_predictions()

# 2. Build a PyTorch model by subclassing `nn.Module`.
#   * Inside should be a randomly initialized `nn.Parameter()` with `requires_grad=True`, one for `weights` and one for `bias`.
#   * Implement the `forward()` method to compute the linear regression function you used to create the dataset in 1.
#   * Once you've constructed the model, make an instance of it and check its `state_dict()`.
#   * **Note:** If you'd like to use `nn.Linear()` instead of `nn.Parameter()` you can.


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        return self.linear_layer.weight * x + self.linear_layer.bias


m = model()
print(m.state_dict())

# 3. Create a loss function and optimizer using `nn.L1Loss()` and `torch.optim.SGD(params, lr)` respectively.
#   * Set the learning rate of the optimizer to be 0.01 and the parameters to optimize should be the model parameters from the model you created in 2.
#   * Write a training loop to perform the appropriate training steps for 300 epochs.
#   * The training loop should test the model on the test dataset every 20 epochs.

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=m.parameters(), lr=0.01)
epochs = 300

for e in range(epochs):
    m.train()
    y_pred = m(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 20 == 0:
        m.eval()
        with torch.inference_mode():
            y_pred = m(X_test)
            test_loss = loss_fn(y_pred, y_test)
            print(
                f"Epoch: {e} | Train loss: {loss} | Test loss: {test_loss}")

# 4. Make predictions with the trained model on the test data.
#   * Visualize these predictions against the original training and testing data (**note:** you may need to make sure the predictions are *not* on the GPU if you want to use non-CUDA-enabled libraries such as matplotlib to plot).

m.eval()
with torch.inference_mode():
    preds = m(X_test)

# plot_predictions(predictions=preds)

# 5. Save your trained model's `state_dict()` to file.
#   * Create a new instance of your model class you made in 2. and load in the `state_dict()` you just saved to it.
#   * Perform predictions on your test data with the loaded model and confirm they match the original model predictions from 4.

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow_exercise_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=m.state_dict(),  # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)

m2 = model()
m2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

pred2 = m2(X_test)

print(preds == pred2)
