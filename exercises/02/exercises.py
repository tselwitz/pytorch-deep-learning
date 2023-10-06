import torch
from torch import nn
from helper_functions import plot_decision_boundary
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Make a binary classification dataset with Scikit-Learn's [`make_moons()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) function.
#     * For consistency, the dataset should have 1000 samples and a `random_state=42`.
#     * Turn the data into PyTorch tensors. Split the data into training and test sets using `train_test_split` with 80% training and 20% testing.

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

SAMPLES = 1000
RANDOM_STATE=42

X, y = make_moons(n_samples=SAMPLES, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.2,
  random_state=RANDOM_STATE
)

X_train = torch.from_numpy(X_train).type(torch.float)
X_test = torch.from_numpy(X_test).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.float)
y_test = torch.from_numpy(y_test).type(torch.float)

# 2. Build a model by subclassing `nn.Module` that incorporates non-linear activation functions and is capable of fitting the data you created in 1.
#     * Feel free to use any combination of PyTorch layers (linear and non-linear) you want.
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 7))
# plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

class MoonClassifier(nn.Module):
  def __init__(self, inp=2, hf=10, outp=1):
    super().__init__()
    self.layer_stack = nn.Sequential(
      nn.Linear(inp, hf),
      nn.ReLU(),
      nn.Linear(hf, outp)
    )
  
  def forward(self, x):
    return self.layer_stack(x)

model = MoonClassifier().to(device)

# 3. Setup a binary classification compatible loss function and optimizer to use when training the model.

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. Create a training and testing loop to fit the model you created in 2 to the data you created in 1.
#     * To measure model accuray, you can create your own accuracy function or use the accuracy function in [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/).
#     * Train the model for long enough for it to reach over 96% accuracy.
#     * The training loop should output progress every 10 epochs of the model's training and test set loss and accuracy.
from helper_functions import accuracy_fn

torch.manual_seed(RANDOM_STATE)

epochs = 2300

for e in range(epochs):
  # Training
  model.train()
  y_logits = model(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits))
  
  loss = loss_fn(y_logits, y_train)
  acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  # Testing
  model.eval()
  with torch.inference_mode():
    test_logits = model(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    
    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuracy_fn(y_test, test_pred)

  if e % 10 == 0:
    print(f"Epoch: {e} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
  

# 5. Make predictions with your trained model and plot them using the `plot_decision_boundary()` function created in this notebook.
# y_logits = model(X_test).squeeze()
# y_pred = torch.round(torch.sigmoid(y_logits))

# plot_decision_boundary(model, X_test, y_test)
# plt.show()

# 6. Replicate the Tanh (hyperbolic tangent) activation function in pure PyTorch.
#     * Feel free to reference the [ML cheatsheet website](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#tanh) for the formula.

def tanh(x):
  return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

# 7. Create a multi-class dataset using the [spirals data creation function from CS231n](https://cs231n.github.io/neural-networks-case-study/) (see below for the code).
#     * Construct a model capable of fitting the data (you may need a combination of linear and non-linear layers).
#     * Build a loss function and optimizer capable of handling multi-class data (optional extension: use the Adam optimizer instead of SGD, you may have to experiment with different values of the learning rate to get it working).
#     * Make a training and testing loop for the multi-class data and train a model on it to reach over 95% testing accuracy (you can use any accuracy measuring function here that you like).
#     * Plot the decision boundaries on the spirals dataset from your model predictions, the `plot_decision_boundary()` function should work for this dataset too.

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data:
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.2,
  random_state=RANDOM_STATE
)

X_train = torch.from_numpy(X_train).type(torch.float)
X_test = torch.from_numpy(X_test).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.long)
y_test = torch.from_numpy(y_test).type(torch.long)

input = D
hidden_features = 10
output = K

spiralModel = nn.Sequential(
  nn.Linear(input, hidden_features),
  nn.ReLU(),
  nn.Linear(hidden_features, output)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(spiralModel.parameters(), lr=0.2)

epochs = 5000

print("EXERCISE #7:\n")
for e in range(epochs):
  spiralModel.train()
  y_logits = spiralModel(X_train).squeeze()
  y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
  
  loss = loss_fn(y_logits, y_train)
  acc = accuracy_fn(y_train, y_preds)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  spiralModel.eval()
  with torch.inference_mode():
      test_logits = spiralModel(X_test).squeeze()
      test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
      
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_test, test_preds)
      
  if e % 10 == 0:
    print(f"Epoch: {e} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.long)

plot_decision_boundary(spiralModel, X, y)
plt.show()