import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import accuracy_fn, plot_decision_boundary
from torchmetrics import Accuracy

# IN THIS MODULE, WE TRY TO CREATE A MULTI-CLASS CLASSIFICATION SOLUTION

# Make up a dataset of blobs, split into training and test, visualize
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob, y_blob = make_blobs(
  n_samples=1000,
  n_features=NUM_FEATURES,
  centers=NUM_CLASSES,
  cluster_std=1.5,
  random_state=RANDOM_SEED
)

# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
print(X_blob[:5], y_blob[:5])

# 3. Split into train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
    y_blob,
    test_size=0.2,
    random_state=RANDOM_SEED
)

# 4. Plot data
# plt.figure(figsize=(10, 7))
# plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu);
# plt.show()

# BUILD THE CLASS FOR THE MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"

class BlobClassifierV0(nn.Module):
  def __init__(self, input_features, output_features, hidden_units=8):
    super().__init__()
    self.linear_layer_stack = nn.Sequential(
      nn.Linear(in_features=input_features, out_features=hidden_units),
      nn.Linear(in_features=hidden_units, out_features=hidden_units),
      nn.Linear(in_features=hidden_units, out_features=output_features),
    )
  
  def forward(self, x):
    return self.linear_layer_stack(x)
  
model4 = BlobClassifierV0(input_features=NUM_FEATURES, output_features=NUM_CLASSES).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model4.parameters(), lr=0.1)

# Let's see if our model class is looking good so far.
print(model4(X_blob_train.to(device))[:5])
print(model4(X_blob_train.to(device))[0].shape, NUM_CLASSES)
# Yup!

# Get the current prediction probabilities for object on each class using the "softmax" method

y_logits = model4(X_blob_test.to(device))

y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:5])
print(y_pred_probs[:5])

# Note that the sum of each softmax vector is 1 or very close to 1!
print([torch.sum(i) for i in y_pred_probs[:5]])

# Which class does the model think is *most* likely at the index 0 sample?
print(y_pred_probs[:5])
print([torch.argmax(i) for i in y_pred_probs[:5]])

epochs = 100
torch.manual_seed = 42

[i.to(device) for i in [X_blob_train, X_blob_test, y_blob_train, y_blob_test]]

for e in range(epochs):
  model4.train()
  y_logits = model4(X_blob_train)
  y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
  
  loss = loss_fn(y_logits, y_blob_train)
  acc = accuracy_fn(
    y_true=y_blob_train,
    y_pred=y_preds
  )
  
  optimizer.zero_grad()
  
  loss.backward()
  
  optimizer.step()
  
  model4.eval()
  with torch.inference_mode():
    test_logits = model4(X_blob_test)
    test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

    test_loss = loss_fn(test_logits, y_blob_test)
    test_acc = accuracy_fn(
      y_true=y_blob_test,
      y_pred=test_pred
    )
  if e % 10 == 0:
            print(f"Epoch: {e} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%") 
    
model4.eval()
with torch.inference_mode():
  y_logits = model4(X_blob_test)

y_pred_probs = torch.softmax(y_logits, dim=1)
y_preds = y_pred_probs.argmax(dim=1)

# Compare first 10 model preds and test labels
print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")
print(f"Test accuracy: {accuracy_fn(y_true=y_blob_test, y_pred=y_preds)}%")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model4, X_blob_test, y_blob_test)
# plt.show()

# Let's see some metrics

torchmetrics_accuracy = Accuracy(task="multiclass", num_classes=4).to(device)
print(torchmetrics_accuracy(y_preds, y_blob_test))
