import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer(as_frame=True)

# Convert to a DataFrame
df = data.frame

# Display the first few rows of the DataFrame

X = df[['mean radius', 'mean texture']].to_numpy()
y = df['target'].to_numpy()


class Perceptron:
	def __init__(self, num_features, alpha=0.1):
		self.num_features = num_features
		self.weights = [random.uniform(-0.5, 0.5)] * num_features
		self.bias = 0.0
		self.alpha = alpha
	
	def forward(self, x):
		weighted_sum_z = self.bias
		for i in range(self.num_features):
			weighted_sum_z += x[i] * self.weights[i]
		
		#threshold
		if (0 < weighted_sum_z):
			return 1
		else:
			return 0
		
		return prediction
	
	def update(self, x, true_y):
		prediction = self.forward(x)
		error = true_y - prediction

		self.bias += self.alpha * error
		for i in range(self.num_features):
			self.weights[i] += self.alpha * error * x[i]
		
		return error

def train(model, features, targets, epochs):
	for epoch in range(epochs):
		error_count = 0

		for x, y in zip(features, targets):
			error = model.update(x, y)
			error_count += abs(error)
		
		print(f"Epoch {epoch + 1} error {error_count}")
		if not error_count:
			break

def compute_accuracy(model, features, targets):
	correct = 0.0

	for x, y in zip(features, targets):
		prediction = model.forward(x)
		correct += int(prediction == y)
	
	return correct / len(targets)

ppn = Perceptron(2)

train(ppn, X, y, 1000)

print("weights:", ppn.weights)
print("bias:", ppn.bias)
target_count = np.bincount(y)
print("proportion of majority class in dataset:", np.max(target_count) / len(y))

train_acc = compute_accuracy(ppn, X, y)
print(f"Accuracy: {train_acc * 100}%")


#Plot the Data
plt.plot(
	X[y == 0, 0],
	X[y == 0, 1],
	marker="D",
	markersize=10,
	linestyle="",
	label="malignant",
)

plt.plot(
	X[y == 1, 0],
	X[y == 1, 1],
	marker="^",
	markersize=13,
	linestyle="",
	label="benign",
)

#line
x_line = np.linspace(0, 40, 10)

y_line = (-x_line * ppn.weights[0] - ppn.bias) / ppn.weights[1]

plt.plot(x_line,y_line)


plt.legend(loc=2)

plt.xlim([0, 40])
plt.ylim([0, 40])

plt.xlabel("radius mean", fontsize=12)
plt.ylabel("texture mean", fontsize=12)

plt.grid()
plt.show()