"""
Handwritten Digit Recognizer
-----------------------------
A beginner Machine Learning project that trains a model to recognize
handwritten digits (0-9) using the classic MNIST-style dataset.

Author: Victor Samuel Ekpenyong
"""

# Step 1: Import the libraries we need
from sklearn.datasets import load_digits          # a small built-in handwriting dataset
from sklearn.model_selection import train_test_split  # to split data into train/test
from sklearn.linear_model import LogisticRegression    # our beginner-friendly model
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 2: Load the dataset
# load_digits() gives us 1,797 small images of handwritten digits (8x8 pixels each)
digits = load_digits()

print("Dataset loaded!")
print("Number of images:", len(digits.images))
print("Each image is", digits.images[0].shape, "pixels")

# Step 3: Look at one example (optional, just to understand the data)
plt.figure(figsize=(3, 3))
plt.imshow(digits.images[0], cmap="gray")
plt.title(f"This digit is labeled: {digits.target[0]}")
plt.savefig("sample_digit.png")
print("Saved an example digit image as sample_digit.png")

# Step 4: Prepare the data
# X = the pixel data (what the model sees)
# y = the correct answer (the actual digit, 0-9)
X = digits.data
y = digits.target

# Split data: 80% for training the model, 20% for testing how well it learned
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining on {len(X_train)} images, testing on {len(X_test)} images")

# Step 5: Create and train the model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

print("\nModel training complete!")

# Step 6: Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy on unseen test data: {accuracy * 100:.2f}%")

# Step 7 (optional but nice): Detailed performance report
print("\nDetailed performance report:")
print(classification_report(y_test, predictions))

# Step 8: Try predicting a single new image and show it
sample_index = 5
sample_image = X_test[sample_index].reshape(1, -1)
predicted_digit = model.predict(sample_image)[0]
actual_digit = y_test[sample_index]

print(f"\nExample prediction -> Model guessed: {predicted_digit}, Actual answer: {actual_digit}")
