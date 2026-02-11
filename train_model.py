import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

print("Loading dataset...")
X = np.load("X.npy")
y = np.load("y.npy")

print(f"Loaded: X={X.shape}, y={y.shape}")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

num_classes = 26  # A–Z only
y_cat = to_categorical(y, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

print("Training model...")

model = Sequential([
    Dense(256, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print(f"Accuracy: {acc * 100:.2f}%")

model.save("isl_mediapipe_AZ.keras")
print("Saved model as isl_mediapipe_AZ.keras")
