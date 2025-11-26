#✅1. Import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#✅ 2. Create synthetic dataset

#class 0
x0 = np.random.randn(100,2) + np.array([-2,-2])
y0 = np.zeros((100,1))

#class 1
x1 = np.random.randn(100,2) + np.array([2,2])
y1 = np.ones((100,1))

#Merge
X = np.vstack([x0,x1])
y = np.vstack([y0,y1])

#visualization
plt.scatter(X[:,0], X[:,1], c=y[:,0])
plt.show()


#✅ 3. Build Logistic Regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#✅ 4. Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#✅ 5. Train the model
history = model.fit(X, y, epochs=50, verbose=1)

#✅ 6. Test prediction
sample = np.array([[1.5, 1.2]])
prediction = model.predict(sample)
print("Probability class 1:", prediction[0][0])
print("Predicted label:", (prediction > 0.5).astype(int))

#✅ 7. Check learned weights (optional but educational!)
weights, bias = model.layers[0].get_weights()
print("Weights:", weights)
print("Bias:", bias)


#Extra: Visualizing the decision boundary in Python
# Extract weights and bias
w, b = model.layers[0].get_weights()
w1, w2 = w[0][0], w[1][0]
b = b[0]

# Create line
x1_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
x2_vals = -(w1 * x1_vals + b) / w2

# Plot
plt.scatter(X[:,0], X[:,1], c=y[:,0])
plt.plot(x1_vals, x2_vals, color='red', linewidth=2)
plt.title("Decision Boundary")
plt.show()







