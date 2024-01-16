import tensorflow as tf
import numpy as np

X = np.array([0, 20, 30, 50, 40, 35], dtype=float)
y = np.array([273, 293, 303, 323, 313, 318], dtype=float)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))


model.compile(optimizer=tf.keras.optimizers.Adam(1), loss="mean_squared_error")
historial = model.fit(X, y, epochs=1000, verbose=False)
y_pred = model.predict([10.0, 15.0, 100.0])
import matplotlib.pyplot as plt

plt.xlabel("# Epoca")
plt.ylabel("Errores")
plt.plot(historial.history["loss"])
plt.show()
print(y_pred)
# print(model.get_weights())
