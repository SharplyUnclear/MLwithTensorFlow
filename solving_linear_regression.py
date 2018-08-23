import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define hyperparameters
learning_rate = 0.01
training_epochs = 100

# Setup fake data
x_train = np.linspace(-1, 1, 101)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33

# Define input and output nodes as placeholders -
# value injected by x_train and y_train
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Defines model as y = w*X:
def model(X, w):
    return tf.multiply(X, w)

# Sets up weight variable:
w = tf.Variable(0.0, name="weights")

# Defines cost function
y_model = model(X, w)
cost = tf.square(Y - y_model)

# Defines operation that will be called on each iteration of learning algorithm
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Sets up session and initialises all variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# loops through dataset multiple times
for epoch in range(training_epochs):
    # Loops through each item in dataset
    for (x, y) in zip(x_train, y_train):
        # Updates the model parameter(s) to try and minimize cost function
        sess.run(train_op, feed_dict={X: x, Y: y})

# Obtains the final parameter value
w_val = sess.run(w)

sess.close()
# Plots the original data
plt.scatter(x_train, y_train)
# Plots the best fit line
y_learned = x_train * w_val
plt.plot(x_train, y_learned, 'r')
plt.show()
