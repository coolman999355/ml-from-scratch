# training data for a simple ML model that should learn y = 2x
data = [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8)
]

# weight: the parameter the model learns (starts random-ish or small)
weight = 0.1

# bias: a constant added to the prediction so the line doesn’t have to pass through (0,0)
bias = 0.0

# learning rate: controls how big each update step is when learning
learning_rate = 0.01

for epoch in range(1000000):

    total_error = 0.0

    # go through each training example (x, y)
    for x, y in data:

        # core prediction: y = wx + b
        prediction = weight * x + bias

        # error = how far off the prediction is from the correct answer
        error = prediction - y

        # update weight and bias using gradient descent
        # the goal is to reduce error over time
        weight += -learning_rate * error * x
        bias += -learning_rate * error

        # squared error (always positive, penalizes big mistakes more)
        total_error += error ** 2

    # stop early if the model is accurate enough
    if total_error < 0.00001:
        print(f"stopped at epoch {epoch} with total error {total_error}")
        break

print(f"Final weight: {weight}, Final bias: {bias}")
print(f"Total error: {total_error}")

# test the trained model
x = int(input("Enter a value for x: "))
predicted_y = weight * x + bias
print(f"Predicted y for x={x}: {predicted_y}")
