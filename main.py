# training data so the AI learns y = 2x
data = [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8)
]

# weight is what the AI learns (we start it small so it can improve)
weight = 0.1

# bias is just a value added so the line doesn't have to go through 0
bias = 0.0

# learning rate is how fast the AI learns (too big = messy, too small = slow)
learning_rate = 0.01

for epoch in range(1000000):

    total_error = 0.0

    # go through all the training data
    for x, y in data:

        # prediction formula: y = wx + b
        prediction = weight * x + bias

        # error = how wrong the AI is
        error = prediction - y

        # update weight and bias to make the error smaller
        weight += -learning_rate * error * x
        bias += -learning_rate * error

        # square error so it’s always positive
        total_error += error ** 2

    # stop if it’s already really accurate
    if total_error < 0.00001:
        print(f"stopped at epoch {epoch} with total error {total_error}")
        break

print(f"Final weight: {weight}, Final bias: {bias}")
print(f"Total error: {total_error}")

# test it
x = int(input("Enter a value for x: "))
predicted_y = weight * x + bias
print(f"Predicted y for x={x}: {predicted_y}")
