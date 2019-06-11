import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

InputData = np.array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 1]])

TargetData = np.array([[0], [1], [1], [0], [1], [0]])

TestData = np.array([[1, 1, 0],
                     [0, 1, 1]])

w1 = np.zeros((4, 3))
b1 = np.random.randn(4, 1)

w2 = np.zeros((1, 4))
b2 = np.random.randn()



iterations = 100000
lr = 0.1
costlist = []

for i in range(iterations):
    random = np.random.choice(len(InputData))

    if i % 100 == 0:
        c=0
        for j in range(len(InputData)):
            ze1 = np.dot(w1, InputData[j].reshape(3, 1)) + b1
            ae1 = sigmoid(ze1)

            ze2 = np.dot(w2, ae1) + b2
            ae2 = sigmoid(ze2)
            c += float(np.square(ae2 - TargetData[j]))
        costlist.append(c)

    z1 = np.dot(w1, InputData[random].reshape(3, 1)) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    #backprop
    dcda2 = 2 * (a2 - TargetData[random])
    da2dz2 = sigmoid_p(z2)
    dz2dw2 = a1

    dz2da1 = w2
    da1dz1 = sigmoid_p(z1)
    dz1dw1 = InputData[random].reshape(1, 3)

    w2 = (w2.T - lr * dcda2 * da2dz2 * dz2dw2).T
    b2 = b2 - lr * dcda2 * da2dz2

    w1 = w1 - np.dot((lr * dcda2 * da2dz2 * w2.T * da1dz1), dz1dw1)
    b1 = b1 - lr * dcda2 * da2dz2 * w2.T * da1dz1

print("w1 \n", w1)
print("b1 \n", b1)
print("w2 \n", w2)
print("b2 \n", b2, "\n")

for j in range(len(InputData)):
    ze1 = np.dot(w1, InputData[j].reshape(3, 1)) + b1
    ae1 = sigmoid(ze1)

    ze2 = np.dot(w2, ae1) + b2
    ae2 = sigmoid(ze2)
    c = float(np.square(ae2 - TargetData[j]))
    print("Prediction: ", ae2)
    print("Cost", c, "\n")


plt.plot(costlist)
plt.show()
