import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_backward(dout, x):
    dz = np.array(dout, copy=True)
    dz[x <= 0] = 0
    return dz

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(dout, x):
    s = sigmoid(x)
    return dout * s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_backward(dout, x):
    return dout * (1 - np.tanh(x) ** 2)

def get_activation(name):
    if name == 'relu':
        return relu, relu_backward
    elif name == 'sigmoid':
        return sigmoid, sigmoid_backward
    elif name == 'tanh':
        return tanh, tanh_backward
    else:
        raise ValueError(f"Unknown activation '{name}'")

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True) 
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(logits, y_true, model=None, reg=0.0):
    probs = softmax(logits)
    N = logits.shape[0]
    log_probs = -np.log(probs[np.arange(N), y_true] + 1e-9)
    loss = np.sum(log_probs) / N
    
    # Add L2
    if model:
        for param in model.params:
            if 'W' in param:
                loss += 0.5 * reg * np.sum(model.params[param] ** 2)
    return loss, probs

def cross_entropy_grad(probs, y_true):
    N = probs.shape[0]
    grad = probs.copy()
    grad[np.arange(N), y_true] -= 1
    return grad / N

def sgd(params, grads, lr, weight_decay=0.0):
    for k in params:
        params[k] -= lr * grads[k]

def accuracy(probs, labels):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == labels)