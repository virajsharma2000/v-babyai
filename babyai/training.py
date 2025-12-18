
import numpy as np

def birth_model(input_size):
    """
    Baby is born.
    No knowledge, only curiosity.
    """
    from .core import BabyModel
    return BabyModel(input_size)


def guess(model, X):
    """
    Baby makes a guess based on current understanding.
    """
    return np.dot(X, model.weights) + model.bias


def how_wrong(model, X, y):
    """
    Loss function.
    Tells how wrong the baby is.
    """
    predictions = guess(model, X)
    return np.mean((predictions - y) ** 2)


def adjust_direction(model, X, y, lr=0.01):
    """
    Gradient adjustment.
    Baby slightly changes thinking direction.
    """
    predictions = guess(model, X)

    # Calculate gradients (hidden math)
    dW = 2 * np.dot(X.T, (predictions - y)) / len(y)
    dB = 2 * np.mean(predictions - y)

    # Update thinking
    model.weights -= lr * dW
    model.bias -= lr * dB


def teach(model, X, y, lessons=100, lr=0.01, verbose=True):
    """
    Teaching loop.
    Baby learns through repetition.
    """
    for lesson in range(lessons):
        adjust_direction(model, X, y, lr)

        if verbose and lesson % 10 == 0:
            print(f"Lesson {lesson}: How wrong? {how_wrong(model, X, y):.4f}")


def is_memorizing(model, train_X, train_y, test_X, test_y):
    """
    Checks overfitting.
    If baby does well in practice but fails in real life.
    """
    train_loss = how_wrong(model, train_X, train_y)
    test_loss = how_wrong(model, test_X, test_y)

    # If test loss is much higher, baby memorized
    return test_loss > train_loss * 2
