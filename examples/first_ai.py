
import numpy as np
import babyai

# Training data (practice questions)
X_train = np.array([[1,1],[2,2],[3,3],[4,4]])
y_train = np.array([2,4,6,8])

# Test data (real life)
X_test = np.array([[5,5],[6,6]])
y_test = np.array([10,12])

model = babyai.birth_model(input_size=2)
babyai.teach(model, X_train, y_train, lessons=100)

print("Baby guess:", babyai.guess(model, np.array([7,7])))
print("Is baby memorizing?", babyai.is_memorizing(model, X_train, y_train, X_test, y_test))
