from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pprint
# Caricamento del dataset
iris = load_iris()
X = iris.data
y = iris.target

# Visualizzazione della distribuzione delle classi
plt.figure(figsize=(12, 4))

# Plot per i sepali
plt.subplot(1, 2, 1)
for i in range(3):
    mask = (y == i)
    print()
    print("La maschera per il target",i,"è:")
    print(mask)
    plt.scatter(X[mask, 0], X[mask, 1], label=iris.target_names[i])
plt.xlabel('Lunghezza sepalo')
plt.ylabel('Larghezza sepalo')
plt.title('Distribuzione dei Sepali')
plt.legend()

# Plot per i petali
plt.subplot(1, 2, 2)
for i in range(3):
    mask = (y == i)
    plt.scatter(X[mask, 2], X[mask, 3], label=iris.target_names[i])
plt.xlabel('Lunghezza petalo')
plt.ylabel('Larghezza petalo')
plt.title('Distribuzione dei Petali')
plt.legend()

plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=26)

