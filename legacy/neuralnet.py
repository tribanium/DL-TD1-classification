import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(1)  # pour que l'exécution soit déterministe

##########################
# Génération des données #
##########################


# N est le nombre de données d'entrée
# D_in est la dimension des données d'entrée
# D_h le nombre de neurones de la couche cachée
# D_out est la dimension de sortie (nombre de neurones de la couche de sortie)
N, D_in, D_h, D_out = 30, 2, 10, 3


# # Création d'une matrice d'entrée X et de sortie Y avec des valeurs aléatoires
# X = np.random.random((N, D_in))
# Y = np.random.random((N, D_out))


# Initialisation aléatoire des poids du réseau
W1 = 2 * np.random.random((D_in, D_h)) - 1
b1 = np.zeros((1, D_h))
W2 = 2 * np.random.random((D_h, D_out)) - 1
b2 = np.zeros((1, D_out))

learning_rate = 0.1
nb_iter = 100000
for t in range(nb_iter):
    ####################################################
    # Passe avant : calcul de la sortie prédite Y_pred #
    ####################################################
    I1 = X.dot(W1) + b1  # Potentiel d'entrée de la couche cachée
    O1 = 1 / (
        1 + np.exp(-I1)
    )  # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
    I2 = O1.dot(W2) + b2  # Potentiel d'entrée de la couche de sortie
    O2 = 1 / (
        1 + np.exp(-I2)
    )  # Sortie de la couche de sortie (fonction d'activation de typesigmoïde)
    Y_pred = O2  # Les valeurs prédites sont les sorties de la couche de sortie

    ########################################################
    # Calcul et affichage de la fonction perte de type MSE #
    ########################################################
    loss = np.square(Y_pred - Y).sum() / 2
    if t % 1000 == 0:
        print(loss)
        plt.scatter(t, loss, color="blue")
        plt.pause(0.01)

    delta_O2 = Y_pred - Y  # dim : N*D_out
    delta_I2 = ((1 - O2) * O2) * delta_O2  # dim : N*D_out
    dW2 = (O1.T).dot(delta_I2)  # D_h * D_out
    db2 = np.sum(delta_I2, axis=0)  # 1*D_out

    delta_01 = delta_I2.dot(W2.T)  # N*D_h
    delta_I1 = ((1 - O1) * O1) * delta_01  # N*D_h
    dW1 = (X.T).dot(delta_I1)
    db1 = np.sum(delta_I1, axis=0)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

plt.show()
