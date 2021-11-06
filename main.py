import numpy as np
import matplotlib.pyplot as plt


path = "./cifar-10-python/cifar-10-batches-py"
# np.random.seed(1)


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict
    # Keys : [b'batch_label', b'labels', b'data', b'filenames']


def lecture_cifar(path: str, nb_batches):
    X = np.zeros(((nb_batches + 1) * 10000, 3072))
    Y = np.zeros(((nb_batches + 1) * 10000))
    for i in range(1, nb_batches + 1):
        batch_path = f"{path}/data_batch_{str(i)}"
        new_dict = unpickle(batch_path)
        batch_array = new_dict[b"data"]
        batch_labels = new_dict[b"labels"]
        X[(i - 1) * 10000 : i * 10000, :] = batch_array
        Y[(i - 1) * 10000 : i * 10000] = batch_labels

    new_dict = unpickle(f"{path}/test_batch")
    batch_array = new_dict[b"data"]
    batch_labels = new_dict[b"labels"]
    X[nb_batches * 10000 : (nb_batches + 1) * 10000, :] = batch_array
    Y[nb_batches * 10000 : (nb_batches + 1) * 10000] = batch_labels

    X = np.float32(X)
    Y = Y.astype(int)

    return X, Y


def decoupage_donnees(X, Y, ratio=0.8, small_sample=False):
    N = X.shape[0]
    indices = np.array(range(N))
    np.random.shuffle(indices)

    if small_sample:
        X_train = X[indices[:500], :]
        Y_train = Y[indices[:500]]

        X_test = X[indices[-100:], :]
        Y_test = Y[indices[-100:]]

        return X_train, Y_train, X_test, Y_test

    M = int(ratio * N)
    X_train = X[indices[:M], :]
    Y_train = Y[indices[:M]]

    X_test = X[indices[M:], :]
    Y_test = Y[indices[M:]]

    return X_train, Y_train, X_test, Y_test


def kppv_distances(X_train, X_test):

    A_2 = (X_test ** 2).sum(axis=1).reshape((-1, 1))
    B_2 = (X_train ** 2).sum(axis=1).reshape((1, -1))
    # print("A2 : ", A_2.shape)
    # print("B2 : ", B_2.shape)
    C = A_2 + B_2
    dot_matrix = 2 * X_test.dot(X_train.T)
    distance_matrix = C - dot_matrix
    print("Distance matrix", distance_matrix.shape)
    return distance_matrix


def kppv_predict(distance_matrix, Y_train, K):
    N = distance_matrix.shape[0]
    Y_pred = np.zeros(N, dtype=int)
    for test_index in range(N):
        classes_kppv = Y_train[distance_matrix[test_index, :].argsort()[:K]]
        Y_pred[test_index] = np.argmax(np.bincount(classes_kppv))
    return np.array(Y_pred)


def eval_classifier(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)


def run_kppv(X_train, Y_train, X_test, Y_test):
    distance_matrix = kppv_distances(X_train, X_test)
    accuracy = list()
    nb_classes = range(8, 13)
    for K in nb_classes:
        Y_pred = kppv_predict(distance_matrix, Y_train, K)
        print(f"Prediction done for {K} classes")
        accuracy.append(eval_classifier(Y_test, Y_pred))

    plt.plot(nb_classes, accuracy, "k", nb_classes, accuracy, "ro")
    plt.title("Accuracy VS K")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy")
    plt.show()


def one_hot_encoding(Y, K):
    Y_out = np.zeros((len(Y), K))
    for i, value in enumerate(Y):
        Y_out[i, value] = 1
    return Y_out


def run_NN(X_train, Y_train, X_test, Y_test):
    ##########################
    # Génération des données #
    ##########################
    Y_train = one_hot_encoding(Y_train, 10)

    N = len(X_train)  # N est le nombre de données d'entrée
    D_in = len(X_train[0])  # D_in est la dimension des données d'entrée
    D_h = 32  # D_h le nombre de neurones de la couche cachée
    D_out = 10  # D_out est la dimension de sortie (nombre de neurones de la couche de sortie)
    # N, D_in, D_h, D_out = 30, 2, 10, 3

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
    for t in range(1):
        ####################################################
        # Passe avant : calcul de la sortie prédite Y_pred #
        ####################################################
        I1 = X_train.dot(W1) + b1  # Potentiel d'entrée de la couche cachée
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
        loss = np.square(Y_pred - Y_train).sum() / N
        print(Y_pred)
        print(Y_train)
        print(Y_pred - Y_train)
        print((Y_pred - Y_train).sum())
        if t % 1000 == 0:
            print(loss)
            # plt.scatter(t, loss, color="blue")
            # plt.pause(0.01)

        delta_O2 = Y_pred - Y_train  # N*D_out
        delta_I2 = ((1 - O2) * O2) * delta_O2  # N*D_out
        dW2 = (O1.T).dot(delta_I2)  # D_h * D_out
        db2 = np.sum(delta_I2, axis=0)  # 1*D_out

        delta_01 = delta_I2.dot(W2.T)  # N*D_h
        delta_I1 = ((1 - O1) * O1) * delta_01  # N*D_h
        dW1 = (X_train.T).dot(delta_I1)
        db1 = np.sum(delta_I1, axis=0)

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # plt.show()


if __name__ == "__main__":
    label_dict = unpickle(f"{path}/batches.meta")
    label_names = label_dict[b"label_names"]

    X, Y = lecture_cifar(path=path, nb_batches=5)
    X_train, Y_train, X_test, Y_test = decoupage_donnees(
        X, Y, ratio=0.8, small_sample=True
    )
    # run_kppv(X_train, Y_train, X_test, Y_test)
    run_NN(X_train, Y_train, X_test, Y_test)
