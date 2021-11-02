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


if __name__ == "__main__":
    label_dict = unpickle(f"{path}/batches.meta")
    label_names = label_dict[b"label_names"]

    X, Y = lecture_cifar(path=path, nb_batches=5)
    X_train, Y_train, X_test, Y_test = decoupage_donnees(
        X, Y, ratio=0.8, small_sample=False
    )
    distance_matrix = kppv_distances(X_train, X_test)
    accuracy = list()
    x = range(8, 13)
    for K in x:
        Y_pred = kppv_predict(distance_matrix, Y_train, K)
        print(f"Prediction done for {K} classes")
        accuracy.append(eval_classifier(Y_test, Y_pred))

    plt.plot(x, accuracy, "k", x, accuracy, "ro")
    plt.title("Accuracy VS K")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy")
    plt.show()
