# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, M Zieba
#  2019
# --------------------------------------------------------------------------
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def load_data():
    PICKLE_FILE_PATH = 'train.pkl'
    with open(PICKLE_FILE_PATH, 'rb') as f:
        return pickle.load(f)

def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 9} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    #data = load_data()
    return ((np.random.rand(x.shape[0])*10).astype(int))
    pass

def show():

    pass

def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """


    x = X#.toarray()#.astype(int)
    #print(x)
    x_trainT = np.transpose(X_train)#.toarray()).astype(int)
    #print(x.shape[1] - x @ x_trainT - (1 - x) @ (1 - x_trainT))
    x_a=x @ x_trainT
    x_c=1 - x
    x_d=1 - x_trainT
    x_b=(x_c) @ (x_d)
    #print((x.shape[1] - x @ x_trainT - (1 - x) @ (1 - x_trainT)))
    return x.shape[1] - x @ x_trainT - (1 - x) @ (1 - x_trainT)


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    order = Dist.argsort(kind='mergesort')
    return y[order]


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    number_of_classes = 10
    result_matrix = []
    for i in range(np.shape(y)[0]):
        helper = []
        for j in range(k):
            helper.append(y[i][j])
        line = np.bincount(helper, None, number_of_classes)
        result_matrix.append([line[0] / k, line[1] / k, line[2] / k, line[3] / k, line[4] / k, line[5] / k, line[6] / k, line[7] / k, line[8] / k, line[9] / k])
    return result_matrix
    pass

def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    #print(p_y_x)
    n = len(p_y_x)
    m = len(p_y_x[0])
    res = 0
    for i in range(0, n):
        if (m - np.argmax(p_y_x[i][::-1]) - 1) != y_true[i]:
            res += 1
    return res/n
    pass


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    sorted_labels = sort_train_labels_knn(hamming_distance(Xval, Xtrain), ytrain)
    errors = list(map(lambda k: classification_error(p_y_x_knn(sorted_labels, k), yval), k_values))
    min_index = np.argmin(errors)
    return min(errors), k_values[min_index], errors


def p_y_x_knn1(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    number_of_classes = 10
    resized = np.delete(y, range(k, y.shape[1]), axis=1)
    summed_with_zero = np.vstack(np.apply_along_axis(np.bincount, axis=1, arr=resized, minlength=number_of_classes + 1))
    summed = np.delete(summed_with_zero, 0, axis=1)
    return summed / k

def classification_error1(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    #print(p_y_x)
    number_of_classes = p_y_x.shape[1]
    reversed_rows = np.fliplr(p_y_x)
    predicted = number_of_classes - np.argmax(reversed_rows, axis=1)
    difference = predicted - y_true
    return np.count_nonzero(difference) / y_true.shape[0]