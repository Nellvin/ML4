import pickle
import numpy as np



def load_data():
    PICKLE_FILE_PATH = 'zapisuje_pociete.pkl'
    with open(PICKLE_FILE_PATH, 'rb') as f:
        return pickle.load(f)


def hamming_distance(X, X_train):
    x = X
    x_trainT = np.transpose(X_train)
    x_a = x @ x_trainT
    x_c = 1 - x
    x_d = 1 - x_trainT
    x_b = (x_c) @ (x_d)
    return ((x_a + x_b) * -1)


def sort_train_labels_knn(Dist, y):
    order = Dist.argsort(kind='mergesort')
    return y[order]


def predict(x):
    data = load_data()
    valx=data[0][:1000]
    valy=data[1][:1000]
    datax=data[0][1000:]
    datay=data[1][1000:]
    #print(datax.shape[0])
    ac = list(map(lambda k: erease(k), x))
    acc = np.array(ac)
    posortowane_etykiety = sort_train_labels_knn(hamming_distance(acc, (data[0] / 100)), data[1])
    liczba_danych_testowych = posortowane_etykiety.shape[0]
    helper = []
    for i in range(0, liczba_danych_testowych):
        helper.append(posortowane_etykiety[i][0])

    result = np.array(helper).reshape(liczba_danych_testowych, 1)
    return result
    pass


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    liczba_klas = 10
    macierz_prawdopodobienstw = []
    for i in range(np.shape(y)[0]):
        helper = []
        for j in range(k):
            helper.append(y[i][j])
        etykietki = np.bincount(helper, None, liczba_klas)
        macierz_prawdopodobienstw.append(etykietki/k)
    return macierz_prawdopodobienstw
    pass


def classification_error(p_y_x, y_true):
    liczba_danych_testowych = len(p_y_x)
    poprawne = 0
    for i in range(0, liczba_danych_testowych):
        if (np.argmax(p_y_x[i])) != y_true[i]:
            poprawne += 1
    return poprawne / liczba_danych_testowych
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
    posortowane_etykiety = sort_train_labels_knn(hamming_distance(Xval, Xtrain), ytrain)
    print(posortowane_etykiety.shape[0])
    liczba_danych_testowych = posortowane_etykiety.shape[0]
    helper = []
    for i in range(0, liczba_danych_testowych):
        helper.append(posortowane_etykiety[i][0])

    yyy = np.array(helper)
    print(yyy)

    errors = list(map(lambda k: classification_error(p_y_x_knn(posortowane_etykiety, k), yval), k_values))
    min_index = np.argmin(errors)
    return min(errors), k_values[min_index], errors


def countDifferences(x):
    zmienna = x
    n = 36
    m = 36
    row = 30
    helper = []
    diffInARow = 0
    for i in range(0, n):
        for j in range(1, m):
            diffInARow = diffInARow + np.abs(x[j + 36 * i] - x[j + 36 * i - 1])
        print(diffInARow)
        diffInARow = 0
    return 0



def printt(x):
    z = x.reshape(36,36)
    for i in range(0, 36):
        print(z[i][np.argmin(z[i])])

def printt2828(x):
    z = x.reshape(28,28)
    for i in range(0, 28):
        print(z[i][np.argmin(z[i])])

def erease(x):
    z=x.reshape(36,36)
    ROWS_TO_DELETE=8
    firstRow=0
    now=firstRow
    deleteFromTop=True
    deleterows=0
    while((ROWS_TO_DELETE-deleterows)>0 and deleteFromTop):
        if(z[now][np.argmin(z[now])]!=0):
            z=np.delete(z,0,0)
            deleterows+=1
        else:
            deleteFromTop=False
    for i in range(0,ROWS_TO_DELETE-deleterows):
        z=np.delete(z,np.int(z.shape[0]/36)-1,0)

    z=z.transpose()

    ROWS_TO_DELETE = 8
    firstRow = 0
    now = firstRow
    deleteFromTop = True
    deleterows = 0
    while ((ROWS_TO_DELETE - deleterows) > 0 and deleteFromTop):
        if (z[now][np.argmin(z[now])] != 0):
            z = np.delete(z, 0, 0)
            deleterows += 1
        else:
            deleteFromTop = False
    for i in range(0, ROWS_TO_DELETE - deleterows):
        z = np.delete(z, z.shape[0] - 1, 0)
    #print(z.shape[0])
    #print(z.shape[1])
    z=z.transpose()
    x=np.array(z)
    yyy=x.reshape(28,28)
    x=x.reshape(784,)
    return x


