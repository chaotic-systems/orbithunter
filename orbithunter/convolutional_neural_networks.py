from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Activation
from sklearn.model_selection import train_test_split

__all__ = ['orbit_cnn']


def orbit_cnn(X, y):
    """
    Parameters
    ----------
    X : ndarray
    Array of orbit states with shape (n_samples, time_discretization, space_discretization)
    shaped such that first axis is batch size or number of samples, then the last two dimensions
    are the 'image' dimensions, i.e. the two dimensions to convolve over. I.e. shape for KSE fields is

    
    y : ndarray
    Must have same length along first axis as X. Contains the "true" values of whatever is being predicted; the
    dimension of each sample is the same as the dimension of the prediction/output layer.
    hyperparameters

    Returns
    -------

    Notes
    -----
    Currently requires all field to have the same shape. I don't think they have periodic convolutions but
    there is a way to get around this with padding I'm sure.

    Examples
    --------
  . Getting the correct for of X is as simple as:
    >>> [orbit_.state for orbit_ in iterable_of_orbits]
    """
    X = (X - X.mean()) / X.std()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sample_shape = X[0].shape
    sample_size = X[0].size

    cnn = Sequential()
    cnn.add(Conv2D(filters=32, kernel_size=8, padding='valid', input_shape=sample_shape
                   ))
    cnn.add(AveragePooling2D(pool_size=2))
    cnn.add(Activation('relu'))

    cnn.add(Conv2D(filters=8, kernel_size=8,
                   padding='valid'
                   ))
    cnn.add(AveragePooling2D(pool_size=2))
    cnn.add(Activation('relu'))
    cnn.add(Flatten())
    cnn.add(Dense(int(sample_size)))
    cnn.add(Dense(y.shape[1], activation='relu'))
    cnn.compile(loss='mse')
    history = cnn.fit(X_train, y_train, validation_data=(X_test, y_test))
    return cnn, history, ((X_train, y_train), (X_test,  y_test))



