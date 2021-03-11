from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv2D, Conv3D, Activation
from tensorflow.keras.layers import AveragePooling1D, AveragePooling2D, AveragePooling3D
from sklearn.model_selection import train_test_split
import numpy as np

__all__ = ["orbit_cnn"]


def orbit_cnn(X, y, dimension=2, **kwargs):
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
    # In order for this to be an option, pool_size has to be an integer i.e. N-dimensional cube.
    if dimension == 3:
        conv_layer = Conv3D
        pool_layer = AveragePooling3D
    elif dimension == 2:
        conv_layer = Conv2D
        pool_layer = AveragePooling2D
    elif dimension == 1:
        conv_layer = Conv1D
        pool_layer = AveragePooling1D
    else:
        raise ValueError("Dimension not recognized.")

    X = np.array(X)
    X = (X - X.mean()) / X.std()
    # To account for possibly higher values of the velocity field, use some value higher than the actual max.
    X = (X - X.min()) / (1.5 * X.max() - X.min())

    X = np.reshape(X, (*X.shape, 1))
    y = np.array(y).reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    (f1, k1, p1, f2, k2, p2) = kwargs.get("hyperparameters", (32, 8, 2, 8, 8, 2))
    sample_size = X.shape[0]

    # Initialization of the keras Sequential model, to which the neural net layers will be added.
    cnn = Sequential()
    cnn.add(
        conv_layer(filters=f1, kernel_size=k1, padding="valid", input_shape=X.shape[1:])
    )
    cnn.add(pool_layer(pool_size=p1))
    cnn.add(Activation("relu"))

    cnn.add(conv_layer(filters=f2, kernel_size=k2, padding="valid"))
    cnn.add(pool_layer(pool_size=p2))
    cnn.add(Activation("relu"))
    cnn.add(Flatten())
    cnn.add(Dense(int(sample_size)))
    cnn.add(Dense(y.shape[1], activation="softmax"))
    cnn.compile(loss="mse", optimizer="adam")
    history = cnn.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        verbose=kwargs.get("verbose", 0),
        epochs=kwargs.get("epochs"),
    )

    return cnn, history, ((X_train, y_train), (X_test, y_test))
