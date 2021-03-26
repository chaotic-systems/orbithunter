import warnings
warnings.simplefilter('ignore', category=FutureWarning)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv2D, Conv3D, Activation
from tensorflow.keras.layers import AveragePooling1D, AveragePooling2D, AveragePooling3D
warnings.resetwarnings()
from sklearn.model_selection import train_test_split
import numpy as np

__all__ = ["orbit_cnn"]


def orbit_cnn(orbits, target, **kwargs):
    """
    Create and train a deep learning model with 2 convolutional and 2 dense layers with Orbit state input
    Should be used as a crude reference due to its hard-coding.

    Parameters
    ----------
    orbits : numpy.ndarray of orbits.
        Array of orbit states with shape (n_samples, time_discretization, space_discretization)
        shaped such that first axis is batch size or number of samples, then the last two dimensions
        are the 'image' dimensions, i.e. the two dimensions to convolve over. I.e. shape for KSE fields is

    target : numpy.ndarray
        Must have same length along first axis as `orbits`. Can be any type of labels/values the
        dimension of each sample is the same as the dimension of the prediction/output layer.

    kwargs : dict, optional
        May contain any and all extra keyword arguments required for numerical methods and Orbit specific methods.

        `hyper_parameters : tuple`
            Hyper parameters for deep learning layers.


    Returns
    -------
    tensorflow.keras.models.Sequential, tf.keras.callbacks.History, tuple
        The model, its History (training and testing error as a function of epoch number) and tuple containing
        the train test splits of the regressors and target data. Train test split returned as
        (X_train, X_test, y_train, y_test).

    """
    # In order for this to be an option, pool_size has to be an integer i.e. N-dimensional cube.
    dimension = orbits[0].shape
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

    X = np.array([orb.state for orb in orbits])
    X = (X - X.mean()) / X.std()
    # To account for possibly higher values of the velocity field, use some value higher than the actual max.
    X = (X - X.min()) / (1.5 * X.max() - X.min())

    X = np.reshape(X, (*X.shape, 1))
    y = np.array(target).reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=kwargs.get('test_size', 0.2), random_state=kwargs.get('random_state', 0)
    )
    (f1, k1, p1, f2, k2, p2) = kwargs.get("hyper_parameters", (32, 8, 2, 8, 8, 2))
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

    return cnn, history, (X_train, X_test, y_train, y_test)
