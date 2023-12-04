import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
from loguru import logger
from data_handler.data_clone import clone_data

def create_model():
    """
    Create the model

    Returns
    -------
    keras.models.Sequential
        The model
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(learning_rate=0.01),
                  metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    """
    Train the model

    Parameters
    ----------
    model : keras.models.Sequential
        The model
    X_train : numpy.ndarray
        The training data
    y_train : numpy.ndarray
        The training labels
    X_test : numpy.ndarray
        The test data
    y_test : numpy.ndarray
        The test labels
    epochs : int
        The number of epochs
    batch_size : int
        The batch size
    """

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model

    Parameters
    ----------
    model : keras.models.Sequential
        The model
    X_test : numpy.ndarray
        The test data
    y_test : numpy.ndarray
        The test labels
    """

    score = model.evaluate(X_test, y_test, verbose=0)
    return score

def save_model(model, filename):
    """
    Save the model

    Parameters
    ----------
    model : keras.models.Sequential
        The model
    filename : str
        The filename
    """

    model.save(filename)

def load_model(filename):
    """
    Load the model

    Parameters
    ----------
    filename : str
        The filename

    Returns
    -------
    keras.models.Sequential
        The model
    """

    return keras.models.load_model(filename)

def predict(model, X):
    """
    Predict the labels

    Parameters
    ----------
    model : keras.models.Sequential
        The model
    X : numpy.ndarray
        The input data

    Returns
    -------
    numpy.ndarray
        The predicted labels
    """

    return model.predict(X)

def load_data():
    data_path = {
        'train': './data/csvTrainImages 60k x 784.csv',
        'test': './data/csvTestImages 10k x 784.csv',
        'train_label': './data/csvTrainLabel 60k x 1.csv',
        'test_label': './data/csvTestLabel 10k x 1.csv'
    }
    X_train = np.loadtxt(data_path['train'], delimiter=',')
    y_train = np.loadtxt(data_path['train_label'], delimiter=',')
    X_test = np.loadtxt(data_path['test'], delimiter=',')
    y_test = np.loadtxt(data_path['test_label'], delimiter=',')
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    return X_train, y_train, X_test, y_test


# Load the data
X_train, y_train, X_test, y_test = load_data()

# Create the model
model = create_model()

# Train the model
model = train_model(model, X_train, y_train, X_test, y_test)

# Evaluate the model
score = evaluate_model(model, X_test, y_test)
logger.success('Test loss:', score[0])
logger.success('Test accuracy:', score[1])

# Save the model
save_model(model, 'model.h5')
logger.info('Model is saved to model.h5')