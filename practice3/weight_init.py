from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation
from deepreplay.datasets.ball import load_data

X, y = load_data(n_dims=10)


def build_model(n_layers, input_dim, units, activation, initializer):
    if isinstance(units, list):
        assert len(units) == n_layers
    else:
        units = [units] * n_layers
        model = Sequential()
    # Adds first hidden layer with input_dim parameter
    model.add(Dense(units=units[0],
                    input_dim=input_dim,
                    activation=activation,
                    kernel_initializer=initializer,
                    name='h1'))
    # Adds remaining hidden layers
    for i in range(2, n_layers + 1):
        model.add(Dense(units=units[i - 1],
                        activation=activation,
                        kernel_initializer=initializer,
                        name='h{}'.format(i)))
    # Adds output layer
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer=initializer, name='o'))
    # Compiles the model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])
    return model
