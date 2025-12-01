VERTEX_RANGE = [3, 6]
BATCH_SIZE = 400

NUM_EPOCHS = 150
NUM_TRIALS = 100

POLYGON_PARAMETER_GRID = {
    'learning_rate': [0.0001, 0.0005, 0.001],
    'conv_filters': [
        [16, 32, 64],
        [32, 64, 128],
        [64, 128, 256],
    ],
    'dropout_rate': [0.1, 0.2, 0.3, 0.5],
    'activation': ['relu', 'elu'],
    'weight_decay': 0,
    'patience': 20
}
