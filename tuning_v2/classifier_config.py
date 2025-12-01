VERTEX_RANGE = [3, 6]
BATCH_SIZE = 400

NUM_EPOCHS = 200
NUM_TRIALS = 100


CLASSIFIER_PARAMETER_GRID = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
    'conv_filters': [
        [8, 16, 32],
        [16, 32, 64],
        [8, 16, 32, 64],
        [32, 64, 128],
        [64, 128, 256],
    ],
    'dense_units': [32, 64, 128, 256],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.5],
    'activation': ['relu', 'silu'],
    'weight_decay': [0, 0.0001, 0.0002],
    'patience': [20]
}
