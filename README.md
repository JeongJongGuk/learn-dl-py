# learn-dl-py
Python, Tensorflow

"""
# Model Hyperparameters
NUM_FILTER = 64
DROPOUT_RATE = 0.5

# Learning Hyperparameters
ITERATION = 1
EPOCHS = 100
BATCH_SIZE = 128
OPTIMIZER = tf.keras.optimizers.Nadam
LOSS_FUNCTION = "categorical_crossentropy"

# Learning Scheduler
INITIAL_LR = 0.1
DECAY_RATE = 0.98
LEARN_SCHEDULE = tf.keras.optimizers.schedules.ExponentialDecay
"""

"""
AlexNet: 
    Total params: 6,283,722 (23.97 MB)

AlexNet_nb: 
    Total params: 6,276,234 (23.94 MB)

AlexNet_gloavg: 
    Total params: 5,102,026 (19.46 MB)

AlexNet_gelu: 
    Total params: 6,283,722 (23.97 MB)

AlexNet_gelu_gloavg: 
    Total params: 5,102,026 (19.46 MB)

AlexNet_res: 
    Total params: 1,229,130 (4.69 MB)

AlexNet_res_deep: 
    Total params: 2,781,002 (10.61 MB)

AlexNet_res_deep_deep: 
    Total params: 4,332,874 (16.53 MB)
"""