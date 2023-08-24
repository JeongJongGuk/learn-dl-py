import tensorflow as tf
from keras import callbacks
from dataset import load_cifar10, augment_image

from CNN.AlexNet import AlexNet
from CNN.AlexNet_nb import AlexNet_nb
from CNN.AlexNet_gelu import AlexNet_gelu
from CNN.AlexNet_gelu_gloavg import AlexNet_gelu_gloavg
from CNN.AlexNet_gloavg import AlexNet_gloavg
from CNN.AlexNet_res import AlexNet_res
from CNN.AlexNet_res_deep import AlexNet_res_deep

from CNN.DenseNet import DenseNet
from CNN.ResNet import ResNet


"""
1. AlexNet
2. ResNet
3. DensNet
"""

# Model Save 이름 지정
LOAD_MODEL = AlexNet_res_deep
SAVE_NAME = "AlexNet_res_deep"
DATASET = "C10"
# LOG_VERSION = ""

# ChcekPoint, Log 경로 지정
CHECKPOINT = f"checkpoints/{SAVE_NAME}_{DATASET}.h5"
# LOG = f"./logs/{SAVE_NAME}_{DATASET}_{LOG_VERSION}_logs"
LOG = f"./logs/{SAVE_NAME}_{DATASET}_logs"

# DataSet Hyperparameters
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

"""
AlexNet:    Total params: 6,283,722 (23.97 MB)
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

AlexNet_nb: Total params: 6,276,234 (23.94 MB)

AlexNet_gloavg: Total params: 5,102,026 (19.46 MB)

AlexNet_gelu: Total params: 6,283,722 (23.97 MB)

AlexNet_gelu_gloavg: Total params: 5,102,026 (19.46 MB)

AlexNet_res: Total params: 1,229,130 (4.69 MB)

AlexNet_res_deep: Total params: 2781002 (10.61 MB)

ResNet:
    NUM_FILTER = 16
    ITERATION = 1
    EPOCHS = 400
    BATCH_SIZE = 128

DenseNet:
    Growth_Rate = 12
    ITERATION = 1
    EPOCHS = 50
    BATCH_SIZE = 256
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

learning_rate = LEARN_SCHEDULE(
    initial_learning_rate=INITIAL_LR,
    decay_steps=EPOCHS,
    decay_rate=DECAY_RATE,
    staircase=True,
)

# Load DataSet
train_images, train_labels, test_images, test_labels = load_cifar10()

# Load Model
model = LOAD_MODEL()._build(
    input_shape=INPUT_SHAPE,
    num_class=NUM_CLASSES,
    num_filter=NUM_FILTER,
    dropout_rate=DROPOUT_RATE,
    compression=0.25,
)
model.summary()

# Define callbacks
learning_rate_callback = callbacks.LearningRateScheduler(learning_rate)

tensorboard_callback = callbacks.TensorBoard(
    log_dir=LOG, histogram_freq=1, write_graph=True, write_images=True
)

checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=CHECKPOINT,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    save_freq="epoch",
)

# Define callback list
callback_list = [learning_rate_callback, tensorboard_callback, checkpoint_callback]

# Train the model
for iteration in range(ITERATION):
    with tf.device("/CPU:0"):
        print(f"Iteration: {iteration + 1}/{ITERATION}")
        augmented_image = augment_image(train_images, iteration)

        # Check if checkpoint file exists
        try:
            model.load_weights(CHECKPOINT)
            print("Loaded checkpoint weights from:", CHECKPOINT)
        except (OSError, tf.errors.NotFoundError):
            print("Checkpoint weights not found. Starting from scratch.")

    # Compile the model
    model.compile(
        optimizer=OPTIMIZER(learning_rate), loss=LOSS_FUNCTION, metrics=["accuracy"]
    )

    # Augment and fit the model
    history = model.fit(
        augmented_image,
        train_labels,
        steps_per_epoch=len(train_images) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callback_list,
        validation_data=(test_images, test_labels),
    )
