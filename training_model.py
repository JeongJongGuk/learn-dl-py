import tensorflow as tf
from keras import callbacks
from dataset import load_cifar10, augment_image
from CNN.AlexNet import AlexNet
from CNN.DenseNet import DenseNet
from CNN.ResNet import ResNet

################################################################################
"""
1. AlexNet
2. ResNet
3. DensNet

"""
# Model Save 이름 지정
MODEL_NAME = ResNet
SAVE_NAME = "ResNet"
DATASET = "C10"

# ChcekPoint, Log 경로 지정
CHECKPOINT_PATH_TO = "checkpoints/"
CHECKPOINT = f"{CHECKPOINT_PATH_TO}{SAVE_NAME}_{DATASET}.h5"
LOG = f"./logs/{SAVE_NAME}_{DATASET}_logs"

# DataSet Hyperparameters
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
NUM_FILTER = 16

"""
AlexNet:
    NUM_FILTER = 64
    ITERATION = 1
    EPOCHS = 400
    BATCH_SIZE = 128
    
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

# Learning Hyperparameters
ITERATION = 1
EPOCHS = 400
BATCH_SIZE = 128
OPTIMIZER = tf.keras.optimizers.Nadam
LOSS_FUNCTION = "categorical_crossentropy"

################################################################################

# Learning Scheduler
LEARN_SCHEDULE = tf.keras.optimizers.schedules.ExponentialDecay
INITIAL_LR = 0.001
decay_rate = 0.96
decay_steps = ITERATION * EPOCHS

learning_rate = LEARN_SCHEDULE(
    initial_learning_rate=INITIAL_LR,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True,
)

# Load DataSet
train_images, train_labels, test_images, test_labels = load_cifar10()

# Load Model
load_model = MODEL_NAME()

model = load_model._build(
    input_shape=INPUT_SHAPE,
    num_class=NUM_CLASSES,
    num_filter=NUM_FILTER,
    dropout_rate=0.2,
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

################################################################################

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

################################################################################
