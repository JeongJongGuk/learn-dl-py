import tensorflow as tf
from keras import callbacks
from dataset import load_cifar10, augment_image

from Validation.Plain14 import Plain14
from Validation.ComNet import ComNet
from Validation.ComNet_P import ComNet_P

# Model Save 이름 지정
LOAD_MODEL = ComNet
SAVE_NAME = "ComNet"
DATASET = "C10"
VERSION = "original"

# ChcekPoint, Log 경로 지정
CHECKPOINT = f"checkpoints/{SAVE_NAME}_{DATASET}_{VERSION}.h5"
LOG = f"./logs/{SAVE_NAME}_{DATASET}_{VERSION}_logs"
# LOG = f"./logs/{SAVE_NAME}_{DATASET}_logs"

# DataSet Hyperparameters
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

# Model Hyperparameters
NUM_FILTER = 64
DROPOUT_RATE = 0.2
C_DROPOUT_RATE = 0.1

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
    classes=NUM_CLASSES,
    filters=NUM_FILTER,
    name=f"{SAVE_NAME}",
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
