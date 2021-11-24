import tensorflow as tf
import numpy as np
import time
import os

def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name


def get_callbacks(config, X_train):
    log_dir = config["logs"]["logs_dir"]
    tensorboard_root_log_dir = config["logs"]["TENSORBOARD_ROOT_LOG_DIR"]
    tensorboard_logs_dir = os.path.join(log_dir, tensorboard_root_log_dir, get_timestamp("tb_logs"))
    os.makedirs(tensorboard_logs_dir, exist_ok=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_dir)

    file_writer = tf.summary.create_file_writer(logdir=tensorboard_logs_dir)

    with file_writer.as_default():
        images = np.reshape(X_train[10:30], (-1, 28, 28, 1)) ### <<< 20, 28, 28, 1
        tf.summary.image("20 handritten digit samples", images, max_outputs=25, step=0)

    params = config["params"]
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=params["patience"], 
                        restore_best_weights=params["restore_best_weights"])

    artifacts = config["artifacts"]
    checkpoint_root_dir = config["artifacts"]["CHECKPOINT_DIR"]
    checkpoint_model_name = config["artifacts"]["checkpoints_model_name"]
    CKPT_dir = os.path.join(artifacts["artifacts_dir"], artifacts["CHECKPOINT_DIR"])
    os.makedirs(CKPT_dir, exist_ok=True)
    model_checkpoint_dir = os.path.join(CKPT_dir, checkpoint_model_name)
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_dir, save_best_only=True)


    return [tensorboard_cb, early_stopping_cb, checkpointing_cb]