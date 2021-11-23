from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model, save_model_plot, get_unique_filename
import argparse
import os
import tensorflow as tf

def training(config_path):
    config = read_config(config_path)
    
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)

    #CALLBACKS
    log_dir = config["logs"]["logs_dir"]
    tensorboard_root_log_dir = config["logs"]["TENSORBOARD_ROOT_LOG_DIR"]
    tensorboard_logs_dir = os.path.join(log_dir, tensorboard_root_log_dir, get_unique_filename("logs"))
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_dir)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    checkpoint_root_dir = config["artifacts"]["CHECKPOINT_DIR"]
    checkpoint_model_name = config["artifacts"]["checkpoints_model_name"]
    model_checkpoint_dir = os.path.join(artifacts_dir, checkpoint_root_dir, checkpoint_model_name)
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_dir, save_best_only=True)

    CALLBACKS_LIST = [tensorboard_cb, early_stopping_cb, checkpointing_cb]

    # Restart training from checkpoint, using load model
    # history = tf.keras.models.load_model(model_checkpoint_dir)
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_SET, callbacks=CALLBACKS_LIST)

    model_dir = config["artifacts"]["model_dir"]
    plot_dir = config["artifacts"]["plot_dir"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    plot_dir_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)

    model_name = config["artifacts"]["model_name"]
    plot_name = config["artifacts"]["plot_name"]

    save_model(model, model_name, model_dir_path)
    save_model_plot(history, plot_name, plot_dir_path)

    print("\nTo Run Tensorboard Logs Run: \ntensorboard --logdir="+tensorboard_logs_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)
