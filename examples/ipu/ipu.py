import os
import time

import determined as det
import determined._core as core

import tensorflow as tf
from tensorflow.python import ipu


# Create a simple model.
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])


# Create a dataset for the model.
def create_dataset():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = x_train / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
    train_ds = train_ds.map(lambda d, l:
                            (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))

    return train_ds.repeat().prefetch(16)


def main(core_context, hparams, user_data, latest_checkpoint):
    # Configure the IPU device.
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    dataset = create_dataset()

    # Create a strategy for execution on the IPU.
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        # Create a Keras model inside the strategy.
        model = create_model()

        # Compile the model for training.
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=["accuracy"],
        )

        model.fit(dataset, epochs=2, steps_per_epoch=100)


if __name__ == "__main__":
    # There's a new (yet-undocumented) ClusterInfo object that tells you information about the
    # Determined cluster you're running in and about the task you are currently running with.  If
    # you are careful to write you code so that only the highest layers of the code access the
    # ClusterInfo then you should be able to run most of your code on- or off-cluster.  Or at least,
    # that's the idea.
    info = det.get_cluster_info()
    assert info is not None, "this example only runs on-cluster, sorry"

    # Here's how you might use the ClusterInfo object:

    # This should be the sample of hparams chosen for this trial.  This is a trial-specific API.
    hparams = info.trial.hparams

    # This will read the `data` field of the experiment config.  Today this is trial-specific but we
    # will be allowing all configs to have a `data` field in the future.
    user_data = info.user_data

    # Find out what checkpoint this task left off on in its previous run.  Today this is
    # trial-specific but we will be checkpointing APIs in non-trial tasks in the future.
    latest_checkpoint = info.latest_checkpoint

    # If you wanted to run in multiple processes (and/or multiple nodes) you would need a
    # DistributedContext.  If none is provided we set up a dummy DistributedContext that assumes
    # there is only one process participating in the Core API.  Note that subprocesses like
    # dataloader workers are irrelevant here.
    #
    # distributed = DistributedContext(
    #     rank=0,
    #     size=1,
    #     local_rank=0,
    #     local_size=1,
    #     cross_rank=0,
    #     cross_rank=1,
    # )
    # with core.init(distributed=distributed) as core_context:
    #    ...

    # This context manager handles things like closing the Distributed context, starting/stopping
    # the preemption backround thread (which polls the master to detect preemption signals), or
    # catching InvalidHP exceptions (which affect the searcher, and probably aren't relevant today).
    with core.init() as core_context:
        main(core_context, hparams, user_data, latest_checkpoint)
