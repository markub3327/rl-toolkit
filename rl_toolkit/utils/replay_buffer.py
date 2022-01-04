import reverb
import tensorflow as tf


def make_reverb_dataset(server_address: str, table: str, batch_size: int):
    def _make_dataset(unused_idx):
        return reverb.TrajectoryDataset.from_table_signature(
            server_address=server_address,
            table=table,
            max_in_flight_samples_per_worker=(2 * batch_size),
        )

    # Create the dataset
    dataset = (
        tf.data.Dataset.range(1)
        .repeat()
        .interleave(
            map_func=_make_dataset,
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset
