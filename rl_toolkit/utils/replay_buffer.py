import reverb
import tensorflow as tf


def make_reverb_dataset(
    server_address: str, table: str, batch_size: int, num_parallel_calls: int = 12
):
    def _make_dataset(unused_idx):
        dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=server_address,
            table=table,
            max_in_flight_samples_per_worker=2 * batch_size,
        )

        dataset = dataset.batch(batch_size, drop_remainder=True)

        return dataset

    # Create the dataset
    dataset = tf.data.Dataset.range(num_parallel_calls)
    dataset = dataset.interleave(
        map_func=_make_dataset,
        cycle_length=num_parallel_calls,
        num_parallel_calls=num_parallel_calls,
        deterministic=False,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
