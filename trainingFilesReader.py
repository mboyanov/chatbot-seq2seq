import sys

import tensorflow as tf


def read(source_path, target_path, max_size = None):
    with tf.gfile.GFile(source_path, mode="r") as source_file, tf.gfile.GFile(target_path, mode="r") as target_file:
        source, target = source_file.readline(), target_file.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            yield(source, target)
            source, target = source_file.readline(), target_file.readline()