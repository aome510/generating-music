import tensorflow as tf
import numpy as np
import os

from trainer import task, util

if __name__ == "__main__":
    args = task.get_args()

    dataset = util.load_data(args.job_dir)
    input_dim = (dataset["input"].shape[1], dataset["input"].shape[2])
    output_dim = (dataset["output"].shape[1])
    int_to_node = dataset["int_to_note"]
    mean = dataset["mean"]
    std = dataset["std"]
    network_input = dataset["input"]

    model = task.create_model(input_dim, output_dim)
    file = tf.gfile.Glob(os.path.join(args.job_dir, '*.hdf5'))
    source_file = file[0]
    model.load_weights(source_file)

    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    prediction_output = []

    print(pattern)

    # generate 500 notes
    for i in range(500):
        if i % 100 == 0:
            print("Prediction Checkpoint", i)

        prediction_input = np.expand_dims(pattern, axis=0)
        
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        output_note = int_to_node[index]

        prediction_output.append(output_note)
        print(output_note)

        index = (index - mean / std)
        pattern = np.append(pattern, np.array([[index]]), axis=0)
        pattern = pattern[1 : ]