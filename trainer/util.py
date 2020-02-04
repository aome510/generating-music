# Preprocessing dataset is based on 
# https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

import os
import glob
import numpy as np
import tensorflow as tf
import csv
from music21 import converter, instrument, note, chord
import pickle
from six.moves import urllib

DATA_URL = "https://drive.google.com/uc?id=1ltPQM0xBtL3bFaF7_UapnpvzvnBaFows&export=download"


def download_notes_csv(job_dir):
    temp_file, _ = urllib.request.urlretrieve(DATA_URL)
    tf.gfile.Copy(temp_file, os.path.join(job_dir, 'data', 'notes.csv'), overwrite=True)
    tf.gfile.Remove(temp_file)


def prep_list_of_notes():
    notes = []

    cnt = 0
    for file in glob.glob('midi_songs/*.mid'):
        print("Parsing {}".format(file))

        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)

        if parts: # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                # a note is represented as its string notation (e.g "F#4, G2, E5")
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                # a chord is represented as a string of encoded notes connected by '.' (e.g "0.4.7", "9.2")
                notes.append(".".join(str(n) for n in element.normalOrder))

    with open("./data/notes", "wb") as file:
        pickle.dump(notes, file)


def create_notes_as_csvfile(): # create inputs for Neural Network
    # read list of notes from saved file
    with open("./data/notes", "rb") as file:
        notes = pickle.load(file)

    with open("./data/notes.csv", 'w', newline="") as file:
        writer = csv.writer(file, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for note in notes:
            writer.writerow([note])


# load data notes.csv dataset and then process it
def load_data(job_dir):
    # download notes.csv from URL
    download_notes_csv(job_dir)

    with tf.gfile.Open("{}/data/notes.csv".format(job_dir), 'r') as file:
        reader = csv.reader(file, delimiter=" ", quotechar="|")
        notes = (list(row[0] for row in reader))

    # map note to int and vice versa
    int_to_note = sorted(set(note for note in notes))
    note_to_int = dict((note, id) for id, note in enumerate(int_to_note))

    # convert a mapped note to a numpy array
    mapped_notes = np.array(list(note_to_int[note] for note in notes))

    # standardize data to have mean = 0 and std = 1
    mu = np.mean(mapped_notes)
    sig = np.std(mapped_notes)
    standardized_notes = (mapped_notes - mu) / sig

    # fixed network sequence length to 50
    sequence_length = 50

    network_input = []
    network_output = []

    for i in range(0, len(standardized_notes) - sequence_length):
        # standardize input and output values
        network_input.append(standardized_notes[i : i + sequence_length])
        network_output.append(mapped_notes[i + sequence_length])

    # modify input and output to fit LTSM network structure
    network_input = np.expand_dims(network_input, axis=2)
    network_output = tf.keras.utils.to_categorical(network_output)

    return {
        "input" : network_input, "output" : network_output, 
        "mean" : mu, "std" : sig, "int_to_note" : int_to_note
    }


# Split dataset into train and validation dataset
def split(dataset):
    network_input = dataset["input"]
    network_output = dataset["output"]

    n_all = network_input.shape[0]

    # shuffle input and output
    perms = np.array(list(range(0, n_all)))
    np.random.shuffle(perms)
    network_input = network_input[perms]
    network_output = network_output[perms]

    # split dataset with ration train/val = 8/2
    n_train = int(0.8 * n_all)
    n_val = n_all - n_train

    train_x = network_input[0 : n_train]
    train_y = network_output[0 : n_train]

    val_x = network_input[n_train : n_all]
    val_y = network_output[n_train : n_all]
    return {
        "train_x": train_x, "train_y": train_y,
        "val_x": val_x, "val_y": val_y
    }    
