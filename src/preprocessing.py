from pathlib import Path
import mne
import numpy as np

# Set the log level to 'WARNING' to suppress informational messages from MNE during data loading and processing
mne.set_log_level('WARNING')

# Define the data directory
data_dir = Path("../data")

# Function to load and preprocess the data for a given subject and run
def load_subject(subject_id, run):
    filename = f"S{subject_id:03d}R{run:02d}.edf" # Construct the filename based on the subject ID and run number
    
    file_path = data_dir / filename # Create the full file path by combining the data directory and the filename

    # Check if the file exists. If it does not, print a message and return None for both data and labels.
    if not file_path.exists():
        print(f"File not found for subject {subject_id}, run {run}")
        return None, None

    # Load the raw EEG data from the EDF file. If there is an error during loading, print the error message and return None for both data and labels.
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True) # Load the raw EEG data from the EDF file
    except Exception as e:
        print(f"Error loading file for subject {subject_id}, run {run}: {e}")
        return None, None
    
    raw.resample(128) # Resample the data to 128 Hz since the data varies between 128 and 160 hz. We downsample to lower frequency for consistency.

    raw_filtered = raw.copy().filter(l_freq=8, h_freq=30) # Filter the data to retain frequencies between 8 and 30 Hz
    events, event_id = mne.events_from_annotations(raw_filtered) # Extract events and their corresponding IDs from the annotations in the raw data
 
    # Create epochs based on the events, specifying the time window and baseline correction
    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id={'T1': 2, 'T2': 3},
        tmin=0.0,
        tmax=4.0,
        baseline=None,
        preload=True
    )

    # Extract the data and labels from the epochs. The data is stored in a 3D array (n_epochs, n_channels, n_times), and the labels are derived from the event IDs.
    X = epochs.get_data()
    y = epochs.events[:, 2] - 2

    return X, y

def main():

    list_X = [] # Initialize an empty list to store the extracted data (X) for all subjects and runs
    list_y = [] # Initialize an empty list to store the extracted labels (y) for all subjects and runs
    for subject_id in range(1, 110):
        for run in (4, 8, 12):
            X, y = load_subject(subject_id, run)
            if X is not None and y is not None:
                list_X.append(X) # Append the extracted data (X) to the list of data (list_X)
                list_y.append(y) # Append the extracted labels (y) to the list of labels (list_y)
    
    # Concatenate the lists of data and labels into single arrays. The data (X) is concatenated along the first axis (n_epochs), and the labels (y) are concatenated into a single array.
    X = np.concatenate(list_X, axis=0) # Concatenate the list of data (list_X) into a single array (X)
    y = np.concatenate(list_y) # Concatenate the list of labels (list_y) into a single array (y)
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}") # Print the shapes of the data and labels arrays to verify the dimensions

    #Save processed data so that we don't have to preprocess it again. This will save time when we want to train our model.
    np.save('../data/X.npy', X)
    np.save('../data/y.npy', y)
    print("Saved X and y to data/")

if __name__ == "__main__":
    main()