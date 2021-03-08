def uniform_epoch(length, n_epochs, sfreq):

    epoch_length = length // n_epochs
    remainder = length % n_epochs

    # print(f"Epoch length is {epoch_length}")
    # print(f"Remainder is {remainder}")

    start = 0
    end = epoch_length+remainder
    epochs = {}

    for i in range(n_epochs):

        epochs[i] = ((start / sfreq, (end - 1) / sfreq))

        start = end
        end += epoch_length

    return epochs


def timing_epochs(timings, fs):

    epochs = []

    for timing in timings:

        epochs.append((int(timing[0] * fs), int(timing[1] * fs)))

    return epochs
