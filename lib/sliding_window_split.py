import math

def sliding_window_split(X, W, test_size=1, step_size=1):
    if isinstance(W, float):
        W = int(X.shape[0] * W)
    count = int(X.shape[0] / W) # Number of training windows
    left = int(X.shape[0] % W) # Number of left-out records

    slices = []
    train_start = 0
    train_end = W

    for i in range(count):
        if i == (count - 1) and left > 0:
            train_end+= left
        test_start = train_end
        test_end = test_start + test_size

        slices.append(
            (
                [i for i in range(train_start, train_end)],
                [i for i in range(test_start, test_end)]
            )
        )
        train_start += step_size
        train_end += step_size

    return slices

def get_sliding_slices(X, W, test_size=1, step_size=1):
    result = []
    for begin in range(0, X.shape[0], step_size):
        end = begin + W
        test_end = end + test_size
        if end >= X.shape[0]:
            test_end = X.shape[0]
            end = test_end - test_size
        train_indices = [i for i in range(begin, end)]
        test_indices = [i for i in range(end, test_end)]
        result.append((train_indices, test_indices))
    return result