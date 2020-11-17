# Unused functions


def expanding_window(pipeline_name, parameters, dataset, symbol, initial_training_window=30, initial_testing_window=7):
    if isinstance(parameters, str):
        with open(parameters, 'r') as f:
            parameters = json.load(f)
    pipeline = importlib.import_module('pipelines.' + pipeline_name)
    features, target, close  = load_symbol(dataset, symbol, target='class')

    # Drop columns whose values are all NaN, as well as rows with ANY nan value, then
    # replace infinity values with nan so that they can later be imputed to a finite value
    features = features.dropna(axis='columns', how='all').replace([np.inf, -np.inf], np.nan)

    X_train = features.iloc[0:initial_training_window].copy() # copy because we're adding rows to this dataset
    X_test = features.iloc[initial_training_window:initial_training_window+initial_testing_window]
    X_trade = features.iloc[initial_training_window+initial_testing_window:]
    y_train = target.iloc[0:initial_training_window].copy() # copy because we're adding rows to this series
    y_test = target.iloc[initial_training_window:initial_training_window+initial_testing_window]
    y_trade = target.iloc[initial_training_window+initial_testing_window:]
    close_train = close.iloc[0:initial_training_window]
    close_test = close.iloc[initial_training_window:initial_training_window+initial_testing_window]
    close_trade = close.iloc[initial_training_window+initial_testing_window:]

    est = pipeline.estimator
    est.set_params(**parameters)
    logger.info('Initial training of {} estimator on {} records'.format(pipeline_name, X_train.shape[0]))
    est.fit(X_train.values, y_train.values)
    logger.info('Testing {} estimator on {} records'.format(pipeline_name, X_test.shape[0]))
    test_predictions = pd.Series(index=X_test.index, dtype='int')
    for i, idx in enumerate(X_test.index):
        sample = X_test.loc[idx]
        shaped = np.reshape(sample.values, (1, -1))
        pred = est.predict(shaped)
        test_predictions.loc[idx] = pred[0]
        X_train.loc[idx] = X_test.loc[idx]
        y_train.loc[idx] = y_test.loc[idx]
        try:
            new_est = est.fit(X_train, y_train)
            est = new_est
        except:
            # Fit failed
            logger.exception('Fit failed at index {}!'.format(idx))
            pass
        _precision = precision_score(y_test.values[0:i], test_predictions.values[0:i], average='micro', zero_division=1)
        logger.info('Estimator precision at day {} ({}): {}'.format(initial_training_window + i + 1, idx, _precision))
    logger.info('Validating {} estimator on {} records'.format(pipeline_name, X_trade.shape[0]))
    validate_predictions = pd.Series(index=X_trade.index)
    for i, idx in enumerate(X_trade.index):
        sample = X_trade.loc[idx]
        shaped = np.reshape(sample.values, (1, -1))
        pred = est.predict(shaped)
        validate_predictions.loc[idx] = pred[0]
        X_train.loc[idx] = X_trade.loc[idx]
        y_train.loc[idx] = y_trade.loc[idx]
        try:
            new_est = est.fit(X_train, y_train)
            est = new_est
        except:
            # Fit failed
            logger.exception('Fit failed at index {}!'.format(idx))
            pass
        _precision = precision_score(y_trade.values[0:i], validate_predictions.values[0:i], average='micro', zero_division=1)
        logger.info('Estimator precision at day {} ({}): {}'.format(initial_training_window + initial_testing_window + i + 1, idx, _precision))
    return est, (y_test, test_predictions, close_test), (y_trade, validate_predictions, close_trade)

def trailing_window(pipeline_name, parameters, dataset, symbol, window_size=30):
    if isinstance(parameters, str):
        with open(parameters, 'r') as f:
            parameters = json.load(f)
    pipeline = importlib.import_module('pipelines.' + pipeline_name)
    features, target, close  = load_symbol(dataset, symbol, target='class')

    # Drop columns whose values are all NaN, as well as rows with ANY nan value, then
    # replace infinity values with nan so that they can later be imputed to a finite value
    features = features.dropna(axis='columns', how='all').replace([np.inf, -np.inf], np.nan)

    est = pipeline.estimator
    est.set_params(**parameters)
    logger.info('Testing {} estimator on {} records'.format(pipeline_name, features.shape[0] - window_size))
    test_predictions = pd.Series(index=features.iloc[window_size:].index, dtype='int')
    indices = []
    # Dalla prima posizione, prendo W record alla volta
    for i in range(0, features.shape[0] - window_size, 1):
        X_train = features.iloc[i:i+window_size]
        y_train = target.iloc[i:i+window_size]
        X_test = features.iloc[i + window_size:i+window_size+1]
        y_test = target.iloc[i + window_size:i+window_size+1]
        try:
            est = est.fit(X_train, y_train)
        except:
            # Fit failed
            logger.exception('Fit failed at index {}!'.format(i))
            pass
        shaped = np.reshape(X_test.values, (1, -1))
        pred = est.predict(shaped)
        idx = X_test.index[0]
        test_predictions.loc[idx] = pred[0]
        indices.append(idx)
        _precision = precision_score(target[indices].values, test_predictions[indices].values, average='micro', zero_division=1)
        logger.info('Estimator precision at day {} ({}): {}'.format(window_size + i + 1, idx, _precision))
    close_test = close.loc[test_predictions.index]
    labels = target.loc[test_predictions.index]
    return labels, test_predictions, close_test
