# imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sk:
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# keras:
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.models import clone_model
from keras.utils import set_random_seed

# utilities:
from itertools import product

# set random seed for keras:
set_random_seed(42)

##############################################
################ ML functions ################
##############################################

def scale_only_numeric(X_train, X_val, numeric_cols):
  """
  Scale only numeric columns.
  returns X_train, X_val scaled.
  """
  scaler = StandardScaler()
  scaler.fit(X_train[numeric_cols])
  X_train = pd.DataFrame(scaler.transform(X_train[numeric_cols]),index=X_train.index, columns=numeric_cols).join(X_train.drop(columns=numeric_cols))
  X_val = pd.DataFrame(scaler.transform(X_val[numeric_cols]),index=X_val.index, columns=numeric_cols).join(X_val.drop(columns=numeric_cols))
  return X_train, X_val


def scale_numeric_and_tfidf(X_train, X_val, numeric_cols, tfidf_cols, X_test=None):
    """
    scales numeric features using standart scaler.
    scales tfidf features using normalizer.
    ignore categorical features.
    """
    # initiate scalers:
    standart_scaler = StandardScaler()
    normalizer = Normalizer()

    # split data:
    X_train_numeric = X_train[numeric_cols]
    X_train_tfidf = X_train[tfidf_cols]
    X_train_else = X_train.drop(numeric_cols + tfidf_cols, axis=1)
    X_val_numeric = X_val[numeric_cols]
    X_val_tfidf = X_val[tfidf_cols]
    X_val_else = X_val.drop(numeric_cols + tfidf_cols, axis=1)

    # fit and transform standart scaler:
    standart_scaler.fit(X_train_numeric)
    X_train_numeric = standart_scaler.transform(X_train_numeric)
    X_val_numeric = standart_scaler.transform(X_val_numeric)

    # fit and transform normalizer:
    normalizer.fit(X_train_tfidf)
    X_train_tfidf = normalizer.transform(X_train_tfidf)
    X_val_tfidf = normalizer.transform(X_val_tfidf)

    # concat by index:
    X_train_scaled = pd.DataFrame(np.concatenate((X_train_numeric, X_train_tfidf, X_train_else), axis=1),
                                  columns=numeric_cols + tfidf_cols + X_train_else.columns.to_list(),
                                  index=X_train.index)
    X_val_scaled = pd.DataFrame(np.concatenate((X_val_numeric, X_val_tfidf, X_val_else), axis=1),
                                columns=numeric_cols + tfidf_cols + X_val_else.columns.to_list(),
                                index=X_val.index)
    
    if not isinstance(X_test, type(None)):     # scale also the test set :)
        X_test_numeric = X_test[numeric_cols]
        X_test_tfidf = X_test[tfidf_cols]
        X_test_else = X_test.drop(numeric_cols + tfidf_cols, axis=1)
        X_test_numeric = standart_scaler.transform(X_test_numeric)
        X_test_tfidf = normalizer.transform(X_test_tfidf)
        X_test_scaled = pd.DataFrame(np.concatenate((X_test_numeric, X_test_tfidf, X_test_else), axis=1),
                                columns=numeric_cols + tfidf_cols + X_val_else.columns.to_list(),
                                index=X_test.index)
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    else:
        return X_train_scaled, X_val_scaled    



# grid search function:

def grid_search(param_grid, model, X_train, y_train, X_val, y_val, verbose=10):
    """

    Inputs:
    - param_grid.
    - model.
    - X_train, y_train, X_val, y_val.

    Outputs:
    - results: DataFrame containing R-squared and MSE scores for each hyperparameter combination.
    - best_estimator: according to MSE.
    """

    # Perform grid search:
    results = []
    best_mse = float('inf')
    best_model = None

    # All combinations of hyperparameters:
    param_combinations = list(product(*[[(k, v) for v in vals] for k, vals in param_grid.items()]))
    print('fitting %d hyperparameters combinations' % len(param_combinations))
    cnt = 1

    # Grid search:
    for params in param_combinations:
        # verbose each x iterations:
        if (cnt % verbose) == 0:
            print('iteration number %d' % cnt)
        
        # fit model:
        model.set_params(**dict(params))
        model.fit(X_train, y_train)

        # Make predictions on the validation set:
        y_pred_val = model.predict(X_val)
        r2_val = r2_score(y_val, y_pred_val)
        mse_val = mean_squared_error(y_val, y_pred_val)

        # Make predictions on the train set:
        y_pred_train = model.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)
        mse_train = mean_squared_error(y_train, y_pred_train)

        # append to dataframe:
        results.append({'params': params,
                        'r2 train': r2_train, 'mse train': mse_train,
                        'r2 val': r2_val, 'mse val': mse_val})

        # Update best model if current MSE is better
        if mse_val < best_mse:
            best_mse = mse_val
            best_model = model

        cnt += 1

    # save results to DataFrame
    results_df = pd.DataFrame(results).sort_values(by='mse val')
    print('\nbest mse is %.2f' % best_mse)
    print('best params:\n', results_df.iloc[0]['params'])

    return results_df, best_model


# plot predictions:

def plot_predictions(y_train, y_train_pred, y_val, y_val_pred):
    """
    Plots y_val vs y_val_pred and y_train vs y_train_pred on a log scale with a red y=x line.
    """

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot y_val vs y_val_pred
    ax1.scatter(y_val, y_val_pred, color='blue', alpha=0.5)
    ax1.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red')
    ax1.set_xlabel('Actual (Validation)')
    ax1.set_ylabel('Predicted (Validation)')
    ax1.set_title('Validation Set')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1, y_val.max())
    ax1.set_ylim(1, y_val.max())

    # Plot y_train vs y_train_pred
    ax2.scatter(y_train, y_train_pred, color='green', alpha=0.5)
    ax2.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red')
    ax2.set_xlabel('Actual (Training)')
    ax2.set_ylabel('Predicted (Training)')
    ax2.set_title('Training Set')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1, y_train.max())
    ax2.set_ylim(1, y_val.max())

    # Show plot
    plt.tight_layout()
    plt.show()


##############################################
################ DL functions ################
##############################################

def generate_params(param_grid):
  """
  create a list of all params with respect to the number of hidden layers.
  """
  all_params = []
  for n in param_grid['layers']:  # for each number of hidden layers:
      new_units = [units for units in param_grid['units'] if len(units) == n]
      new_regs = [regs for regs in param_grid['regs'] if len(regs) == n]
      new_dropout_rates = [dropout_rates for dropout_rates in param_grid['dropout_rates'] if len(dropout_rates) == n]
      new_batch_norms = [batch_norms for batch_norms in param_grid['batch_norms'] if len(batch_norms)==n]
      new_param_grid = {'layers': [n], 'units': new_units, 'regs': new_regs,
                        'dropout_rates': new_dropout_rates, 'batch_norms': new_batch_norms,
                        'learning_rate': param_grid['learning_rate'],
                        'batch_size': param_grid['batch_size'], 'epochs': param_grid['epochs'], 'input_dim': [param_grid['input_dim']]}   # this params are not in the build function!
      keys, values = zip(*new_param_grid.items())
      param_combinations_for_n = [dict(zip(keys, v)) for v in product(*values)]
      all_params += param_combinations_for_n
  return all_params


def build_model(input_dim, layers=1, units=[32], regs=[(0.1,0.1)], dropout_rates=[0.2], batch_norms = [False], learning_rate=0.01):
  """
  This funcction inputs:
  - layers: int, number of hidden layers.
  - units: list, number of units in each hidden layer.
  - regularization: list of tuples, (l1, l2) regularization for each hidden layer.
  - dropout_rates: list, dropout rate for each hidden layer.
  -  batch_norms: list, True / False for every hidden layer.
  - learning_rate: float, learning rate for the optimizer.
  """
  model = Sequential()
  input_dim = input_dim

  for i in range(layers):
      # Add each layer
      if i == 0:
          # First layer needs to specify input_dim
          model.add(Dense(units[i], input_dim=input_dim, kernel_regularizer=l1_l2(*regs[i]), kernel_initializer='glorot_uniform'))
          model.add(LeakyReLU(alpha=0.01))
      else:
          model.add(Dense(units[i], kernel_regularizer=l1_l2(*regs[i]), kernel_initializer='glorot_uniform'))
          model.add(LeakyReLU(alpha=0.01))
      if batch_norms[i]:
          model.add(BatchNormalization())
          
      model.add(Dropout(dropout_rates[i]))

  # Output layer
  model.add(Dense(1))

  # Compile model
  optimizer = Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

  return model


def grid_search_keras(param_grid, build_fn, X_train, y_train, X_val, y_val, verbose=10):
    """
    Perform grid search for Keras model and return the best model based on MSE.
    """
    results = []
    best_mse = float('inf')
    best_model = None
    best_history = None
    param_grid['input_dim'] = X_train.shape[1]

    # Generate all combinations of hyperparameters:
    param_combinations = generate_params(param_grid)
    print('Fitting %d hyperparameter combinations' % len(param_combinations))
    
    cnt = 1
    for params in param_combinations:
        # Verbose:
        if cnt % verbose == 0:
            print('Iteration number %d' % cnt)
        # remove batch_size and epochs:
        batch_size = params.pop('batch_size')
        epochs = params.pop('epochs')

        # Build and compile model
        model = build_fn(**params)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_val, y_val))
        
        # Evaluate model
        mse_val = model.evaluate(X_val, y_val, verbose=0)[0]
        mse_train = model.evaluate(X_train, y_train, verbose=0)[0]
        r2_val = r2_score(y_val, model.predict(X_val))
        r2_train = r2_score(y_train, model.predict(X_train))

        # Append results
        results.append({'params': params, 'mse train': mse_train, 'mse val': mse_val,
                        'r2 val': r2_val, 'r2 train': r2_train})

        # Update best model
        if mse_val < best_mse:
            best_mse = mse_val
            best_model = clone_model(model)
            best_model.set_weights(model.get_weights())
            best_history = history

        cnt += 1

    # Plotting
    if best_history is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(best_history.history['loss'], label='Training Loss')
        plt.plot(best_history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Results to DataFrame and sort by validation MSE
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='mse val', inplace=True)

    # print best_params:
    print('\nbest mse is %.2f' % best_mse)
    print('best params:\n', results_df.iloc[0]['params'])

    return results_df, best_model

def cross_validate_best_model(X, y,best_params, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    all_history = []

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model(**best_params)
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=300,      # allow more training epochs
                            batch_size=64,   # same as used before
                            verbose=0)
        all_history.append(history.history)

    return all_history

def find_optimal_epochs(all_history):
    avg_val_loss = np.mean([history['val_loss'] for history in all_history], axis=0)
    optimal_epochs = np.argmin(avg_val_loss) + 1
    return optimal_epochs
