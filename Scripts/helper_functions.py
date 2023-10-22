import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import time

from IPython.display import display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, classification_report

ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]

def create_windows(data, window_size, step_size=1, with_label=True):
    """
    Create windows from a data matrix.
    data: The input data. The last column is assumed to be the label if with_label is True.
    window_size: The size of the sliding window.
    step_size: The step size of the sliding window.
    with_label: Whether the data contains a label column.
    """
    windows, labels = [], []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size, :-1]
        windows.append(window)
        if with_label:
            # Take the mode (most common) label in the window as the label for this window
            label = np.bincount(data[i:i + window_size, -1].astype(int)).argmax()
            labels.append(label)
    return np.array(windows), np.array(labels)

def create_model(X_train, y_train, init_lr = 0.0005, task = 'multi_class'):
  # Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, kernel_size=(4, X_train.shape[2]), activation='relu', input_shape=X_train.shape[1:]),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(y_train.shape[1] if task == 'multi_class' else 1, activation='softmax' if task == 'multi_class' else 'sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr),
                  loss='categorical_crossentropy' if task == 'multi_class' else 'binary_crossentropy',
                  metrics=['accuracy'])
    print("---> Creating CNN model completed")
    return model

def train_model(model_PATH, hist_PATH, X_train, y_train, epochs):
  #Early stopping callback
  #cnn_es_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

  init_lr = 0.0005
  # Check if the CNN model can be reused
  if os.path.exists(model_PATH):
      # load the model;
      cnn_model = load_model(model_PATH)

      #display model architecture;
      cnn_model.summary()

      #train for one epoch
      cnn_model_history = cnn_model.fit(X_train, y_train, epochs=1, #Train model for 1 epoches
                      batch_size=32, validation_split =0.2)
      cnn_model_his = np.load(hist_PATH, allow_pickle=True).item()

  else:
      cnn_model = create_model( X_train, y_train, init_lr =init_lr)
      cnn_model.summary()
      # Start measuring time
      start_time = time.time()
      cnn_model_his= cnn_model.fit(X_train, y_train, epochs=epochs,
                            batch_size=32, validation_split=0.2)
      # Stop measuring time
      end_time = time.time()

      # Calculate the elapsed time
      elapsed_time_cnn = end_time - start_time
      print(f"Model training completed in: {elapsed_time_cnn} seconds")

      cnn_model.save(model_PATH, save_format='tf') #save model

      #Option to save training history. The history can be loaded later for measuring accuracy, loss and learning rates
      #However, it significantly increases the size of the model.

      np.save(hist_PATH, cnn_model_his)
      cnn_model_his = np.load('Model_data/cnn_model_his.npy', allow_pickle=True).item()
  return cnn_model, cnn_model_his

# Function for plotting Accuracy, Loss and Learning curves
def plot_history(model_history):
    # Create a 1x2 grid of subplots
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Plot Accuracy Curve
    axes[0].plot(model_history.history['accuracy'], 'r-', label='Training Accuracy')
    axes[0].plot(model_history.history['val_accuracy'], 'b-', label='Validation Accuracy')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Curve")
    axes[0].grid(True)
    axes[0].legend(loc="lower right")

    # Plot Loss Curve
    axes[1].plot(model_history.history['loss'], 'r-', label='Training Loss')
    axes[1].plot(model_history.history['val_loss'], 'b-', label='Validation Loss')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Curve")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")


    plt.tight_layout()
    plt.show()
    
def plot_cm(y_train, y_train_pred, y_test, y_test_pred):
  #  CONFUSION MATRIX
  y_train = np.argmax(y_train, axis=-1)
  y_test = np.argmax(y_test, axis=-1)

  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
  plt.subplots_adjust(wspace=0.7)
  plt.rc('font', size=9)
  cm_train = ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[0], display_labels=ACT_LABELS)
  axs[0].set_title("Confusion matrix: Training")
  axs[0].set_xticklabels(ACT_LABELS, rotation=90)
  cm_train.im_.colorbar.remove()
  axs[0].set_ylabel('TRUE LABEL')
  axs[0].set_xlabel('PREDICTED LABEL')

  plt.rc('font', size=10)
  cm_test =ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=axs[1], display_labels=ACT_LABELS)
  axs[1].set_title("Confusion matrix: Testing")
  axs[1].set_xticklabels(ACT_LABELS, rotation=90)
  cm_test.im_.colorbar.remove()
  axs[1].set_ylabel('')
  axs[1].set_xlabel('PREDICTED LABEL')
  plt.show()
  
def eval_model(model, X_set, y_set, task = 'multi_class'):
      # Evaluate
  y_pred_proba = model.predict(X_set)
  y_pred = np.argmax(y_pred_proba, axis=1) #if task == 'multi_class' else (y_pred_proba > 0.5).astype("int32")
  y_labels = np.argmax(y_set, axis=1) #if task == 'multi_class' else y_test

  return y_pred_proba, y_pred, y_labels

def perf_metric(y_train,y_train_pred,y_test,y_test_pred):

  # F1 SCORE FOR TRAINING SET

  f1_score_train_cnn = f1_score(y_train, y_train_pred, average='micro')
  accuracy_score_train_cnn = accuracy_score(y_train, y_train_pred)

  # F1 SCORE FOR TESTING SET
  f1_score_test_cnn = f1_score(y_test, y_test_pred, average='micro')
  accuracy_score_test_cnn = accuracy_score(y_test, y_test_pred)

  #prepare data frame for all the precision scores
  dset_ls = ["Training", "Testing"]
  f1_ls_cnn = [f'{f1_score_train_cnn:.2%}' , f'{f1_score_test_cnn:.2%}' ]
  acc_ls_cnn = [f'{accuracy_score_train_cnn:.2%}', f'{accuracy_score_test_cnn:.2%}' ]
  xcols = ['F1 SCORE', 'ACCURACY SCORE']

  data = {'F1 SCORE': f1_ls_cnn, 'ACCURACY SCORE': acc_ls_cnn}
  df = pd.DataFrame(data, index=dset_ls, columns=xcols)
  print("\033[1m","------------PERFORMANCE MEASURES----------",'\033[0m')
  df
  
def transform_data(dataset, task = 'multi_class',
                   window_size=50, step_size=1,
                   positive_label=None, preprocess='standard'):
  target='act'
  # Convert to windows
  if task == 'multi_class':
      data_with_labels = np.c_[dataset.drop(columns=[target]).values,
                              pd.factorize(dataset[target])[0]]  # Convert labels to integers
  else:
      data_with_labels = np.c_[dataset.drop(columns=[target]).values,
                              dataset[target].apply(lambda x: 1 if x == positive_label else 0).values]

  X, y = create_windows(data_with_labels, window_size, step_size)
  if task == 'multi_class':
      y = pd.get_dummies(y).values

  # Preprocess
  if preprocess == 'standard':
      print("---> Data Standardization Started")
      start_std = time.time()
      X = np.array([StandardScaler().fit_transform(window) for window in X])
        # Stop measuring time
      end_std = time.time()

      # Calculate the elapsed time
      elapsed_time_std = end_std - start_std
      print(f"Standardization Completed in: {elapsed_time_std} seconds")
  elif preprocess == 'normal':
      print("---> Data Normalization")
      X = np.array([MinMaxScaler().fit_transform(window) for window in X])

  X = np.expand_dims(X, axis=3)  # Add a channel dimension

  return X, y

def optimized_train_eval_model(dataset, model_PATH, hist_PATH, X, y,
                               test_size=0.2, random_state=None, epochs=10,
                               batch_size=32, save_model=False):



    # Train-test split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("--->Training Model")
    model, history = train_model(model_PATH, hist_PATH, X_train, y_train, epochs=epochs)
    # Get Model Evaluation Results
    print("---> Model Evaluation-Train")
    y_train_pred_proba, y_train_pred, y_train_labels = eval_model(model, X_set = X_train, y_set = y_train)

    print("---> Model Evaluation - Test")
    y_test_pred_proba, y_test_pred, y_test_labels = eval_model(model, X_set = X_test, y_set= y_test)


    return history, X_train, X_test, y_train, y_train_pred, y_train_labels, y_test, y_test_pred, y_test_labels

# Example usage:
#file_path = 'clean.csv'
"""
results = optimized_train_eval_model(file_path, target='act', task='multi_class', positive_label=None,
                 preprocess='standard', test_size=0.2,
                 random_state=42, epochs=10, batch_size=32,
                 save_model=True)

"""

#Function to make prediction
def pred(cnn,X):
    y_pred = cnn.predict(X)
    return y_pred

def perf(y,y_pred):
    # calculates precision score
    prec= precision_score(y, y_pred)
    prec_score=round(prec*100,2)

    # calculates recall score
    rs = recall_score(y, y_pred)
    #print("3. RECALL:", round(rs*100,2), "%")
    rec_score=round(rs*100,2)

    return prec_score, rec_score

def plot_roc_multimodel(y_set1, y_set2, y_set_pred_proba, model_names):
    n_classes = y_set1.shape[1]

    # Create a grid of subplots for ROC curves
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 7))
    axes = axes.ravel()

    for i in range(n_classes):
        ax = axes[i]

        for model_name in model_names:
            if model_name == 'cnn_baseline':
                y_set = y_set1
            elif model_name == 'cnn_ma':
                y_set = y_set2
            else:
                raise ValueError(f"Unknown model name: {model_name}")

            fpr, tpr, _ = roc_curve(y_set[:, i], y_set_pred_proba[model_name][:, i])
            roc_auc = auc(fpr, tpr)
            if model_name == 'cnn_ma':
              ax.plot(fpr, tpr, lw=2, label=f'{model_name}', linestyle=(0,(5,10)))
            else:
              ax.plot(fpr, tpr, lw=2, label=f'{model_name}')

        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([-0.1, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve for Class {i}')
        ax.legend(loc="lower right")

    # Remove any empty subplots
    for i in range(n_classes, n_rows * n_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    
# Function for plotting Accuracy, Loss and Learning curves for multiple models
def plot_history(model_histories, model_labels, linestyles):
    # Create a 1x2 grid of subplots
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Plot Accuracy Curve for each model
    for i, model_history in enumerate(model_histories):
        training_accuracy = model_history.history['accuracy']
        validation_accuracy = model_history.history['val_accuracy']
        training_color = 'r'  # Blue for training accuracy
        validation_color = 'b'  # Red for validation accuracy
        linestyle = linestyles[i]

        label = f'{model_labels[i]} Training Accuracy'
        axes[0].plot(training_accuracy, linestyle, color=training_color, label=label)

        label = f'{model_labels[i]} Validation Accuracy'
        axes[0].plot(validation_accuracy, linestyle, color=validation_color, label=label)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Curve")
    axes[0].grid(True)
    axes[0].legend(loc="lower right")

    # Plot Loss Curve for each model
    for i, model_history in enumerate(model_histories):
        training_loss = model_history.history['loss']
        validation_loss = model_history.history['val_loss']
        training_color = 'r'  # Blue for training loss
        validation_color = 'b'  # Red for validation loss
        linestyle = linestyles[i]

        label = f'{model_labels[i]} Training Loss'
        axes[1].plot(training_loss, linestyle, color=training_color, label=label)

        label = f'{model_labels[i]} Validation Loss'
        axes[1].plot(validation_loss, linestyle, color=validation_color, label=label)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Curve")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    
def plot_cms(y_test_raw, y_test_pred_raw, y_test_quality, y_test_pred_quality):
  #  CONFUSION MATRIX
  #y_train = np.argmax(y_train, axis=-1)
  plt.style.use('bmh')
  y_test_raw = np.argmax(y_test_raw, axis=-1)
  y_test_quality = np.argmax(y_test_quality, axis=-1)

  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
  plt.subplots_adjust(wspace=0.7)
  plt.rc('font', size=9)
  cm_raw =ConfusionMatrixDisplay.from_predictions(y_test_raw, y_test_pred_raw, ax=axs[0], display_labels=ACT_LABELS,normalize="true", values_format=".00%")
  axs[0].set_title("Confusion matrix: Raw Data")
  axs[0].set_xticklabels(ACT_LABELS, rotation=90)
  cm_raw.im_.colorbar.remove()
  axs[0].set_ylabel('TRUE LABEL')
  axs[0].set_xlabel('PREDICTED LABEL')

  plt.rc('font', size=10)
  cm_quality =ConfusionMatrixDisplay.from_predictions(y_test_quality, y_test_pred_quality, ax=axs[1], display_labels=ACT_LABELS,normalize="true", values_format=".00%")
  axs[1].set_title("Confusion matrix: Data portion (MA)")
  axs[1].set_xticklabels(ACT_LABELS, rotation=90)
  cm_quality.im_.colorbar.remove()
  axs[1].set_ylabel('')
  axs[1].set_xlabel('PREDICTED LABEL')
  plt.show()