import logging
import warnings
# Suppress tf.function retracing warnings from Keras Tuner internal loop
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

import gc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, backend as K
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
import keras_tuner as kt
import pandas as pd
import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
import json

# ==================== CONFIGURATION ====================
OUTPUT_DIR = '/kaggle/working'
os.makedirs(OUTPUT_DIR, exist_ok=True)

num_datasets = 100
n_train = 1260
n_test = 252
n_slides = 10
ebteda = 0
epochs = 100
depth = 15
indicator = 15
lag = 15
batch_size_train = 1008
batch_size_val = 252
batch_size_test = 252

# ==================== TUNING CONFIGURATION ====================
# Set ENABLE_TUNING = False to skip tuning and use default hyperparameters
ENABLE_TUNING   = False
MAX_TRIALS      = 50    # Number of HP combinations to try per stock
TUNING_EPOCHS   = 100    # Max epochs per trial during search
TUNER_DIRECTORY = os.path.join(OUTPUT_DIR, 'tuner_results')
os.makedirs(TUNER_DIRECTORY, exist_ok=True)

# Tuner type: 'random' | 'hyperband' | 'bayesian'
TUNER_TYPE = 'hyperband'

# Fixed shapes
INPUT_SHAPE = (indicator, lag, 1)

# ---- Number of classes is now a single switch -------------------- #
# 2  -> binary  labels: 0 = SELL,    1 = BUY
# 3  -> 3-class labels: 0 = Falling, 1 = Steady, 2 = Rising
# Change CLASS_NAMES if your own dataframe['LABEL'] uses a different
# mapping/order — the index position must match the integer label value.
NUM_CLASSES  = 2
CLASS_NAMES  = ['SELL', 'Buy']   # index 0=Sell (max), 1=Hold (neither), 2=Buy (min)
# ------------------------------------------------------------------------ #


# Stock ticker names
stock_names = [ "ABT", "ACN", "AMT", "AON", "APD", "AXP", "BA", "BAC", "BBY", "BLK", 
            "BMY", "BSX", "BX", "C", "CAT", "CB", "CI", "CL", "CMG", "COF", 
            "COP", "CRM", "CVS", "CVX", "D", "DE", "DECK", "DG", "DHR", "DIS", 
            "DUK", "ECL", "EL", "ELV", "EMR", "EOG", "ETN", "F", "FCX", "FDX", 
            "GD", "GE", "GIS", "GS", "HSY", "IBM", "ICE", "ITW", "JNJ","JPM", 
            "KMB", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MCO","MDT","MMM", 
            "MO", "MRK", "MS", "NEE", "NEM", "NKE", "NOC", "NSC", "ORCL","OXY", 
            "PFE", "PG", "PGR","PLD", "PM", "PNC", "RTX", "SCHW", "SLB", "SO", 
            "SPGI", "SYK", "T", "TGT", "TJX", "TMO", "TRI", "TXN", "UNH", "UNP", 
            "UPS", "USB", "V","VLO", "VZ", "WFC", "WM", "WMT", "XOM", "YUM"]

# ==================== DATA PREPARATION ====================

def prepare_data_for_slide(matrix, dataframe, slide_idx):
    """Prepare train/val/test arrays for a single slide window."""
    st = (n_test * slide_idx) + ebteda
    en = n_train + st

    train_images = matrix[st:en].reshape(-1, indicator, lag, 1).astype('float32')
    test_images  = matrix[en:en + n_test].reshape(n_test, indicator, lag, 1).astype('float32')
    train_labels = dataframe['LABEL'].iloc[st:en].values.astype('int32')
    test_labels  = dataframe['LABEL'].iloc[en:en + n_test].values.astype('int32')

    split_idx = int(0.8 * len(train_images))
    x_train, x_val = train_images[:split_idx], train_images[split_idx:]
    y_train, y_val = train_labels[:split_idx],  train_labels[split_idx:]

    return x_train, x_val, test_images, y_train, y_val, test_labels


def create_tf_datasets(x_train, x_val, x_test, y_train, y_val, y_test):
    """Wrap numpy arrays in tf.data.Dataset pipelines."""
    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(len(x_train)).batch(len(x_train)).prefetch(tf.data.AUTOTUNE))
    val_ds   = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
                .batch(len(x_val)).prefetch(tf.data.AUTOTUNE))
    test_ds  = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                .batch(len(x_test)).prefetch(tf.data.AUTOTUNE))
    return train_ds, val_ds, test_ds


# ==================== TRAINING ====================

def train_with_callbacks(model, train_ds, val_ds,
                         num_epochs, stock_name, slide_idx,
                         checkpoint_path, learning_rate=1e-3):
    """
    Compile and fit the model with ReduceLROnPlateau, EarlyStopping,
    and ModelCheckpoint.  learning_rate comes from tuned HPs when available.
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-5, min_delta=0.001, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, min_delta=0.01,
                      mode='min', restore_best_weights=False, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy',
                        mode='max', save_best_only=True,
                        save_weights_only=True, verbose=1),
    ]

    print(f"\n{'='*60}")
    print(f"Training: {stock_name} — Slide {slide_idx + 1}  "
          f"(lr={learning_rate:.2e})")
    print(f"{'='*60}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ==================== TESTING ====================

def test_model(model, x_test, y_test):
    """
    Returns:
        y_pred    : predicted class labels (list of ints)
        y_true    : ground-truth labels (list of ints)
        y_prob    : full softmax probability matrix, shape (N, NUM_CLASSES)
                    (list of lists) — works for 2 or 3+ classes
        accuracy  : percentage accuracy
    """
    probs    = model.predict(x_test, verbose=0)
    y_pred   = np.argmax(probs, axis=1).tolist()
    y_prob   = probs.tolist()
    y_true   = y_test.tolist()
    accuracy = 100.0 * accuracy_score(y_true, y_pred)
    return y_pred, y_true, y_prob, accuracy


# ==================== METRICS (generalized to NUM_CLASSES) ====================

def calculate_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    """
    Per-slide metrics for a single stock/slide run.
    Works for 2 or more classes: precision/recall/f1 are reported
    separately for EACH class (keyed by its name from class_names),
    plus overall accuracy and MCC (MCC generalizes to multiclass).
    """
    n_classes = len(class_names)
    labels    = list(range(n_classes))

    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recalls    = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s        = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'mcc':      matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
    }
    for i, name in enumerate(class_names):
        metrics[f'precision_{name.lower()}'] = precisions[i]
        metrics[f'recall_{name.lower()}']    = recalls[i]
        metrics[f'f1_{name.lower()}']        = f1s[i]

    return metrics


def calculate_metrics_from_cm(cm, class_names=CLASS_NAMES):
    """
    Compute accuracy/precision/recall/f1/mcc from an NxN confusion matrix
    (N = len(class_names)). Used to get per-stock metrics (summed over all
    slides for that stock) so we can later report mean ± std across stocks.

    Generalized one-vs-rest per class: for class i,
        TP = cm[i, i]
        FN = row sum    - TP
        FP = column sum - TP
        TN = total       - TP - FN - FP
    """
    cm    = np.asarray(cm, dtype=float)
    n     = cm.shape[0]
    total = cm.sum()

    metrics = {'accuracy': (np.trace(cm) / total) if total > 0 else 0.0}

    # Multiclass MCC from the confusion matrix (generalized formula)
    if n == 2 or total == 0:
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics['mcc'] = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    else:
        # Gorodkin's multiclass MCC generalization
        t_k = cm.sum(axis=1)   # true counts per class
        p_k = cm.sum(axis=0)   # predicted counts per class
        c   = np.trace(cm)
        s   = total
        num = c * s - np.dot(t_k, p_k)
        den = np.sqrt((s**2 - np.dot(p_k, p_k)) * (s**2 - np.dot(t_k, t_k)))
        metrics['mcc'] = float(num / den) if den > 0 else 0.0

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        key = name.lower()
        metrics[f'precision_{key}'] = precision
        metrics[f'recall_{key}']    = recall
        metrics[f'f1_{key}']        = f1

    return metrics


# ==================== VISUALIZATION ====================

def plot_training_curves(history, stock_name, slide_idx):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 20,
                         'axes.titlesize': 20, 'axes.labelsize': 20,
                         'xtick.labelsize': 18, 'ytick.labelsize': 18})

    train_losses     = history.history['loss']
    val_losses       = history.history['val_loss']
    train_accuracies = [v * 100 for v in history.history['accuracy']]
    val_accuracies   = [v * 100 for v in history.history['val_accuracy']]
    epochs_range     = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, val_losses,   'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right', fontsize=18); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs_range, val_accuracies,   'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.legend(loc='lower right', fontsize=18); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    #plt.savefig(os.path.join(OUTPUT_DIR,f'{stock_name}_slide{slide_idx+1}_training.png'), dpi=150)
    #plt.show()
    plt.close(fig)

    print(f"\nFinal Metrics:")
    print(f"  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%")
    print(f"  Val Loss:   {val_losses[-1]:.4f},   Val Acc:   {val_accuracies[-1]:.2f}%")


def plot_confusion_matrix_percent(cm, class_names, title, save_path=None):

    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div-by-zero on an empty class
    cm_percent = (cm / row_sums) * 100

    n = len(class_names)
    labels = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            labels[i, j] = f"{cm_percent[i, j]:.1f}%\n(n={int(cm[i, j]):,})"

    fig, ax = plt.subplots(figsize=(6 + n, 5 + n))
    sns.heatmap(cm_percent, annot=labels, fmt='', linewidths=1,
                cmap='Blues', vmin=0, vmax=100,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '% of true class'},
                annot_kws={"size": 16})
    ax.set_xlabel('Predicted Label', fontsize=18)
    ax.set_ylabel('Actual Label',    fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.show()
    plt.close(fig)


def plot_probability_heatmap(df_all_logits, stock_names, output_dir, value_label):
    print("\n" + "="*80)
    print("GENERATING PROBABILITY HEATMAP")
    print("="*80)

    df_all_logits['Date'] = pd.to_datetime(df_all_logits['Date'], errors='coerce')

    fig, ax = plt.subplots(figsize=(40, 20))
    prob_data = df_all_logits[stock_names].values
    im = ax.imshow(prob_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xlabel('Stocks',          fontsize=50, fontweight='bold')
    ax.set_ylabel('Testing Samples', fontsize=50, fontweight='bold')
    ax.set_title('New York Stock Exchange', fontsize=60, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=50)
    cbar.ax.set_ylabel(value_label, fontsize=50, rotation=270,
                        labelpad=50, fontweight='bold')

    for x in range(len(stock_names) + 1):
        ax.axvline(x - 0.5, color='white', linewidth=2, alpha=0.9)

    tick_interval = max(1, len(stock_names) // 100)
    tick_indices  = np.arange(0, len(stock_names), tick_interval)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([stock_names[i] for i in tick_indices],
                       rotation=90, ha='center', fontsize=25)

    y_tick_interval = max(1, prob_data.shape[0] // 10)
    y_tick_indices  = np.arange(0, prob_data.shape[0], y_tick_interval)
    ax.set_yticks(y_tick_indices)
    dates = df_all_logits['Date']
    date_labels = []
    for idx in y_tick_indices:
        if idx < len(dates) and pd.notna(dates.iloc[idx]):
            fmt = '%Y/%m/%d' if len(y_tick_indices) > 15 else '%Y-%m-%d'
            date_labels.append(dates.iloc[idx].strftime(fmt))
        else:
            date_labels.append(str(idx))
    ax.set_yticklabels(date_labels, fontsize=30)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'probability_heatmap.svg')
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")
    plt.show()
    plt.close(fig)

# ==================== MAIN TRAINING LOOP ====================

def main():
    global df_all_logits, y_prob_all, all_test_dates, all_metrics

    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {gpus if gpus else 'None — using CPU'}")
    print(f"Num classes: {NUM_CLASSES}  ({', '.join(CLASS_NAMES)})")
    print(f"Hyperparameter tuning: {'ENABLED' if ENABLE_TUNING else 'DISABLED'}")
    if ENABLE_TUNING:
        print(f"  Tuner: {TUNER_TYPE.upper()} | Trials: {MAX_TRIALS} | "
              f"Epochs/trial: {TUNING_EPOCHS}")

    matrices   = [globals()[f'm{i}']         for i in range(1, num_datasets + 1)]
    dataframes = [globals()[f'dataframe{i}'] for i in range(1, num_datasets + 1)]

    y_pred_all              = [[] for _ in range(num_datasets)]
    # confidence in the winning class per prediction (works for any NUM_CLASSES)
    y_prob_all              = {name: [] for name in stock_names}
    confusion_matrix_total  = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int) for _ in range(num_datasets)]
    results_per_slide       = []
    all_test_predictions    = {name: [] for name in stock_names}
    all_test_dates          = []

    # Storage for tuned HPs: keyed by stock name, populated once per stock
    all_tuned_hyperparameters = {}

    all_metrics = {'accuracy': 0, 'mcc': 0, 'total_predictions': 0}
    for name in CLASS_NAMES:
        key = name.lower()
        all_metrics[f'precision_{key}'] = 0
        all_metrics[f'recall_{key}']    = 0
        all_metrics[f'f1_{key}']        = 0

    print("\nStarting TensorFlow training pipeline …")
    print(f"Config: {num_datasets} datasets | {n_slides} slides | {epochs} max epochs")

    for dataset_idx in range(num_datasets):
        stock_name = stock_names[dataset_idx]
        print(f"\n{'#'*60}")
        print(f"  Dataset {dataset_idx + 1}/{num_datasets}: {stock_name}")
        print(f"{'#'*60}")

        matrix    = matrices[dataset_idx]
        dataframe = dataframes[dataset_idx]

        # ------------------------------------------------------------------ #
        # SLIDE LOOP                                                           #
        # ------------------------------------------------------------------ #
        for slide_idx in range(n_slides):
            print(f"\n  {'='*50}")
            print(f"  Slide {slide_idx + 1}/{n_slides}")
            print(f"  {'='*50}")

            x_train, x_val, x_test, y_train, y_val, y_test = \
                prepare_data_for_slide(matrix, dataframe, slide_idx)

            print(f"  Train: {x_train.shape} | Val: {x_val.shape} | "
                  f"Test: {x_test.shape}")

            train_ds, val_ds, _ = create_tf_datasets(
                x_train, x_val, x_test, y_train, y_val, y_test
            )

            # ---------------------------------------------------------- #
            # HYPERPARAMETER TUNING — runs once per slide using that     #
            # slide's own train/val data                                  #
            # ---------------------------------------------------------- #
            if ENABLE_TUNING:
                best_hps = tune_hyperparameters(
                    x_train, y_train, x_val, y_val, stock_name, slide_idx
                )
                all_tuned_hyperparameters[f'{stock_name}_slide{slide_idx+1}'] = best_hps

                hp_path = os.path.join(
                    OUTPUT_DIR, f'best_hps_{stock_name}_slide{slide_idx+1}.json'
                )
                with open(hp_path, 'w') as f:
                    json.dump(best_hps, f, indent=4)
                print(f"  Tuned HPs saved → {hp_path}")
            else:
                best_hps = None  # build_model will use its own defaults

            # Build model — uses tuned HPs if available, else defaults
            model = build_model(input_shape=INPUT_SHAPE,
                                num_classes=NUM_CLASSES,
                                hyperparameters=best_hps)

            # Use the tuned learning rate when available, otherwise fall back
            lr = (best_hps.get('learning_rate', 1e-3)
                  if best_hps is not None else 1e-3)

            checkpoint_path = os.path.join(
                OUTPUT_DIR,
                f'best_model_{stock_name}_slide{slide_idx + 1}.weights.h5'
            )

            history = train_with_callbacks(
                model, train_ds, val_ds,
                num_epochs=epochs,
                stock_name=stock_name,
                slide_idx=slide_idx,
                checkpoint_path=checkpoint_path,
                learning_rate=lr
            )

            model.load_weights(checkpoint_path, skip_mismatch=True)

            y_pred, y_true, y_prob, test_acc = \
                test_model(model, x_test, y_test)

            # confidence in the predicted class (works for any NUM_CLASSES)
            y_confidence = [probs[pred] for probs, pred in zip(y_prob, y_pred)]

            # Checkpoint has already been loaded into the model — the file
            # on disk is no longer needed, so free the space immediately.
            try:
                os.remove(checkpoint_path)
            except OSError as e:
                print(f"  Warning: could not remove {checkpoint_path}: {e}")

            val_probs  = model.predict(x_val, verbose=0)
            val_y_pred = np.argmax(val_probs, axis=1).tolist()
            val_metrics  = calculate_metrics(y_val.tolist(), val_y_pred)
            test_metrics = calculate_metrics(y_true, y_pred)

            print(f'\n  Test Acc: {test_acc:.2f}%  Test MCC: {test_metrics["mcc"]:.4f}')
            print(f'  Val  Acc: {val_metrics["accuracy"]*100:.2f}%  '
                  f'Val MCC: {val_metrics["mcc"]:.4f}')

            slide_result = {
                'Stock':          stock_name,
                'Dataset_Index':  dataset_idx + 1,
                'Slide':          slide_idx + 1,
                'Tuning_Enabled': ENABLE_TUNING,
                'Epochs_Trained': len(history.history['loss']),
                'Val_Accuracy':   val_metrics['accuracy'] * 100,
                'Val_MCC':        val_metrics['mcc'],
                'Test_Accuracy':  test_metrics['accuracy'] * 100,
                'Test_MCC':       test_metrics['mcc'],
            }
            for name in CLASS_NAMES:
                key = name.lower()
                cap = name.capitalize()
                slide_result[f'Val_Precision_{cap}']  = val_metrics[f'precision_{key}']
                slide_result[f'Val_Recall_{cap}']     = val_metrics[f'recall_{key}']
                slide_result[f'Val_F1_{cap}']          = val_metrics[f'f1_{key}']
                slide_result[f'Test_Precision_{cap}'] = test_metrics[f'precision_{key}']
                slide_result[f'Test_Recall_{cap}']    = test_metrics[f'recall_{key}']
                slide_result[f'Test_F1_{cap}']         = test_metrics[f'f1_{key}']
            results_per_slide.append(slide_result)

            #plot_training_curves(history, stock_name, slide_idx)

            y_pred_all[dataset_idx].append(y_pred)
            all_test_predictions[stock_name].extend(y_pred)
            y_prob_all[stock_name].extend(y_confidence)

            if dataset_idx == 0:
                test_start_idx = n_train + (n_test * slide_idx) + ebteda
                test_end_idx   = test_start_idx + n_test
                if 'Date' in dataframe.columns:
                    slide_dates = dataframe['Date'].iloc[test_start_idx:test_end_idx].values
                elif 'DATE' in dataframe.columns:
                    slide_dates = dataframe['DATE'].iloc[test_start_idx:test_end_idx].values
                else:
                    slide_dates = dataframe.index[test_start_idx:test_end_idx].values
                all_test_dates.extend(slide_dates)

            cm_slide = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
            print(f"  Confusion Matrix:\n{cm_slide}")
            confusion_matrix_total[dataset_idx] += cm_slide

            # ---------------------------------------------------------- #
            # MEMORY CLEANUP — critical when building/training ~200      #
            # models (num_datasets * n_slides) in one process. Without   #
            # this, Keras/TF keeps every previous model's graph nodes    #
            # alive, RAM climbs slide after slide, and Kaggle eventually #
            # OOM-kills and restarts the kernel.                          #
            # ---------------------------------------------------------- #
            del model, history, train_ds, val_ds
            del x_train, x_val, x_test, y_train, y_val, y_test
            del val_probs, val_y_pred, y_pred, y_prob, y_confidence
            K.clear_session()
            gc.collect()

    # ==================== SAVE TUNED HPs COMBINED ====================
    if ENABLE_TUNING:
        combined_hp_path = os.path.join(OUTPUT_DIR, 'all_tuned_hyperparameters.json')
        with open(combined_hp_path, 'w') as f:
            json.dump(all_tuned_hyperparameters, f, indent=4)
        print(f"\n✓ All tuned HPs saved → {combined_hp_path}")

    # ==================== OVERALL METRICS ====================
    print("\n" + "="*80)
    print("OVERALL METRICS — ALL STOCKS COMBINED")
    print("="*80)

    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for dataset_idx in range(len(confusion_matrix_total)):
        if isinstance(confusion_matrix_total[dataset_idx], np.ndarray):
            total_cm += confusion_matrix_total[dataset_idx]
        else:
            print(f"Warning: Dataset {dataset_idx} has no confusion matrix.")

    print(f"\nTotal Confusion Matrix ({num_datasets} Stocks), classes {CLASS_NAMES}:")
    print(total_cm)
    print(f"\n  Total predictions: {total_cm.sum():,}")

    pooled_metrics = calculate_metrics_from_cm(total_cm, CLASS_NAMES)
    all_metrics.update(pooled_metrics)
    all_metrics['accuracy_percent']  = pooled_metrics['accuracy'] * 100
    all_metrics['total_predictions'] = int(total_cm.sum())
    for i, name in enumerate(CLASS_NAMES):
        all_metrics[f'{name.lower()}_support'] = int(total_cm[i, :].sum())

    total_predictions = all_metrics['total_predictions']
    if total_predictions == 0:
        print("Warning: No predictions to evaluate!")

    print(f"\n{'='*50}")
    print("FINAL METRICS")
    print(f"{'='*50}")
    print(f'Accuracy: {all_metrics["accuracy"]:.6f}  ({all_metrics["accuracy_percent"]:.2f}%)')
    print(f'MCC:      {all_metrics["mcc"]:.6f}')
    for name in CLASS_NAMES:
        key = name.lower()
        print(f'{name:<10} Precision: {all_metrics[f"precision_{key}"]:.6f}  '
              f'Recall: {all_metrics[f"recall_{key}"]:.6f}  '
              f'F1: {all_metrics[f"f1_{key}"]:.6f}')

    # ==================== PER-STOCK MEAN ± STD ====================
    # Computed from each stock's own confusion matrix (summed over its
    # n_slides slides), so std reflects variability *across stocks*.
    per_stock_metrics = [calculate_metrics_from_cm(cm, CLASS_NAMES) for cm in confusion_matrix_total]
    metric_keys = ['accuracy', 'mcc'] + [
        f'{m}_{name.lower()}' for name in CLASS_NAMES for m in ('precision', 'recall', 'f1')
    ]

    metrics_mean_std = {}
    for k in metric_keys:
        vals = np.array([m[k] for m in per_stock_metrics])
        mean_v, std_v = float(vals.mean()), float(vals.std())
        metrics_mean_std[k]           = f"{mean_v:.4f}±{std_v:.4f}"
        metrics_mean_std[f'{k}_mean'] = mean_v
        metrics_mean_std[f'{k}_std']  = std_v

    all_metrics['per_stock_mean_std'] = metrics_mean_std

    print(f"\n{'='*50}")
    print("PER-STOCK METRICS (mean ± std across stocks)")
    print(f"{'='*50}")
    for k in metric_keys:
        print(f"  {k:<20}: {metrics_mean_std[k]}")

    # ==================== CONFUSION MATRIX PLOT (PERCENTAGE) ====================
    if total_predictions > 0:
        plot_confusion_matrix_percent(
            total_cm, CLASS_NAMES,
            title='EFS+HPL(Buy/Sell)',
            save_path=os.path.join(OUTPUT_DIR, 'combined_confusion_matrix.png'),
        )

    # ==================== PROBABILITY / CONFIDENCE HEATMAP ====================
    df_all_logits = pd.DataFrame(y_prob_all)
    df_all_logits.insert(0, 'Sample_ID', range(1, len(df_all_logits) + 1))
    df_all_logits.insert(1, 'Date',  all_test_dates[:len(df_all_logits)])
    df_all_logits.insert(2, 'Slide',
                         [(i // n_test) + 1 for i in range(len(df_all_logits))])

    heatmap_label = 'Probability (BUY)' if NUM_CLASSES == 2 else 'Prediction Confidence'
    plot_probability_heatmap(df_all_logits, stock_names, OUTPUT_DIR, heatmap_label)

    # ==================== SAVE RESULTS ====================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    df_per_slide = pd.DataFrame(results_per_slide)

    summary_data = []
    for dataset_idx in range(num_datasets):
        sn     = stock_names[dataset_idx]
        slices = df_per_slide[df_per_slide['Stock'] == sn]
        row = {
            'Stock':             sn,
            'Tuning_Enabled':    ENABLE_TUNING,
            'Avg_Epochs':        slices['Epochs_Trained'].mean(),
            'Avg_Val_Accuracy':  slices['Val_Accuracy'].mean(),
            'Avg_Val_MCC':       slices['Val_MCC'].mean(),
            'Avg_Test_Accuracy': slices['Test_Accuracy'].mean(),
            'Avg_Test_MCC':      slices['Test_MCC'].mean(),
        }
        for name in CLASS_NAMES:
            cap = name.capitalize()
            row[f'Avg_Val_Precision_{cap}']  = slices[f'Val_Precision_{cap}'].mean()
            row[f'Avg_Val_Recall_{cap}']     = slices[f'Val_Recall_{cap}'].mean()
            row[f'Avg_Val_F1_{cap}']          = slices[f'Val_F1_{cap}'].mean()
            row[f'Avg_Test_Precision_{cap}'] = slices[f'Test_Precision_{cap}'].mean()
            row[f'Avg_Test_Recall_{cap}']    = slices[f'Test_Recall_{cap}'].mean()
            row[f'Avg_Test_F1_{cap}']         = slices[f'Test_F1_{cap}'].mean()
        summary_data.append(row)
    df_summary = pd.DataFrame(summary_data)

    df_predictions = pd.DataFrame(all_test_predictions)
    df_predictions.insert(0, 'Sample_ID', range(1, len(df_predictions) + 1))
    df_predictions.insert(1, 'Date',  all_test_dates[:len(df_predictions)])
    df_predictions.insert(2, 'Slide',
                          [(i // n_test) + 1 for i in range(len(df_predictions))])

    df_per_slide.to_csv(  os.path.join(OUTPUT_DIR, 'results_per_slide.csv'),          index=False)
    df_summary.to_csv(    os.path.join(OUTPUT_DIR, 'results_summary.csv'),            index=False)
    df_predictions.to_csv(os.path.join(OUTPUT_DIR, 'predictions_binary.csv'),         index=False)
    print(f"✓ Training complete!")


if __name__ == "__main__":
    main()
