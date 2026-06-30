# !pip install pretty_midi librosa soundfile tqdm

import os
import numpy as np
import librosa
import pretty_midi
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from sklearn.model_selection import train_test_split

def sparse_weighted_categorical_crossentropy(y_true, y_pred):
    """
    Кастомная loss-функция для ладов с маской для класса тишины (20).
    Уменьшает вес ошибки, если правильный ответ - тишина, заставляя сеть учить настоящие лады.
    """
    # The 20 fret classes have a weight of 1.0, while the silence class has a weight of 0.05.
    class_weights = tf.constant([1.0] * 20 + [0.05], dtype=tf.float32)
    # Determine the weight for each sample in the batch.
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    
    # Standard cross-entropy.
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    # Apply the weights.
    return tf.reduce_mean(cce * weights)

def weighted_binary_crossentropy(y_true, y_pred):
    """
    Кастомная loss-функция для головы onset.
    Решает проблему дисбаланса классов, когда модель "перестраховывается" и пишет 0 (тишину).
    Штрафует за пропуск удара в 15 раз сильнее.
    """
    pos_weight = 15.0
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    bce = - (y_true * pos_weight * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce)

# ==========================================
# 0. Environment detection (Colab)
# ==========================================
IN_COLAB = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    # Specify the path to your folder on Google Drive.
    SAVE_DIR = '/content/drive/MyDrive/dombra_project'
    DATASET_FOLDER = '/content/generated_dataset' # Stored locally in Colab for faster performance.
else:
    SAVE_DIR = '.'
    DATASET_FOLDER = 'generated_dataset'

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 1. Pipeline settings
# ==========================================
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'dombra_crnn_model.keras')
WEIGHTS_SAVE_PATH = os.path.join(SAVE_DIR, 'dombra_crnn_weights.weights.h5')
LOGS_SAVE_PATH = os.path.join(SAVE_DIR, 'training_log.csv')

SR = 22050
HOP_LENGTH = 512
# Replace N_MELS with N_BINS for the Constant-Q Transform.
N_BINS = 168 # 7 octaves × 24 bins per octave.
FMIN = librosa.note_to_hz('C2') # Lowest frequency for the CQT.
FRAME_RATE = SR / float(HOP_LENGTH)

MAX_FRET = 19
NUM_CLASSES = 21  # From 0 to 19 inclusive (20 frets) + 1 silence class (20).
SILENCE_CLASS = 20

BATCH_SIZE = 8
EPOCHS = 50

# ==========================================
# 2. Data preparation (Parser).
# ==========================================
def process_file_pair(track_id):
    wav_path = os.path.join(DATASET_FOLDER, f"track_{track_id}.wav")
    npy_path = os.path.join(DATASET_FOLDER, f"track_{track_id}_labels.npy")
    onset_path = os.path.join(DATASET_FOLDER, f"track_{track_id}_onsets.npy")

    # --- Audio (X) via CQT ---
    audio, _ = librosa.load(wav_path, sr=SR)
    
    # Compute the Constant-Q Transform instead of the Mel spectrogram.
    cqt = librosa.cqt(y=audio, sr=SR, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS, bins_per_octave=24)
    # Convert the amplitude to decibels (logarithmic scale).
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max).T  # (Time, Frequencies)

    # Normalize
    cqt_db = (cqt_db + 80.0) / 80.0
    cqt_db = np.clip(cqt_db, 0.0, 1.0)
    cqt_db = cqt_db[..., np.newaxis]  # (Time, Frequencies, 1)

    # --- Fret labels (y) ---
    labels_raw = np.load(npy_path) # (Frames, 2), values in [-1, 19]
    onsets_raw = np.load(onset_path) # (Frames, 2) - Now two strings!

    # We create a one-hot encoding with the shape: (Frames, 2, NUM_CLASSES)
    frames_len = labels_raw.shape[0]
    labels_onehot = np.zeros((frames_len, 2, NUM_CLASSES), dtype=np.float32)

    for i in range(frames_len):
        f_lower = labels_raw[i, 0]
        f_upper = labels_raw[i, 1]
        
        # We ensure that the labels are within the valid range.
        f_lower = min(max(int(f_lower), 0), NUM_CLASSES - 1)
        f_upper = min(max(int(f_upper), 0), NUM_CLASSES - 1)
        
        labels_onehot[i, 0, f_lower] = 1.0
        labels_onehot[i, 1, f_upper] = 1.0

    # We split the data into y_lower and y_upper: (Frames, NUM_CLASSES)
    y_lower = labels_onehot[:, 0, :]
    y_upper = labels_onehot[:, 1, :]

    # --- Synchronization of lengths ---
    min_len = min(cqt_db.shape[0], y_lower.shape[0], y_upper.shape[0], onsets_raw.shape[0])
    return cqt_db[:min_len], y_lower[:min_len], y_upper[:min_len], onsets_raw[:min_len]


def create_tf_dataset(track_ids, shuffle=False):
    """
    Генератор данных для двух струн.
    """
    def generator():
        ids = list(track_ids)
        if shuffle:
            np.random.shuffle(ids)
        for tid in ids:
            try:
                X, y_l, y_u, y_o = process_file_pair(tid)
                yield X, (y_l, y_u, y_o)
            except Exception as e:
                print(f"Ошибка при обработке track_{tid}: {e}")

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, N_BINS, 1), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32),
                tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32) # Onset now has the shape (None, 2).
            )
        )
    )
    dataset = dataset.padded_batch(
        BATCH_SIZE,
        padded_shapes=(
            [None, N_BINS, 1],
            ([None, NUM_CLASSES], [None, NUM_CLASSES], [None, 2])
        )
    )
    # --- FIX: cache() without arguments caches everything in RAM.
    # With 100 tracks of 10 seconds each, this is ~4–8 GB. Removing cache(), keeping prefetch.
    # Adding repeat() to prevent the generator from running out of data.
    return dataset.repeat().prefetch(tf.data.AUTOTUNE)


# ==========================================
# 3. Architecture of the Neural Network (CRNN)
# ==========================================
def build_model():
    inputs = layers.Input(shape=(None, N_BINS, 1))

    # --- CNN блок ---
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)  # (T, 64, 32)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)  # (T, 32, 64)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)  # (T, 16, 128)
    x = layers.Dropout(0.3)(x)

    # FIX 3: Remove the hardcoded value "16 * 128".
    # Using tf.reshape with a hardcoded shape will fail if N_MELS changes.
    # TimeDistributed(Flatten) correctly flattens (T, H, C) into (T, H*C)
    # regardless of the values of N_MELS and the number of filters.
    x = layers.TimeDistributed(layers.Flatten())(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # --- RNN block ---
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    # --- Output (Three heads) ---
    # We use multi-class classification (Softmax)
    # One specific fret or "silence" is selected from 21 classes.
    out_lower = layers.TimeDistributed(
        layers.Dense(NUM_CLASSES, activation='softmax'), name='out_lower'
    )(x)
    
    out_upper = layers.TimeDistributed(
        layers.Dense(NUM_CLASSES, activation='softmax'), name='out_upper'
    )(x)

    # Prediction of string strike (Onset) - now 2 outputs (one for each string)
    out_onset = layers.TimeDistributed(
        layers.Dense(2, activation='sigmoid'), name='out_onset'
    )(x)

    model = Model(inputs=inputs, outputs=[out_lower, out_upper, out_onset])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'out_lower': sparse_weighted_categorical_crossentropy,
            'out_upper': sparse_weighted_categorical_crossentropy,
            'out_onset': weighted_binary_crossentropy
        },
        metrics={
            'out_lower': [
                tf.keras.metrics.CategoricalAccuracy(name='acc'),
            ],
            'out_upper': [
                tf.keras.metrics.CategoricalAccuracy(name='acc'),
            ],
            'out_onset': [
                tf.keras.metrics.BinaryAccuracy(name='acc'),
            ]
        }
    )
    return model


# ==========================================
# 4. Process Execution
# ==========================================
if __name__ == "__main__":
    print("Поиск файлов...")

    #  Find all .wav files that have a corresponding .npy file.
    track_ids = set()
    for f in os.listdir(DATASET_FOLDER):
        if f.startswith('track_') and f.endswith('.wav'):
            try:
                tid = int(f.split('_')[1].split('.')[0])
                # Check that all data (wav + npy + onsets.npy) exists
                npy_path = os.path.join(DATASET_FOLDER, f"track_{tid}_labels.npy")
                onset_path = os.path.join(DATASET_FOLDER, f"track_{tid}_onsets.npy")
                if os.path.exists(npy_path) and os.path.exists(onset_path):
                    track_ids.add(tid)
            except (ValueError, IndexError):
                print(f"[ПРЕДУПРЕЖДЕНИЕ] Неверный формат имени: {f}, пропускаем.")
    track_ids = sorted(track_ids)

    if not track_ids:
        raise ValueError(f"В папке {DATASET_FOLDER} нет сгенерированных треков!")

    train_ids, val_ids = train_test_split(track_ids, test_size=0.15, random_state=67)

    print(f"Тренировочных треков: {len(train_ids)}")
    print(f"Валидационных треков:  {len(val_ids)}")

    # shuffle=True only for training
    train_ds = create_tf_dataset(train_ids, shuffle=True)
    val_ds   = create_tf_dataset(val_ids,   shuffle=False)

    model = build_model()
    model.summary()

    callbacks_list = [
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min', verbose=1
        ),
        callbacks.ModelCheckpoint(
            WEIGHTS_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min', 
            save_weights_only=True, verbose=1
        ),
        callbacks.CSVLogger(LOGS_SAVE_PATH, append=True),
        callbacks.EarlyStopping(
            patience=15, restore_best_weights=True, monitor='val_loss', mode='min', verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, min_lr=1e-5, monitor='val_loss', mode='min', verbose=1
        ),
    ]

    print("\nСтарт обучения...")
    
    # Calculate the number of steps
    train_steps = len(train_ids) // BATCH_SIZE
    val_steps = len(val_ids) // BATCH_SIZE

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks_list
    )

    print(f"\nОбучение завершено. Лучшая модель сохранена как '{MODEL_SAVE_PATH}'")