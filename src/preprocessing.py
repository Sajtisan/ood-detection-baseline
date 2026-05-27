import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

def min_max_scaler(image):
    """
    Klasszikus min-max normalizáció: a pixelértékeket 0 és 1 közé skálázza.
    """
    return image / 255.0

def z_score_standardize(image):
    """
    Z-score standardizáció: a pixelértékeket úgy skálázza, hogy az adatok átlaga 0 és szórása 1 legyen.
    TF.-ben beépített függvénye képenként végzi el a műveletet, ami hasznos a fényességbeli különbségek eltüntetéséhez.
    """
    return tf.image.per_image_standardization(image)

def create_dataset_pipeline(x, y, batch_size=64, is_training=False, norm_method='minmax'):
    """
    Felépíti a TensorFlow adat pipeline-t a megadott normalizációs módszerrel.
    Paraméterek:
    - x: Képek numpy tenzora
    - y: Címkék (labels) numpy tenzora
    - batch_size: A kötegek mérete (általában 32, 64, vagy 128)
    - is_training: Ha True, bekapcsolja az adatkeverést (shuffling). Tesztnél SOHA ne legyen True!
    - norm_method: 'minmax' vagy 'zscore'
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    if norm_method == 'minmax':
        dataset = dataset.map(lambda img, label: (min_max_scaler(img), label), num_parallel_calls=AUTOTUNE)
    elif norm_method == 'zscore':
        dataset = dataset.map(lambda img, label: (z_score_standardize(img), label), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

def create_inference_pipeline(x, batch_size=64, norm_method='minmax'):
    """
    Könnyített adatcsővezeték az Inferencia (OOD detekció) fázishoz,
    amikor nincs szükségünk a címkékre (y), és szigorúan TILOS keverni (shuffle).
    Paraméterek:
    - x: Képek numpy tenzora
    - batch_size: A kötegek mérete
    - norm_method: 'minmax' vagy 'zscore'
    """
    dataset = tf.data.Dataset.from_tensor_slices(x)
    if norm_method == 'minmax':
        dataset = dataset.map(lambda img, label: (min_max_scaler(img), label), num_parallel_calls=AUTOTUNE)
    elif norm_method == 'zscore':
        dataset = dataset.map(lambda img, label: (z_score_standardize(img), label), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

if __name__ == "__main__":
    from data_loader import load_mnist_id
    
    print("⏳ A Data Pipeline tesztelése...")
    (x_train, y_train), _ = load_mnist_id()
    
    # Készítünk egy optimalizált pipeline-t a betöltött adatokból
    train_dataset = create_dataset_pipeline(x_train, y_train, batch_size=64, is_training=True, norm_method='minmax')
    
    # Kiveszünk egyetlen batch-et a generátorból, hogy ellenőrizzük
    for images, labels in train_dataset.take(1):
        print("\n✅ Batch sikeresen betöltve a GPU-nak!")
        print(f"📦 Batch dimenziója: {images.shape} (batch_size, width, height, channels)")
        print(f"📉 Értékek tartománya (Normalizáció ellenőrzése):")
        print(f"   Min érték: {np.min(images.numpy())} (Elvárt: 0.0)")
        print(f"   Max érték: {np.max(images.numpy())} (Elvárt: 1.0)")