import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_id():
    """
    Betölti a MNIST (In-Distribution) adathalmazt.
    A CNN miatta (28, 28) dimenziót (28, 28, 1)-re bőviti.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32')
    x_test = np.expand_dims(x_test, axis=-1).astype('float32')
    return (x_train, y_train), (x_test, y_test)

def load_fashion_mnist_ood():
    """
    Betölti a Fashion-MNIST (Out-of-Distribution) adathalmazt.
    Mivel az OOD adatokat csak tesztelésre használjuk, csak a teszt részét töltjük be. (meg ezeken tilos tanítani)
    """
    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = np.expand_dims(x_test, axis=-1).astype('float32')
    return (x_test, y_test)

def plot_id_vs_ood_batch(id_images, ood_images, num_samples=5):
    """
    Vizuális ellenőrző script.
    """
    plt.figure(figsize=(num_samples * 2, 4))

    for i in range(num_samples):
        ax = plt.subplot(2, num_samples, i + 1)
        plt.imshow(id_images[i].squeeze(), cmap='gray')
        plt.title("ID (MNIST)")
        plt.axis('off')

        ax = plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(ood_images[i].squeeze(), cmap='gray')
        plt.title("OOD (Fashion-MNIST)")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("⏳ Adatok betöltése folyamatban...\n")
    
    (x_train_id, y_train_id), (x_test_id, y_test_id) = load_mnist_id()
    x_test_ood, y_test_ood = load_fashion_mnist_ood()

    print(f"✅ MNIST Train (ID) shape:    {x_train_id.shape}")
    print(f"✅ MNIST Test (ID) shape:     {x_test_id.shape}")
    print(f"✅ Fashion-MNIST (OOD) shape: {x_test_ood.shape}\n")

    print("🎨 Vizuális ellenőrzés (Plot) indítása...")
    # Az első 5 képet átadjuk a plotoló függvénynek
    plot_id_vs_ood_batch(x_test_id, x_test_ood, num_samples=5)