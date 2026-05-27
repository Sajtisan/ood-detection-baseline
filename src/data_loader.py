import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io  as sio

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
    Vizuális ellenőrző szkript: egymás alá rajzolja az ID és az OOD képeket.
    Kezeli a szürkeárnyalatos és az RGB képeket, valamint a float32 figyelmeztetéseket is.
    """
    plt.figure(figsize=(num_samples * 2, 4))

    for i in range(num_samples):
        # --- FELSŐ SOR: In-Distribution ---
        ax = plt.subplot(2, num_samples, i + 1)
        img_id = id_images[i].squeeze()
        
        # Ha float32, alakítsuk át uint8-ra csak a rajzolás kedvéért (hogy ne sírjon a Matplotlib)
        if img_id.dtype != np.uint8:
            img_id = img_id.astype('uint8')
            
        # Ha 2 dimenziós maradt (szürke), kell a cmap='gray', különben színes
        if len(img_id.shape) == 2:
            plt.imshow(img_id, cmap='gray')
        else:
            plt.imshow(img_id)
            
        plt.title("ID Data")
        plt.axis('off')

        # --- ALSÓ SOR: Out-of-Distribution ---
        ax = plt.subplot(2, num_samples, i + 1 + num_samples)
        img_ood = ood_images[i].squeeze()
        
        if img_ood.dtype != np.uint8:
            img_ood = img_ood.astype('uint8')

        if len(img_ood.shape) == 2:
            plt.imshow(img_ood, cmap='gray')
        else:
            plt.imshow(img_ood)
            
        plt.title("OOD Data")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def load_cifar10_id():
    """
    Betölti a CIFAR-10 (In-Distribution) RGB adathalmazt.
    A CNN miatt (32, 32, 3) dimenziót használunk.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)

def load_svhn_ood():
    """
    Betölti a SVHN (Street View House Numbers) (Out-of-Distribution) adathalmazt.
    A MATLAB formátum dimenzióit (32, 32, 3, N) át kell forgatni (N, 32, 32, 3)-ra, 
    hogy kompatibilis legyen a Keras CNN rétegekkel és a CIFAR-10-zel!
    """
    url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    path = tf.keras.utils.get_file('svhn_test_32x32.mat', origin=url)
    mat = sio.loadmat(path)
    x_test = mat['X']  # (32, 32, 3, N)
    y_test = mat['y']  # Címkék

    # Átforgatjuk a dimenziókat (N, 32, 32, 3)
    x_test = np.transpose(x_test, (3, 0, 1, 2)).astype('float32')
    return x_test, y_test
    

if __name__ == "__main__":
    print("⏳ Szürkeárnyalatos (MNIST) adatok ellenőrzése...")
    (x_train_mnist, _), (x_test_mnist, _) = load_mnist_id()
    x_test_fmnist, _ = load_fashion_mnist_ood()
    print(f"✅ MNIST (ID) shape: {x_train_mnist.shape}")
    
    print("\n⏳ Színes (CIFAR/SVHN) adatok ellenőrzése (Letöltés indulhat)...")
    (x_train_cifar, _), (x_test_cifar, _) = load_cifar10_id()
    x_test_svhn, _ = load_svhn_ood()
    
    print(f"✅ CIFAR-10 Train (ID) shape: {x_train_cifar.shape}")
    print(f"✅ CIFAR-10 Test (ID) shape:  {x_test_cifar.shape}")
    print(f"✅ SVHN Test (OOD) shape:     {x_test_svhn.shape}\n")

    print("🎨 Vizuális ellenőrzés (Plot) indítása az RGB adatokra...")
    # A korábbi plot_id_vs_ood_batch függvény hívása, ami a színes képeket is megeszi 
    # (a squeeze() színesnél nem okoz gondot, mert a csatorna dimenziója 3, nem 1)
    plot_id_vs_ood_batch(x_test_cifar, x_test_svhn, num_samples=5)