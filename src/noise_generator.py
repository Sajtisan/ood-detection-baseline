import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_noise(image, severity=1):
    """
    Gauss-zaj (normál  eloszlású zaj) hozzáadása.
    """
    # 1-es szint: 0.04-es szórás, 10-es szint: 0.4-es szórás
    std = severity * 0.04
    noise = np.random.normal(loc=0.0, scale=std, size=image.shape)

    # np.clip garantálja, hogy a zaj hozzáadása után se lépjük túl a [0, 1] értékhatárokat
    return np.clip(image + noise, 0.0, 1.0)

def apply_uniform_noise(images, severity=1):
    """
    Egyenletes (Uniform) zaj hozzáadása (Implementációra vár).
    
    Technikai követelmények:
    1. A 'severity' (1-10) alapján számold ki a limitet: limit = severity * 0.05
    2. Használd az np.random.uniform függvényt a [-limit, limit] intervallumon 
       az 'images.shape' dimenzióival.
    3. Add hozzá a képekhez, majd (KULCSFONTOSSÁGÚ) használd az np.clip-et, 
       hogy a pixelértékek [0.0, 1.0] között maradjanak!
    """
    # TODO: A fenti logika implementálása
    pass

def apply_salt_and_pepper_noise(images, severity=1):
    """
    Só-bors (Salt & Pepper) zaj hozzáadása (Implementációra vár).
    
    Technikai követelmények (For ciklus nélkül, mátrixmaszkolással!):
    1. A sérült pixelek aránya: amount = severity * 0.03
    2. Készíts egy másolatot a bemeneti képekről (out = np.copy(images)).
    3. Generálj egy véletlenszerű mátrixot (np.random.rand) az 'images.shape' dimenziókkal.
    4. Maszkolás: Ahol a random mátrix értéke < (amount / 2), ott állítsd a pixelt 1.0-ra (só).
    5. Maszkolás: Ahol a random mátrix értéke (amount / 2) és 'amount' között van, 
       ott állítsd 0.0-ra (bors).
    6. Itt nem kell np.clip, mert dedikáltan 0.0-t és 1.0-t adsz értékül.
    """
    # TODO: A fenti logika implementálása
    pass

def apply_noise(images, noise_type, severity=1):
    """
    Központi wrapper függvény a zajok egységes alkalmazásához.
    Támogatott noise_type: 'gaussian', 'uniform', 'salt_and_pepper'.
    """
    if noise_type == 'gaussian':
        return apply_gaussian_noise(images, severity)
    elif noise_type == 'uniform':
        return apply_uniform_noise(images, severity)
    elif noise_type == 'salt_and_pepper':
        return apply_salt_and_pepper_noise(images, severity)
    else:
        raise ValueError("Ismeretlen zajtípus! Válassz a következők közül: 'gaussian', 'uniform', 'salt_and_pepper'.")

# --- TESZTELÉSI ÉS VIZUALIZÁCIÓS BLOKK ---
if __name__ == "__main__":
    from data_loader import load_mnist_id
    from preprocessing import min_max_scale
    
    print("⏳ Bázis zajgenerátor (Gauss) vizuális tesztelése...")
    
    # 1. Adat betöltése és skálázása [0, 1] közé
    (_, _), (x_test, _) = load_mnist_id()
    test_image = min_max_scale(x_test[0:1]) # Csak az első képet vesszük ki (batch_size=1)
    
    # 2. Gauss-zaj tesztelése (mert csak ez van kész)
    noisy_gaussian_1 = apply_gaussian_noise(test_image, severity=1)
    noisy_gaussian_10 = apply_gaussian_noise(test_image, severity=10)
    
    # 3. Plot felépítése (1 sor, 3 oszlop)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig.suptitle("Kép Torzítási Szintek - Gauss Zaj", fontsize=14)
    
    axes[0].imshow(test_image.squeeze(), cmap='gray')
    axes[0].set_title("Tiszta Kép")
    axes[0].axis('off')
    
    axes[1].imshow(noisy_gaussian_1.squeeze(), cmap='gray')
    axes[1].set_title("Szint: 1 (Gyenge)")
    axes[1].axis('off')
    
    axes[2].imshow(noisy_gaussian_10.squeeze(), cmap='gray')
    axes[2].set_title("Szint: 10 (Erős)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()