import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_noise(images, severity=1):
    """
    Gauss-zaj (normál eloszlású zaj) hozzáadása.
    Támogat egyetlen képet (H, W, C), vagy egy teljes köteget is (Batch, H, W, C).
    """
    # 1-es szint: 0.04-es szórás, 10-es szint: 0.4-es szórás
    std = severity * 0.04
    
    # A size=images.shape miatt dinamikusan alkalmazkodik a batch méretéhez!
    noise = np.random.normal(loc=0.0, scale=std, size=images.shape)

    # np.clip garantálja, hogy a zaj hozzáadása után se lépjük túl a [0, 1] értékhatárokat
    return np.clip(images + noise, 0.0, 1.0)

def apply_uniform_noise(images, severity=1):
    limit = severity * 0.05
    # 2. Egyenletes zaj generálása a megfelelő intervallumon és dimenziókkal
    noise = np.random.uniform(low=-limit, high=limit, size=images.shape)
    # 3. Zaj hozzáadása a képhez és az értékek vágása (clipping) a [0.0, 1.0] tartományra
    noisy_images = np.clip(images + noise, 0.0, 1.0)
    return noisy_images

def apply_salt_and_pepper_noise(images, severity=1):
    # sérült pixelek arányának meghatározása
    amount = severity * 0.03
    
    # másolat készítése, hogy az eredeti képeket ne módosítsuk
    out = np.copy(images)
    
    # véletlenszerű mátrix generálása
    # np.random.rand elvárja a dimenziókat argumentumként, ezért kicsomagoljuk a shape-et (*images.shape)
    random_matrix = np.random.rand(*images.shape)
    
    # maszkolás (Só)
    out[random_matrix < (amount / 2.0)] = 1.0
    
    # maszkolás (Bors)
    # logikai ÉS (&) operátorral kombináljuk a két feltételt
    pepper_mask = (random_matrix >= (amount / 2.0)) & (random_matrix < amount)
    out[pepper_mask] = 0.0

    return out

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
    from preprocessing import min_max_scaler
    
    print("⏳ Összes zajgenerátor vizuális tesztelése...")
    
    # 1. Adat betöltése és skálázása [0, 1] közé
    (_, _), (x_test, _) = load_mnist_id()
    test_image = min_max_scaler(x_test[0:1]) # Csak az első képet vesszük ki (batch_size=1)
    
    # 2. Plot felépítése (3 sor, 3 oszlop)
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    fig.suptitle("Kép Torzítási Szintek (1-10 skálán)", fontsize=16)
    
    noise_types = [('Gauss', 'gaussian'), ('Egyenletes', 'uniform'), ('Só-Bors', 'salt_and_pepper')]
    
    for row_idx, (display_name, n_type) in enumerate(noise_types):
        # Tiszta kép
        axes[row_idx, 0].imshow(test_image.squeeze(), cmap='gray')
        axes[row_idx, 0].set_title(f"Tiszta Kép")
        axes[row_idx, 0].axis('off')
        
        # 1-es szint
        noisy_1 = apply_noise(test_image, noise_type=n_type, severity=1)
        axes[row_idx, 1].imshow(noisy_1.squeeze(), cmap='gray')
        axes[row_idx, 1].set_title(f"{display_name} (Szint: 1)")
        axes[row_idx, 1].axis('off')
        
        # 10-es szint
        noisy_10 = apply_noise(test_image, noise_type=n_type, severity=10)
        axes[row_idx, 2].imshow(noisy_10.squeeze(), cmap='gray')
        axes[row_idx, 2].set_title(f"{display_name} (Szint: 10)")
        axes[row_idx, 2].axis('off')

    plt.tight_layout()
    plt.show()
