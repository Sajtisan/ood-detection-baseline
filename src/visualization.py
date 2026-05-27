import os
import matplotlib.pyplot as plt

def plot_training_history(history, model_name, save_dir="results/plots"):
    """
    Kirajzolja és PNG formátumban elmenti a tanítási és validációs veszteség (loss) és pontosság (accuracy) görbéket.
    """
    # Mappa ellenőrzése és létrehozása, ha nem létezik
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Két részre osztott grafikon (1 sor, 2 oszlop)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Tanítási metrikák: {model_name}", fontsize=16)

    # Veszteség (Loss) görbe
    ax1.plot(history.history['loss'], label='Tanítási veszteség', color='blue', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validációs veszteség', color='red', linestyle='--', linewidth=2)
    ax1.set_title('Modell veszteség (Loss)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Veszteség')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Accuracy görbe
    ax2.plot(history.history['accuracy'], label='Tanítási pontosság', color='green', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validációs pontosság', color='orange', linestyle='--', linewidth=2)
    ax2.set_title('Modell pontosság (Accuracy)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Pontosság')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    # Kép  mentése és lezárása
    save_path = os.path.join(save_dir, f"{model_name}_history.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Grafikon mentve: {save_path}")
    plt.close()

def plot_msp_distributions(id_msp, ood_msp, experiment_name="Baseline", save_dir="results/plots"):
    # mappa ellenőrzése és létrehozása, ha nem létezik
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.figure(figsize=(10, 6))
    
    # közös bin (kosár) beállítások a szép átfedéshez [0.0, 1.0] tartományban
    bins = np.linspace(0.0, 1.0, 50)
    
    # ID eloszlás (Kék színnel)
    plt.hist(id_msp, bins=bins, alpha=0.6, color='blue', density=True, 
             label='In-Distribution (ID)', edgecolor='black', linewidth=0.5)
             
    # OOD eloszlás (Piros színnel)
    plt.hist(ood_msp, bins=bins, alpha=0.6, color='red', density=True, 
             label='Out-of-Distribution (OOD)', edgecolor='black', linewidth=0.5)

    # grafikon formázása
    plt.title(f"MSP Eloszlás: {experiment_name}", fontsize=16)
    plt.xlabel('Maximum Softmax Probability (Konfidencia)', fontsize=12)
    plt.ylabel('Sűrűség (Density)', fontsize=12)
    plt.xlim(0.0, 1.0)
    
    # fontos referenciapont: a véletlen tippelés (Random Guess) vonala
    # CIFAR-10 esetén (10 osztály) a legkisebb lehetséges max softmax érték 0.1
    plt.axvline(x=0.1, color='gray', linestyle='--', label='Random Guess (1/10)')
    
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)

    # fájlnév normalizálása (szóközök alulvonásra cserélése)
    safe_name = experiment_name.replace(" ", "_").replace("=", "")
    save_path = os.path.join(save_dir, f"msp_dist_{safe_name}.png")
    
    # kép mentése és lezárása
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"MSP eloszlás grafikon mentve: {save_path}")
    plt.close()
