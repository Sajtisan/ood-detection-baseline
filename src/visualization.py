import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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

def plot_roc_curves(results_dict, title="ROC Görbék", save_name="roc_curve", save_dir="results/plots"):
    """
    Több modell vagy paraméter (pl. Temperature) ROC görbéjének közös ábrázolása.
    
    Bemenet:
    results_dict: Szótár, ahol a kulcs a név (pl. "ResNet", "T=5"), 
                  az érték pedig egy tuple: (y_true_címkék, msp_pontszámok)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.figure(figsize=(8, 6))
    
    # Görbék kirajzolása a szótár alapján
    for label_name, (y_true, y_scores) in results_dict.items():
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{label_name} (AUC = {roc_auc:.4f})')

    # Véletlen tippelés (Random Guess) átlója
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Véletlen (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.7)
    
    save_path = os.path.join(save_dir, f"{save_name}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"ROC görbe mentve: {save_path}")
    plt.close()

def plot_pr_curves(results_dict, title="Precision-Recall Görbék", save_name="pr_curve", save_dir="results/plots"):
    """
    Több modell vagy paraméter Precision-Recall (PR) görbéjének közös ábrázolása.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.figure(figsize=(8, 6))
    
    for label_name, (y_true, y_scores) in results_dict.items():
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        plt.plot(recall, precision, lw=2, label=f'{label_name} (AUC = {pr_auc:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Visszahívás)', fontsize=12)
    plt.ylabel('Precision (Precízió)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=':', alpha=0.7)
    
    save_path = os.path.join(save_dir, f"{save_name}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"PR görbe mentve: {save_path}")
    plt.close()

# --- HASZNÁLATI PÉLDA (MOCK ADATOKKAL) ---
if __name__ == "__main__":
    print("⏳ Görbék tesztelése mock adatokkal...")
    
    # 1. Címkék generálása: Tegyük fel, van 1000 ID adatunk (1) és 1000 OOD adatunk (0)
    y_true = np.concatenate([np.ones(1000), np.zeros(1000)])
    
    # 2. MSP értékek szimulálása (Valós projektben ezeket a lementett .npy fájlokból töltöd be!)
    # ID esetén magasabb, OOD esetén alacsonyabb MSP-ket generálunk
    msp_mlp = np.concatenate([np.random.normal(0.8, 0.1, 1000), np.random.normal(0.5, 0.2, 1000)])
    msp_cnn = np.concatenate([np.random.normal(0.85, 0.1, 1000), np.random.normal(0.4, 0.15, 1000)])
    msp_resnet = np.concatenate([np.random.normal(0.9, 0.05, 1000), np.random.normal(0.3, 0.1, 1000)])
    
    # Fontos: A valószínűségeket 0 és 1 közé vágjuk
    msp_mlp = np.clip(msp_mlp, 0, 1)
    msp_cnn = np.clip(msp_cnn, 0, 1)
    msp_resnet = np.clip(msp_resnet, 0, 1)

    # 3. Modellek összehasonlítása (ROC és PR)
    model_comparison_dict = {
        "MLP Baseline": (y_true, msp_mlp),
        "Standard CNN": (y_true, msp_cnn),
        "Complex ResNet": (y_true, msp_resnet)
    }
    
    plot_roc_curves(model_comparison_dict, title="ROC: Modellek összehasonlítása", save_name="roc_models_compare")
    plot_pr_curves(model_comparison_dict, title="PR: Modellek összehasonlítása", save_name="pr_models_compare")

    # 4. Temperature Scaling hatásának bemutatása egyetlen modellen (pl. ResNet)
    # Szimuláljuk, hogy a T növelésével kicsit "simul" az eloszlás, ami ronthatja vagy javíthatja a detekciót
    msp_resnet_t5 = np.clip(msp_resnet - np.random.normal(0.05, 0.05, 2000), 0, 1)
    msp_resnet_t10 = np.clip(msp_resnet - np.random.normal(0.1, 0.05, 2000), 0, 1)

    temp_comparison_dict = {
        "ResNet (T=1)": (y_true, msp_resnet),
        "ResNet (T=5)": (y_true, msp_resnet_t5),
        "ResNet (T=10)": (y_true, msp_resnet_t10)
    }
    
    plot_roc_curves(temp_comparison_dict, title="ROC: Temperature Scaling (ResNet)", save_name="roc_resnet_temperature")

# --- VIZUALIZÁCIÓS TESZT ÉS GENERÁLÓ BLOKK ---
if __name__ == "__main__":
    print("⏳ Átfedő hisztogramok generálásának indítása...\n")
    
    # 1. SZIMULÁCIÓ: Baseline (Tiszta adatok, T=1)
    # A modell nagyon magabiztos az ID adatokon (0.8-0.9 körül), 
    # és bizonytalan az OOD adatokon (0.3-0.5 körül)
    baseline_id = np.random.normal(loc=0.85, scale=0.1, size=5000)
    baseline_id = np.clip(baseline_id, 0.1, 1.0)
    
    baseline_ood = np.random.normal(loc=0.35, scale=0.15, size=5000)
    baseline_ood = np.clip(baseline_ood, 0.1, 1.0)
    
    plot_msp_distributions(baseline_id, baseline_ood, experiment_name="Baseline (Tiszta ID)")
    
    # 2. SZIMULÁCIÓ: Torzított kísérlet (pl. 5-ös szintű Só-bors zaj)
    # A zaj miatt az ID adatokon a modell magabiztossága összeomlik, 
    # és teljesen összefolyik az OOD eloszlással.
    noisy_id = np.random.normal(loc=0.45, scale=0.2, size=5000)
    noisy_id = np.clip(noisy_id, 0.1, 1.0)
    
    noisy_ood = np.random.normal(loc=0.40, scale=0.18, size=5000)
    noisy_ood = np.clip(noisy_ood, 0.1, 1.0)
    
    plot_msp_distributions(noisy_id, noisy_ood, experiment_name="Noisy S&P (Szint 5)")
    
    print("\n✅ Generálás kész! Ellenőrizd a 'results/plots' mappát!")
