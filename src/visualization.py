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