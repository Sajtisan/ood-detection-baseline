import numpy as np
import tensorflow as tf
import os

class TemperatureScaledInference:
    def __init__(self, model, temperature=1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0.")
        self.model = model
        self.temperature = temperature
    
    def predict_probabilities(self, x):
        """
        1. lekéri a logitokat
        2. skálázza T-vel
        3. softmaxolja
        """
        logits = self.model.predict(x, verbose=0)
        scaled_logits = logits / self.temperature
        probabilities = tf.nn.softmax(scaled_logits, axis=-1).numpy()
        return probabilities
    
    def predict_msp(self, x):
        """
        Közvetlenül a Maximum Softmax Probability-t adja vissza, ami a legnagyobb valószínűségi érték a softmax kimenetből.
        """
        probabilities = self.predict_probabilities(x)
        return np.max(probabilities, axis=-1)

def extract_msp_for_dataset(model, dataset, temperatures=[1, 2, 5, 10, 50, 100]):
    """
    Kinyeri a Maximum Softmax Probability (MSP) értékeket egy teljes adathalmazra,
    több különböző hőmérséklet (Temperature) beállítás mellett.
    """
    # Hozzatok létre egy eredmény-szótárat...
    msp_results = {t: [] for t in temperatures}
    
    # Egy külső ciklus menjen végig a 'temperatures' listán.
    for t in temperatures:
        # Minden T esetén példányosítsátok a fenti 'TemperatureScaledInference' osztályt...
        wrapper = TemperatureScaledInference(model, temperature=t)
        
        # Egy belső ciklus menjen végig a 'dataset'-en...
        for batch in dataset:
            # (Figyelem: Ha a dataset ad címkéket is (x, y), azt megfelelően csomagoljátok ki!)
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x = batch[0]
            else:
                x = batch
            
            # Hívjátok meg a wrapper.predict_msp(x) függvényt a batch-re, és fűzzétek hozzá...
            msp = wrapper.predict_msp(x)
            msp_results[t].append(msp)
            
    # A végén az 'np.concatenate' segítségével olvasszátok össze a batch-listákat...
    for t in temperatures:
        msp_results[t] = np.concatenate(msp_results[t], axis=0)
        
    return msp_results

def run_full_inference_pipeline(model_path, x_id, x_ood, apply_noise_fn, save_dir="msp_results", temperatures=[1, 2, 5, 10, 50, 100]):
    """
    Végrehajtja a teljes Milestone 3 inferenciát és elmenti az eredményeket.
    """
    # Mappa létrehozása a kimentéshez
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Modell betöltése
    print(f"📦 Modell betöltése: {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # (Megjegyzés: Az extract_msp_for_dataset dataset iterátort vár. 
    # Ha x_id egyetlen nagy numpy tömb, berakjuk egy listába [x_id], hogy a belső for batch in dataset: működjön.
    # Ha tf.data.Dataset, akkor a szögletes zárójelek nem kellenek).
    dataset_id = [x_id] if isinstance(x_id, np.ndarray) else x_id
    dataset_ood = [x_ood] if isinstance(x_ood, np.ndarray) else x_ood

    # 2. Tiszta ID adatok inferenciája
    print("🟢 Tiszta ID adatok feldolgozása...")
    id_msp_dict = extract_msp_for_dataset(model, dataset_id, temperatures)
    for t, msp_array in id_msp_dict.items():
        np.save(os.path.join(save_dir, f"msp_id_clean_T{t}.npy"), msp_array)
        
    # 2/b. Tiszta OOD adatok inferenciája
    print("🔴 Tiszta OOD adatok feldolgozása...")
    ood_msp_dict = extract_msp_for_dataset(model, dataset_ood, temperatures)
    for t, msp_array in ood_msp_dict.items():
        np.save(os.path.join(save_dir, f"msp_ood_clean_T{t}.npy"), msp_array)

    # 3. Zajjal torzított adatok inferenciája
    print("🌪️ Zajjal torzított adatok feldolgozása (1-10 szintek)...")
    noise_types = ['gaussian', 'uniform', 'salt_and_pepper']
    
    for noise in noise_types:
        for severity in range(1, 11):
            print(f"   -> Zaj: {noise}, Szint: {severity}")
            
            # Kép torzítása
            x_noisy = apply_noise_fn(x_id, noise_type=noise, severity=severity)
            dataset_noisy = [x_noisy] if isinstance(x_noisy, np.ndarray) else x_noisy
            
            # MSP kinyerése
            noisy_msp_dict = extract_msp_for_dataset(model, dataset_noisy, temperatures)
            
            # Kimentés
            for t, msp_array in noisy_msp_dict.items():
                filename = f"msp_id_{noise}_sev{severity}_T{t}.npy"
                np.save(os.path.join(save_dir, filename), msp_array)
                
    print(f"\n✅ MINDEN INFERENCIA KÉSZ! Fájlok mentve ide: {save_dir}/")


def run_unit_test():
    print("⏳ Temperature Scaling Unit Test indítása...\n")
    class DummyModel:
        def predict(self, x, verbose=0):
            return np.array([[5.0, 1.0, 0.2]])

    dummy_net = DummyModel()
    dummy_input = np.array([0]) 
    
    wrapper_t1 = TemperatureScaledInference(dummy_net, temperature=1.0)
    probs_t1 = wrapper_t1.predict_probabilities(dummy_input)
    msp_t1 = wrapper_t1.predict_msp(dummy_input)
    
    print("--- Teszt 1: T=1.0 (Alapértelmezett kimenet) ---")
    print(f"Valószínűségek: {probs_t1}")
    print(f"Magabiztosság (MSP): {msp_t1[0]:.4f} (Elvárt: ~0.98)\n")
    
    temperatures_to_test = [2, 5, 10, 50, 100]
    print("--- Teszt 2: T > 1.0 (Eloszlás laposodásának ellenőrzése) ---")
    for t in temperatures_to_test:
        wrapper = TemperatureScaledInference(dummy_net, temperature=t)
        probs = wrapper.predict_probabilities(dummy_input)
        msp = wrapper.predict_msp(dummy_input)
        
        print(f"T={t:<3} | Valószínűségek: {np.round(probs, 3)} | MSP: {msp[0]:.4f}")
        assert msp[0] < msp_t1[0], f"Hiba! A T={t} MSP nem csökkent!"

    print("\n✅ MINDEN UNIT TEST SIKERESEN LEFUTOTT!")


if __name__ == "__main__":
    import os
    print("\n==========================================================")
    print("ÉLES INFERENCIA PIPELINE INDÍTÁSA")
    print("==========================================================\n")
    
    # 1. VALÓS ADATOK ÉS FÜGGVÉNYEK BEIMPORTÁLÁSA
    from data_loader import load_cifar10_id, load_svhn_ood 
    from noise_generator import apply_noise 
    from preprocessing import min_max_scaler # KÖTELEZŐ A NORMALIZÁLÁSHOZ!
    
    # 2. ADATOK BETÖLTÉSE ÉS NORMALIZÁLÁSA
    print("Képek betöltése a memóriába...")
    (_, _), (x_id_test, _) = load_cifar10_id() # Tiszta ID teszthalmaz (32x32x3)
    (_, _), (x_ood_test, _) = load_svhn_ood()  # Tiszta OOD teszthalmaz (32x32x3)
    
    print("Képek normalizálása [0.0, 1.0] tartományba...")
    x_id_test = min_max_scaler(x_id_test)
    x_ood_test = min_max_scaler(x_ood_test)
    
    # 3. MODELL ÚTVONALA
    # Ha a projekt gyökeréből futtatod (python src/inference.py), akkor a 
    # 'saved_models' mappa a gyökérben lesz a run_training.py lefutása után.
    # FIGYELEM: Mivel CIFAR-10 és SVHN adatot töltöttél be (32x32x3), 
    # SZIGORÚAN a CNN vagy ResNet modellt kell betöltened, az MLP (28x28) itt elszállna!
    modell_utvonal = "saved_models/cnn_cifar.keras" 
    
    # Biztonsági ellenőrzés
    if not os.path.exists(modell_utvonal):
        print(f"\n❌ HIBA: Nem található betanított modell ezen az útvonalon: {modell_utvonal}")
        print("💡 Megoldás: Először futtasd le a 'python src/run_training.py' parancsot, hogy a hálózat betanuljon!")
        exit(1)
        
    # 4. PIPELINE INDÍTÁSA
    run_full_inference_pipeline(
        model_path=modell_utvonal,
        x_id=x_id_test,
        x_ood=x_ood_test,
        apply_noise_fn=apply_noise,
        save_dir="results/msp_outputs", # A projekt gyökerében lévő results mappába ment
        temperatures=[1, 2, 5, 10, 50, 100]
    )