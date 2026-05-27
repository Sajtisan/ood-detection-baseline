import numpy as np
import tensorflow as tf

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

    MIÉRT VAN ERRE SZÜKSÉG?
    -------------------------------------------------
    1. Memóriakezelés (OOM elkerülése): 
       A teszthalmazok (pl. 10 000 - 26 000 kép) nem férnek be egyszerre a GPU 
       memóriájába. Ez a függvény felel azért, hogy a tf.data.Dataset batch-eken 
       (kötegeken) iteráljon végig memóriabiztosan.
       
    2. Hőmérséklet-söprés (Temperature Sweeping): 
       A kutatás megköveteli az optimális 'T' érték megkeresését. Ahelyett, hogy 
       kézzel hatszor lefuttatnánk a teljes Inferencia szkriptet, ez a függvény 
       egyetlen futtatás alatt kigenerálja és letárolja az eredményeket az 
       összes megadott T értékre.
       
    3. Híd a Milestone 4 felé (Scikit-learn kompatibilitás): 
       A kimenetnek egybefüggő NumPy tömböknek kell lenniük, amiket a következő 
       fázisban egy-az-egyben be tudunk dobni a ROC/PR görbe rajzoló függvényekbe.

    TECHNIKAI ELVÁRÁSOK AZ IMPLEMENTÁCIÓHOZ:
    -------------------------------------------------------------
    - Hozzatok létre egy eredmény-szótárat (dict), ahol a kulcs a T értéke, 
      az érték pedig egy üres lista a batch-ek eredményeinek.
    - Egy külső ciklus menjen végig a 'temperatures' listán.
    - Minden T esetén példányosítsátok a fenti 'TemperatureScaledInference' 
      osztályt a 'model' és az adott 'T' átadásával.
    - Egy belső ciklus menjen végig a 'dataset'-en (for batch in dataset: ...).
      (Figyelem: Ha a dataset ad címkéket is (x, y), azt megfelelően csomagoljátok ki!)
    - Hívjátok meg a wrapper.predict_msp(x) függvényt a batch-re, és fűzzétek 
      hozzá a listához.
    - A végén az 'np.concatenate' segítségével olvasszátok össze a batch-listákat 
      egyetlen nagy 1D-s numpy tömbbé (T értékenként).

    VISSZATÉRÉSI ÉRTÉK (Elvárt Output):
    -----------------------------------
    Egy dictionary, ahol a kulcs a hőmérséklet (int/float), az érték az MSP tömb (np.array).
    Példa: { 1: [0.99, 0.82, ...], 2: [0.71, 0.60, ...] }
    """
    # TODO: A fenti dokumentáció és iránymutatás alapján implementáljátok az adatfeldolgozó logikát!
    pass

if __name__ == "__main__":
    print("⏳ Temperature Scaling Unit Test indítása...\n")
    
    # 1. Létrehozunk egy Mock (ál) modellt, ami fix logitokat ad vissza
    class DummyModel:
        def predict(self, x, verbose=0):
            # Szimulálunk egy olyan hálózatot, ami nagyon biztos a dolgában 
            # (az első osztály kiugróan magas: 5.0)
            return np.array([[5.0, 1.0, 0.2]])

    dummy_net = DummyModel()
    dummy_input = np.array([0]) # Csak egy helykitöltő bemenet
    
    # 2. T=1 Teszt (Eredeti hálózat)
    wrapper_t1 = TemperatureScaledInference(dummy_net, temperature=1.0)
    probs_t1 = wrapper_t1.predict_probabilities(dummy_input)
    msp_t1 = wrapper_t1.predict_msp(dummy_input)
    
    print("--- Teszt 1: T=1.0 (Alapértelmezett kimenet) ---")
    print(f"Valószínűségek: {probs_t1}")
    print(f"Magabiztosság (MSP): {msp_t1[0]:.4f} (Elvárt: ~0.98)\n")
    
    # 3. T>1 Teszt (Simított eloszlás)
    temperatures_to_test = [2, 5, 10, 50, 100]
    
    print("--- Teszt 2: T > 1.0 (Eloszlás laposodásának ellenőrzése) ---")
    for t in temperatures_to_test:
        wrapper = TemperatureScaledInference(dummy_net, temperature=t)
        probs = wrapper.predict_probabilities(dummy_input)
        msp = wrapper.predict_msp(dummy_input)
        
        print(f"T={t:<3} | Valószínűségek: {np.round(probs, 3)} | MSP: {msp[0]:.4f}")
        
        # Szigorú szoftvermérnöki ellenőrzés (Assert)
        assert msp[0] < msp_t1[0], f"Hiba! A T={t} MSP nem csökkent!"

    print("\n✅ MINDEN UNIT TEST SIKERESEN LEFUTOTT!")
    print("A T=1 megegyezik az alap modellel, T>1 esetén az eloszlás egyre közelebbít az egyenleteshez (laposodik).")