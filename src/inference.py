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
    Végigmegy egy adathalmazon, és különböző T értékekkel kinyeri az MSP-ket.
    (Implementációra vár a csapat által).
    """
    # TODO: A fenti TemperatureScaledInference osztály felhasználásával 
    # kinyerni az MSP vektorokat a 'dataset' képeiből.
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