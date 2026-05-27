import os
import random
import numpy as np
import tensorflow as tf

def set_global_seed(seed=42):
    """
    Rögzíti az összes létező véletlenszám-generátort a projektben, 
    hogy garantálja a kísérletek (tanítás, adat-keverés, zaj) reprodukálhatóságát.
    """
    # 1. Python beépített random modulja
    random.seed(seed)
    
    # 2. Hash seed rögzítése (a Python szótárak és set-ek sorrendjéhez)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy seed (Ez a legfontosabb a zaj-generátorok miatt a M3-ban!)
    np.random.seed(seed)
    
    # 4. TensorFlow globális seed (Súly-inicializáció, tf.data.Dataset shuffle)
    tf.random.set_seed(seed)
    
    # PRO TIPP GPU-hoz:
    # A modern GPU-k (mint az RTX 5070) a párhuzamosítás miatt alapból nem 
    # determinisztikusak a mátrixszorzásoknál. Ez a parancs rákényszeríti 
    # a TensorFlow-t, hogy GPU-n is szigorúan ugyanazt a matekot futtassa minden alkalommal.
    # Figyelem: minimálisan (1-2%-kal) lassíthatja a tanítást, de vizsgára kötelező!
    tf.config.experimental.enable_op_determinism()
    
    print(f"🌱 Globális seed beállítva: {seed} (Reprodukálhatóság garantálva!)")