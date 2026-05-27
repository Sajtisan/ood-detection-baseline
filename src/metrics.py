import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def _get_binary_labels(id_msp, ood_msp):
    """
    Segédfüggvény a bináris címkék legenerálásához.
    In-Distribution (ID) = 1 (Pozitív osztály)
    Out-of-Distribution (OOD) = 0 (Negatív osztály)
    """
    y_true = np.concatenate([np.ones_like(id_msp), np.zeros_like(ood_msp)])
    y_scores = np.concatenate([id_msp, ood_msp])
    return y_true, y_scores


def calculate_auroc(id_msp, ood_msp):
    """
    Kiszámolja az AUROC (Area Under the Receiver Operating Characteristic) metrikát.
    Minél közelebb van az 1.0-hoz, annál jobban el tudja különíteni a modell az ID és OOD adatokat.
    """
    y_true, y_scores = _get_binary_labels(id_msp, ood_msp)
    return roc_auc_score(y_true, y_scores)


def calculate_aupr(id_msp, ood_msp):
    """
    Kiszámítja mindkét AUPR (Area Under the Precision-Recall curve) értéket.
    Visszatérési érték: (aupr_in, aupr_out)
    """
    y_true, y_scores = _get_binary_labels(id_msp, ood_msp)
    
    # 1. AUPR-In: Az ID adatokat tekintjük a pozitív osztálynak (1)
    aupr_in = average_precision_score(y_true, y_scores)
    
    # 2. AUPR-Out: Az OOD adatokat tekintjük pozitív osztálynak (1)
    # Ehhez meg kell fordítanunk a címkéket, ÉS a magabiztossági pontszámokat is, 
    # hiszen az OOD képeknél az ALACSONY MSP a jó indikátor!
    y_true_out = 1 - y_true
    y_scores_out = -y_scores 
    aupr_out = average_precision_score(y_true_out, y_scores_out)
    
    return aupr_in, aupr_out


def calculate_fpr_at_95_tpr(id_msp, ood_msp):
    """
    FPR at 95% TPR. Megmutatja, hogy mekkora a False Positive Rate (OOD képek aránya,
    amiket tévesen elfogadott a hálózat), ha a True Positive Rate (ID képek felismerése)
    szigorúan 95%-ra van beállítva. (Minél KISEBB, annál jobb!)
    """
    y_true, y_scores = _get_binary_labels(id_msp, ood_msp)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Megkeressük azt az indexet, ahol a TPR eléri (vagy átlépi) a 95%-ot
    idx = np.where(tpr >= 0.95)[0][0]
    
    return fpr[idx]

def generate_evaluation_report(id_npy_path, ood_npy_path):
    """
    Betölti a kimentett MSP vektorokat (.npy fájlokból), kiszámolja rájuk a 
    küszöbérték-független metrikákat, és generál egy könnyen olvasható riportot.

    MIÉRT VAN ERRE SZÜKSÉG? (Magyarázat a csapatnak)
    -------------------------------------------------
    1. A Hálózat és a Matek szétválasztása (Decoupling): 
       A neurális hálózatok futtatása (Inferencia) órákig is eltarthat és drága 
       GPU-t igényel. A metrikák számolása viszont csak egyszerű CPU-s matek, 
       ami tizedmásodpercek alatt lefut. Mivel az inferencia kimenetét (.npy) 
       lementettétek a lemezre, mostantól bármikor újraszámolhatjátok a metrikákat 
       GPU nélkül is, ha esetleg elrontottatok valamit a képletben!
       
    2. Publikációra kész eredmények: 
       A végső vizsgán vagy egy kutatási cikkben nem mutathatunk nyers NumPy 
       tömböket. Ez a függvény a felelős azért, hogy a nyers matematikát "emberi 
       nyelvre", szép százalékos értékekre fordítsa le.

    TECHNIKAI ELVÁRÁSOK AZ IMPLEMENTÁCIÓHOZ (Hogyan csináljátok):
    -------------------------------------------------------------
    1. Használjátok a 'numpy.load(filepath)' függvényt mindkét elérési útra, 
       hogy betöltsétek a memóriába az ID és OOD MSP tömböket.
    2. Hívjátok meg rájuk a fenti 3 matematikai függvényt (calculate_auroc, 
       calculate_aupr, calculate_fpr_at_95_tpr). Figyelem: az AUPR két értéket ad vissza!
    3. Használjatok formázott printelés (f-string) a konzolos megjelenítéshez!
       (Trükk: Érdemes a kapott értékeket 100-zal beszorozni és 2 tizedesjegyre 
       kerekíteni, pl. 0.9523 -> "95.23%", mert a szakirodalomban így szokás megadni).
    
    BÓNUSZ FELADAT (Opcionális):
    Ha akarjátok, a függvény térjen vissza egy dictionary-vel is (pl. 
    {'AUROC': 95.2, 'FPR95': 12.4}), hogy később esetleg egy Pandas DataFrame-be 
    tudjuk menteni a kísérleteket!
    """
    # TODO: A fenti dokumentáció alapján írjátok meg a betöltő és riportáló logikát!
    pass

if __name__ == "__main__":
    print("⏳ Metrikák matematikai ellenőrzése (Unit Test)...\n")
    
    # Szimulálunk egy "TÖKÉLETES" modellt, ami sosem hibázik:
    # Az ID képekre mindig magas (0.8-0.9) MSP-t ad, az OOD-re mindig alacsonyt (0.1-0.2).
    perfect_id_msp = np.array([0.9, 0.85, 0.95, 0.88, 0.99])
    perfect_ood_msp = np.array([0.1, 0.2, 0.15, 0.05, 0.25])
    
    print("--- 1. TÖKÉLETES MODELL TESZTJE ---")
    auroc = calculate_auroc(perfect_id_msp, perfect_ood_msp)
    aupr_in, aupr_out = calculate_aupr(perfect_id_msp, perfect_ood_msp)
    fpr95 = calculate_fpr_at_95_tpr(perfect_id_msp, perfect_ood_msp)
    
    print(f"AUROC:     {auroc:.4f} (Elvárt: 1.0000)")
    print(f"AUPR-In:   {aupr_in:.4f} (Elvárt: 1.0000)")
    print(f"AUPR-Out:  {aupr_out:.4f} (Elvárt: 1.0000)")
    print(f"FPR@95TPR: {fpr95:.4f} (Elvárt: 0.0000)\n")
    
    # Szigorú szoftvermérnöki ellenőrzés
    assert auroc == 1.0, "Hiba az AUROC számításban!"
    assert fpr95 == 0.0, "Hiba az FPR@95TPR számításban!"