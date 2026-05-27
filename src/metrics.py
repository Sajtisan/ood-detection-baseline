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
    # adatok betöltése
    id_msp = np.load(id_npy_path)
    ood_msp = np.load(ood_npy_path)
    
    # metrikák kiszámítása
    auroc = calculate_auroc(id_msp, ood_msp)
    aupr_in, aupr_out = calculate_aupr(id_msp, ood_msp)
    fpr95 = calculate_fpr_at_95_tpr(id_msp, ood_msp)
    
    # eredmények felszorzása 100-zal a százalékos kijelzéshez
    auroc_pct = auroc * 100
    aupr_in_pct = aupr_in * 100
    aupr_out_pct = aupr_out * 100
    fpr95_pct = fpr95 * 100
    
    # riport generálása és kiírása a konzolra
    print("==================================================")
    print("      OUT-OF-DISTRIBUTION DETECTION REPORT        ")
    print("==================================================")
    print(f"AUROC       (Magasabb a jobb): {auroc_pct:>6.2f}%")
    print(f"AUPR-In     (Magasabb a jobb): {aupr_in_pct:>6.2f}%")
    print(f"AUPR-Out    (Magasabb a jobb): {aupr_out_pct:>6.2f}%")
    print(f"FPR @ 95TPR (Alacsonyabb a jobb): {fpr95_pct:>6.2f}%")
    print("==================================================")
    
    # visszatérés a bónusz dictionary-vel
    results_dict = {
        'AUROC': auroc_pct,
        'AUPR-In': aupr_in_pct,
        'AUPR-Out': aupr_out_pct,
        'FPR95': fpr95_pct
    }
    
    return results_dict

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
