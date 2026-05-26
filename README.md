# OOD Detection Baseline

Ez a projekt a *Maximum Softmax Probability (MSP)* alapú Out-of-Distribution (OOD) detekciót vizsgálja mély neurális hálózatokban. A kutatás és az implementáció [Hendrycks és Gimpel (2017): "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks"](https://arxiv.org/abs/1610.02136) című publikációjára épül.

Kisebb összegzés [itt](https://uszeged-my.sharepoint.com/:w:/g/personal/raffai_sajti_david_o365_u-szeged_hu/IQC_Wtt1AVSOTYq80VEYxNFrAYHtNocwOjUJq1_ootIhyao?e=MdigGc)

A projekt az SZTE TTIK kurzusának keretein belül, egy 3 fős csapatmunkában készül.

##  A Kísérleti Mátrix (3x3x3)

A robusztusság és a kalibráció vizsgálata az alábbi struktúra mentén történik:

**1. Adatbázisok (In-Distribution $\rightarrow$ Out-of-Distribution)**
- MNIST $\rightarrow$ Fashion-MNIST
- CIFAR-10 $\rightarrow$ SVHN (Street View House Numbers)
- CIFAR-10 $\rightarrow$ Torzított (zajos) CIFAR-10

**2. Modelltípusok**
- Baseline MLP (Multilayer Perceptron)
- Standard CNN (Convolutional Neural Network)
- Complex CNN (ResNet blokk / Wide ResNet fragmentum)

**3. Vizsgált Paraméterek**
- **Zaj injektálás:** Bemeneti torzítások (Gauss, egyenletes, só-bors) vizsgálata 1-10 skálán.
- **Temperature Scaling ($T$):** A softmax eloszlás simításának hatása az OOD detekcióra.
- **Detekciós Metrikák:** FPR at 95% TPR, AUROC, és AUPR (In/Out) a klasszikus *accuracy* helyett.

---

## Környezet Beállítása (Setup)

A kód hardver-agnosztikus, futtatható lokális CPU-n, dedikált GPU-n (pl. RTX 5070) és Google Colab környezetben is. A csomagütközések elkerülése végett szigorúan az alábbi lépéseket kövesd a telepítésnél.

### 1. Repository klónozása
Nyiss egy terminált, és klónozd le a projektet:
```bash
git clone https://github.com/Sajtisan/ood-detection-baseline.git
cd ood-detection-baseline
```

### 2. Virtuális környezet létrehozása
Érdemes egy dedikált környezetet létrehozni, hogy a csomagok ne keveredjenek a rendszered többi Python csomagjával.
```bash
python -m venv venv

# Aktiválás Windowson:
venv\Scripts\activate
```

### 3. Függőségek telepítése
A `.gitignore` és a `requirements.txt` már be van állítva a projektben. Telepítsd a szükséges csomagokat:
```bash
pip install -r requirements.txt
```

## Futtatás (TBI)
A scriptek (Data Pipeline, Training, Inference) dokumentációja a Milestone-ok előrehaladtával ide fog felkerülni.
- Adatok letöltése: `[HAMAROSAN]`
- Modellek tanítása: `[HAMAROSAN]`
- Inferencia és Kiértékelés: `[HAMAROSAN]`