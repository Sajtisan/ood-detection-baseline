# Out-of-Distribution (OOD) Detekció Baseline és Robusztusság Vizsgálat

Ez a projekt egy átfogó keretrendszer neurális hálózatok (MLP, CNN, ResNet) megbízhatóságának és robusztusságának vizsgálatára. A gépi tanulási modellek gyakran indokolatlanul magabiztosak olyan adatokon is, amelyeket a tanítás során sosem láttak (Out-of-Distribution - OOD adatok). Célunk, hogy a hálózatok nyers kimeneteinek (logitok és softmax valószínűségek) elemzésével képesek legyünk elkülöníteni az ismert (In-Distribution) és az ismeretlen (OOD) adathalmazokat, továbbá megvizsgáljuk, hogyan hatnak a különböző képi zajok a detekciós rátára.

## Hivatkozott Publikációk és Források

A projekt implementációja és a kiértékelési metódusok az alábbi tudományos publikáción és szakmai cikkeken alapulnak:

1. **Dan Hendrycks, Kevin Gimpel (2016):** *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* - **[arXiv:1610.02136 (PDF letöltése)](https://arxiv.org/pdf/1610.02136)**
   > Ez a cikk vezette be a Maximum Softmax Probability (MSP) használatát alapértelmezett (baseline) metrikaként az OOD képek kiszűrésére. Megállapították, hogy a helyesen osztályozott ID adatokhoz magasabb MSP érték tartozik, mint az OOD adatokhoz.

2. **Medium / Analytics Vidhya:** *Out-of-Distribution Detection in Deep Neural Networks.*
   - **[Cikk elolvasása](https://medium.com/analytics-vidhya/out-of-distribution-detection-in-deep-neural-networks-450da9ed7044)**
   > Ez az átfogó cikk szolgáltatta az elméleti áttekintést és a gyakorlati megvalósítás (Temperature Scaling, eloszlások simítása) alapjait a projekt logikájának felépítéséhez.

## A Projekt Funkciói és Architektúrája

- **Modellek:** Egyedi implementációjú MLP, Standard CNN és Complex ResNet hálózatok.
- **Adathalmazok:** - *In-Distribution (ID):* MNIST, CIFAR-10.
  - *Out-of-Distribution (OOD):* Olyan adathalmazok, amelyeken a modellt nem tanítottuk (pl. SVHN, Fashion-MNIST).
- **Zajgenerátor (Robustness Testing):** 3 féle zajtípus (Gauss, Egyenletes, Só-Bors) 1-10-es intenzitási skálán paraméterezve a képek torzítására.
- **Temperature Scaling Wrapper:** A hálózati logitok skálázása a magabiztossági eloszlás optimalizálásához.
- **Metrikák:** Küszöbérték-független kiértékelés (AUROC, AUPR-In, AUPR-Out, FPR@95TPR).

## Telepítés és Futtatás

**1. Virtuális környezet létrehozása és aktiválása:**
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**2. függőségek telepítése:**
```bash
pip install -r requirements.txt
```

**3. Adatok elemzése és vizualizáció:**

A végső metrikák, a modell összeomlási pontjai (zaj robusztusság) és az optimális T paraméter elemzése a `final_analysis.ipynb` Jupyter Notebookban futtatható. A futtatás során az összesített eredménymátrix automatikusan kimentésre kerül a `results/metrics_summary.md` fájlba.

## Eredmények:

Az elemzések kimenetei, a generált hisztogramok és a metrikákat összefoglaló táblázatok a `results/` mappában találhatóak. A kutatás bebizonyította, hogy a megfelelő Temperature paraméter kiválasztása szignifikánsan növeli az AUROC értéket, míg a magasabb szintű (pl. 5+) képi zajok drasztikusan rontják a hálózatok detekciós megbízhatóságát, megközelítve a véletlen tippelés (50%) szintjét.
