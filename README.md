# VGG-16 Transfer Learning na CIFAR-10

> Porovnanie stratégií fine-tuningu predtrénovaného VGG-16 (ImageNet) na datasete CIFAR-10 s 10 triedami.

---

## Obsah

- [Prehľad projektu](#prehľad-projektu)
- [Požiadavky](#požiadavky)
- [Inštalácia](#inštalácia)
- [Štruktúra projektu](#štruktúra-projektu)
- [Reprodukcia kódu](#reprodukcia-kódu)
- [Experimenty](#experimenty)
- [Výsledky](#výsledky)
- [Ablation Study](#ablation-study)

---

## Prehľad projektu

Projekt skúma vplyv rôznych stratégií rozmrazovania vrstiev predtrénovaného VGG-16 pri prenosovom učení (transfer learning) na datasete **CIFAR-10**. Bolo spustených 5 hlavných experimentov (E1–E5) a 1 ablation štúdia (A1).

Architektúra: **VGG-16** predtrénovaný na ImageNet, klasifikačná hlava nahradená vrstvou pre 10 tried.

---

## Požiadavky

### Hardware
| Komponent | Odporúčané minimum |
|-----------|-------------------|
| GPU | NVIDIA GPU s CUDA podporou (testované na RTX 4060, 8.6 GB VRAM) |
| RAM | 8 GB+ |
| Disk | ~1 GB pre dataset CIFAR-10 + modely |

> Bez GPU sú niektoré experimenty (najmä E5 – Full fine-tuning) extrémne pomalé. E5 trvalo na RTX 4060 ~900 minút.

### Software
- Python **3.13** (testované) alebo 3.10+
- CUDA **12.4** (pre PyTorch 2.6.0+cu124)

### Python knižnice
```
torch>=2.6.0
torchvision>=0.21.0
numpy
pandas
matplotlib
seaborn
```

---

## Inštalácia

### 1. Klonovanie repozitára
```bash
git clone https://github.com/<tvoj-username>/<nazov-repo>.git
cd <nazov-repo>
```

### 2. Vytvorenie virtuálneho prostredia (odporúčané)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Inštalácia PyTorch s CUDA podporou
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> Pre CPU-only verziu (pomalšie):
> ```bash
> pip install torch torchvision
> ```

### 4. Inštalácia ostatných závislostí
```bash
pip install -r requirements.txt
```

---

## Štruktúra projektu

```
project/
├── VGG-16_CODE.ipynb   # Hlavný notebook so všetkým kódom
├── data/               # CIFAR-10 sa stiahne automaticky sem
└── results/            # Výstupné súbory (modely, JSON histórie, grafy)
    ├── E1_-_Frozen_feature_extraction_best.pth
    ├── E1_-_Frozen_feature_extraction_history.json
    ├── ...
    ├── accuracy_val_all.png
    ├── loss_all.png
    ├── speed_accuracy.png
    ├── ablation_e2_vs_a1.png
    ├── confusion_matrix.png
    ├── per_class_accuracy.png
    └── summary_results.csv
```

---

## Reprodukcia kódu

### Krok 1 – Otvorenie notebooku
Otvor `VGG-16_CODE.ipynb` v Jupyter Lab alebo Jupyter Notebook:
```bash
jupyter lab VGG-16_CODE.ipynb
```

### Krok 2 – Spustenie buniek v poradí

Notebook je rozdelený do týchto sekcií (spúšťaj postupne):

| # | Sekcia | Popis |
|---|--------|-------|
| 1 | **IMPORTS** | Import knižníc, overenie CUDA a GPU |
| 2 | **KONFIGURÁCIA** | Nastavenie hyperparametrov, ciest a seed-u |
| 3 | **DATASET** | Definícia transformácií a augmentácií |
| 4 | **DataLoadery** | Stiahnutie CIFAR-10 a vytvorenie DataLoaderov |
| 5 | **Build VGG-16** | Zostavenie modelu s novou klasifikačnou hlavou |
| 6 | **Zmrazenie vrstiev** | Funkcia `set_frozen_layers()` a `get_optimizer()` |
| 7 | **Tréningové funkcie** | Funkcie `train_epoch()`, `eval_epoch()`, `run_experiment()` |
| 8–12 | **Experimenty E1–E5** | Spustenie 5 experimentov (viď nižšie) |
| 13 | **Ablation A1** | Experiment bez dátovej augmentácie |
| 14 | **Vizualizácia** | Grafy accuracy, loss, speed–accuracy scatter |
| 15 | **Konfúzna matica** | Analýza chýb najlepšieho modelu |
| 16 | **Súhrnná tabuľka** | Export výsledkov do CSV |

Sekcie 1 až 7 musia byť spustené vždy pred spúšťaním experimentu. Pred spustením každého experimentu ak ste pred nim spustili druhý odporúčame reštartovať kernel kvôli pretekaniu VRAM do shared memory čo výrazne spomalí ďalšie experimenty.

### Krok 3 – Overenie inštalácie (bunka 1)
Po spustení prvej bunky by mal výstup vyzerať takto:
```
PyTorch: 2.6.0+cu124
CUDA dostupná: True
GPU: NVIDIA GeForce RTX 4060
VRAM: 8.6 GB
```

### Krok 4 – Dataset (bunka 4)
CIFAR-10 sa stiahne automaticky do `./data/`. Výstup:
```
Train: 45,000 | Val: 5,000 | Test: 10,000
Train: 45,000 | Val: 5,000 | Test: 10,000
```

### Krok 5 – Spustenie experimentov
Každý experiment sa spúšťa samostatnou bunkou. Výsledky (váhy `.pth` a história `.json`) sa ukladajú do `./results/`.
Vizualizačné sekcie vyžadújú aby boli výsledky experimentov uložené v ./results/.
---

## Experimenty

| Experiment | Rozmrazené vrstvy | Trénovateľné parametre |
|------------|-------------------|------------------------|
| **E1** – Frozen feature extraction | len klasifikačná hlava | 89.0 % |
| **E2** – Unfreeze Block 5 | block5 + hlava | 89.0 % |
| **E3** – Unfreeze Block 4–5 | block4–5 + hlava | 96.1 % |
| **E4** – Unfreeze Block 3–5 | block3–5 + hlava | 99.8 % |
| **E5** – Full fine-tuning | celá sieť | 100 % |

### Hyperparametre
```python
INPUT_SIZE   = 224       # VGG-16 vstupná veľkosť
BATCH_SIZE   = 64
NUM_EPOCHS   = 10
LR_HEAD      = 1e-3      # learning rate pre klasifikačnú hlavu
LR_BACKBONE  = 1e-4      # learning rate pre konvolučné vrstvy
WEIGHT_DECAY = 1e-4
SEED         = 42
```

### Augmentácie (tréning)
- `RandomHorizontalFlip(p=0.5)`
- `RandomCrop(224, padding=8)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`
- Normalizácia ImageNet mean/std

---

## Poznámky k reprodukovateľnosti

- Nastavený `SEED = 42` pre `torch` aj `numpy` pre deterministické výsledky.
- Malé odchýlky vo výsledkoch sú možné kvôli CUDA non-deterministickým operáciám.
- Pre úplnú reprodukovateľnosť možno pridať:
  ```python
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
---
