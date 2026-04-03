# Face Detection Evaluation

Tento projekt obsahuje 5 různých detektorů obličejů a nástroje pro jejich vyhodnocení na datasetu UTKFace.

## Dataset UTKFace

UTKFace je rozsáhlý dataset obsahující více než 20 000 obrázků obličejů s informacemi o věku (0–116), pohlaví a rase.

V této složce jsou však zahrnuty pouze 1000 obrázků pro rychlé testování. Celý dataset je možné stáhnout zde: [UTKFace - Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)

### Struktura projektu

* **main.ipynb** – Jupyter notebook, použití detektorů na celý dataset UTKFace. Zobrazení heat-map a evaluation tabulek.
* **main_face_mask.ipynb** – Jupyter notebook pro vyhodnocení detekce roušek na Face Mask Detection datasetu.
* **images/** – Obsahuje podmnožinu datasetu UTKFace (1000 obrázků).
* **detectors/** – Implementace pěti různých detektorů obličeje.
* **evaluation/** - Logika pro vyhodnocení přesnosti detektorů (IoU, výpočet metrik).
* **utils/** – Pomocné skripty (loaders pro UTKFace a Face Mask dataset, ukládání výsledků, vizualizace).
* **results/** - Výsledky vyhodnocení ve formátu .csv.

### Detektory obličejů

V tomto projektu se využívá celkem 5 různých detektorů obličejů:

1.  **Haar Cascade (OpenCV)**: Klasický algoritmus využívající Cascade Classifier z knihovny OpenCV. Je velmi rychlý, ale citlivý na natočení obličeje.
2.  **MTCNN (Multi-task Cascaded Convolutional Networks)**: Hluboká neuronová síť schopná detekovat nejen obličeje, ale i obličejové body (oči, nos, ústa). Velmi přesný a robustní detektor.
3.  **BlazeFace (MediaPipe)**: Detektor od Googlu optimalizovaný pro mobilní zařízení a real-time aplikace. Vyniká extrémní rychlostí. CNN.
4.  **SCRFD (Sample and Computation Redistribution)**: Moderní a vysoce efektivní detektor z balíčku `insightface`. CNN.
5.  **Dlib HOG**: Detektor založený na algoritmu Histogram of Oriented Gradients (HOG) a Linear SVM. Je spolehlivý pro čelní pohledy na obličej.


    **Face Recognition (face_recognition)**: Jednoduchý wrapper nad knihovnou `dlib`, který usnadňuje detekci i rozpoznávání obličejů. Pro detekci standardně využívá HOG model, ale je možné ho přepnout i na přesnější CNN model.

## Detekce roušek (Face Mask Detection)

Projekt byl rozšířen o možnost detekce a vyhodnocení nošení roušek.

### Dataset Face Mask Detection
Dataset obsahuje obrázky s anotacemi ve formátu PASCAL VOC XML. Každý obličej je zařazen do jedné ze tří kategorií:
- `with_mask` (s rouškou)
- `without_mask` (bez roušky)
- `mask_weared_incorrect` (nesprávně nasazená rouška)

Celý dataset je dostupný na: [Face Mask Detection - Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

### Evaluace a metriky
Pro vyhodnocení kvality detekce a klasifikace roušek používám IoU (Intersection over Union) s prahem 0.5 a následující statistické metriky:

- **Accuracy (Přesnost)**: Procentuální podíl správně detekovaných a klasifikovaných tváří (TP) vůči všem detekcím (TP + FP + FN).
- **Precision (Preciznost)**: Vyjadřuje, kolik z detekovaných tváří skutečně patří do dané kategorie. `TP / (TP + FP)`
- **Recall (Senzitivita)**: Vyjadřuje, jakou část z celkového počtu tváří v datasetu dokázal detektor najít. `TP / (TP + FN)`

Výsledky jsou generovány samostatně pro každou kategorii i souhrnně pro celý detektor.

### Spuštění

Před spuštěním je třeba nainstalovat potřebné knihovny:

```bash
pip install -r requirements.txt
```

Spuštění detektorů a evaluace:

*   **UTKFace**: Použijte Jupyter notebook `main.ipynb`.
*   **Face Mask Detection**: Použijte Jupyter notebook `main_face_mask.ipynb`.

Při spuštění detektorů se ve složce `results/` vytvoří souhrnné CSV reporty a statistiky.


### Příspěvky

Příspěvky a vylepšení jsou vítány. Pokud máte návrhy na zlepšení nebo chcete přidat nový detektor, neváhejte vytvořit pull request.

