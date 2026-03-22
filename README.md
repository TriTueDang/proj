# Face Detection Evaluation

Tento projekt obsahuje 5 různých detektorů obličejů a nástroje pro jejich vyhodnocení na datasetu UTKFace.

## Dataset UTKFace

UTKFace je rozsáhlý dataset obsahující více než 20 000 obrázků obličejů s informacemi o věku (0–116), pohlaví a rase.

V této složce jsou však zahrnuty pouze 1000 obrázků pro rychlé testování. Celý dataset je možné stáhnout zde: [UTKFace - Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)

### Struktura projektu

* **main.ipynb** – Jupyter notebook, použití detektorů na celý dataset. Zobrazení heat-map a evaluation tabulek.
* **images/** – Obsahuje podmnožinu datasetu (1000 obrázků), celý dataset má více než 20 000 obrázků
* **detectors/** – Implementace pěti různých detektorů obličeje
* **evaluation/** - Skript pro vyhodnocení přesnosti detektorů.
* **utils/** – Pomocné skripty pro načtení obrazků, uložení výsledků složky **results/**, vykreslení heat-map, ověření detektoru
* **results/** - Výsledky vyhodnocení. Soubory formátu .csv.

### Detektory obličejů

V tomto projektu se využívá celkem 5 různých detektorů obličejů:

1.  **Haar Cascade (OpenCV)**: Klasický algoritmus využívající Cascade Classifier z knihovny OpenCV. Je velmi rychlý, ale citlivý na natočení obličeje.
2.  **MTCNN (Multi-task Cascaded Convolutional Networks)**: Hluboká neuronová síť schopná detekovat nejen obličeje, ale i obličejové body (oči, nos, ústa). Velmi přesný a robustní detektor.
3.  **BlazeFace (MediaPipe)**: Detektor od Googlu optimalizovaný pro mobilní zařízení a real-time aplikace. Vyniká extrémní rychlostí. CNN.
4.  **SCRFD (Sample and Computation Redistribution)**: Moderní a vysoce efektivní detektor z balíčku `insightface`. CNN.
5.  **Dlib HOG**: Detektor založený na algoritmu Histogram of Oriented Gradients (HOG) a Linear SVM. Je spolehlivý pro čelní pohledy na obličej.


    **Face Recognition (face_recognition)**: Jednoduchý wrapper nad knihovnou `dlib`, který usnadňuje detekci i rozpoznávání obličejů. Pro detekci standardně využívá HOG model, ale je možné ho přepnout i na přesnější CNN model.

### Spuštění

Před spuštěním je třeba nainstalovat potřebné knihovny:

```bash
pip install -r requirements.txt
```

Spuštění detektorů a evaluace:


Jupyter notebook: main.ipynb
Při spuštění detektoru obličeje se vytvoří složka results, kam se uloží celý dataframe ve formátu csv.


### Příspěvky

Příspěvky a vylepšení jsou vítány. Pokud máte návrhy na zlepšení nebo chcete přidat nový detektor, neváhejte vytvořit pull request.

