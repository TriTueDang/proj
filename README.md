# Face Detection Evaluation

Tento projekt obsahuje 4 různé detektory obličejů a nástroje pro jejich vyhodnocení na datasetu UTKFace.

## Dataset UTKFace

UTKFace je rozsáhlý dataset obsahující více než 20 000 obrázků obličejů s informacemi o věku (0–116), pohlaví a rase.

V této složce jsou však zahrnuty pouze 1000 obrázků pro rychlé testování. Celý dataset je možné stáhnout zde: [UTKFace - Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)

### Struktura projektu

* **images/** – Obsahuje podmnožinu datasetu (1000 obrázků)
* **detectors/** – Implementace čtyř různých detektorů obličejů
* **evaluation/** – Skripty pro vyhodnocení přesnosti detektorů podle věku, pohlaví a rasy


### Detektory obličejů

1. Haar Cascade (OpenCV)
2. MTCNN (mtcnn)
3. BLAZEFACE (mediapipe)
4. SCRFD (insightface)

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

