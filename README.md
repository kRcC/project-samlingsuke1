# Prosjekt 1 – Segmentering av verktøy med DinoV3 + logistisk regresjon

Dette prosjektet ble gjort i forbindelse med samlingsuke i faget **Anvendt maskinlæring**. Målet var å bygge en komplett pipeline for segmentering av verktøy i bilder – fra rådata til en enkel modell som kan gjenkjenne og maskere nye verktøy.

---

## Mål
- Samle egne bilder av verktøy (hammer, skrutrekker osv.)
- Lage binære masker (hvit = verktøy, svart = bakgrunn)
- Ekstrahere features med DinoV3
- Trene en enkel logistisk regresjon
- Teste modellen på nye bilder og visualisere resultatene

---

## Prosjektinnhold

- `tools/fix_heic_tools.py`  
  Script som konverterer HEIC-bilder til ekte PNG.
  Dette var nødvendig siden mange mobilbilder ble lagret som HEIC.

- `tools/click_n_mask_images.py` - Opprettet av KI
  Et interaktivt GUI (Tkinter + OpenCV + SAM) der vi kunne klikke på bilder for å lage masker:  
  - Venstreklikk = positivt punkt (verktøy)  
  - Høyreklikk = negativt punkt (bakgrunn)  
  - `Ctrl+Z` = angre siste punkt  
  - `r` = reset  
  - `s` = lagre binær maske (0/255)  
  - `o` = lagre overlay med maske på originalbildet  
  - Piltaster = bla til neste/forrige bilde  

- `main.ipynb`  
  Notebook der vi kjører hele KI-pipelinen:
  1. Leser inn bilder og masker
  2. Ekstraherer features fra DinoV3
  3. Trener en logistisk regresjon
  4. Tester modellen og visualiserer resultater

- `sam_vit_b_01ec64.pth`  
  Modellvektene for Segment Anything (SAM), brukt i annoterings-GUI.

- `requirements.txt`  
  Liste over alle python-pakker vi brukte.

---

pip install -r requirements.txt
