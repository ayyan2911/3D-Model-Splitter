# Guida al 3D Model Splitter

## Come avviare e utilizzare il tool

### Prerequisiti

Assicurati di avere Python installato e le dipendenze necessarie:

```
pip install numpy dash plotly shutil
```

### Avvio dello splitter

Posizionati nella cartella del progetto:

```
cd /percorso/del/tuo/progetto
```

Lancia lo script principale:

```
python split_colmap_interactive.py /percorso/del/tuo/colmap /output --overlap 15
```

Parametri:
- /percorso/del/tuo/colmap: cartella contenente l'output COLMAP (deve avere sparse/0/ con cameras.bin, images.bin, points3D.bin)
- /output: cartella dove salvare le parti
- --overlap 15: percentuale di overlap tra le parti (default 15%)

Apri il browser e vai a:

```
http://127.0.0.1:8050
```

---

## Come funziona l'interfaccia

### Pannello di controllo (sinistra)

Modalità di taglio:
- Croce (X + Y): taglia sia in X che in Y
- Solo Asse X: taglia solo in orizzontale
- Solo Asse Y: taglia solo in verticale

Visualizzazione:
- Numero di parti totali in tempo reale
- Distribuzione dei punti per cella
- Tagli attuali visualizzati come piani colorati

### Interazione con la nuvola 3D (destra)

- Clicca sulla nuvola per aggiungere un taglio
- I tagli si posizionano automaticamente per bilanciare la densità dei punti
- Le parti vengono colorate con colori diversi
- Puoi ruotare e zoomare la scena con il mouse

### Pulsanti

- Cancella tutti i tagli: rimuove tutti i tagli
- Annulla ultimo taglio: rimuove l'ultimo taglio aggiunto
- Salva: genera i modelli COLMAP divisi e il file cluster_bounds.json

---

## Cosa viene salvato

### Struttura output:

```
output/
├── part_0/
│   ├── sparse/0/
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin
│   └── images/
│       └── (immagini copiate)
├── part_1/
│   └── ...
└── cluster_bounds.json
```

### File cluster_bounds.json:

Contiene i bounds di ogni parte per il post-processing:

```json
{
  "parts": [...],
  "overlap": 0.15,
  "x_cuts": [...],
  "y_cuts": [...],
  "global_bounds": {...}
}
```

---

## Pulire i Gaussian PLY

Dopo aver addestrato i Gaussian Splatting su ogni parte, puoi usare clean_gaussian.py per tagliare i PLY:

```
python clean_gaussian.py input.ply cluster_bounds.json /percorso/colmap/sparse/0 --part 0 -o part_0_clean.ply --use_overlap
```

Parametri:
- input.ply: il file PLY di Gaussian Splatting
- cluster_bounds.json: generato dallo splitter
- /percorso/colmap/sparse/0: cartella COLMAP originale
- --part 0: indice della parte da estrarre
- -o part_0_clean.ply: file di output
- --use_overlap: usa i bounds con overlap

---

## Unire i PLY puliti

Dopo aver pulito tutte le parti, usa merge_ply_files.py per unirle:

```
python merge_ply_files.py part_0_clean.ply part_1_clean.ply ... -o modello_completo.ply
```

Oppure con una cartella:

```
python merge_ply_files.py --folder ./parti_pulite/ -o modello_completo.ply
```

---

## Flusso di lavoro completo

1. COLMAP (modello originale)
2. split_colmap_interactive.py
3. Parti COLMAP + cluster_bounds.json
4. Addestramento Gaussian Splatting su ogni parte
5. clean_gaussian.py (usa cluster_bounds.json)
6. PLY puliti per ogni parte
7. merge_ply_files.py
8. Modello Gaussian Splatting completo e pulito

---

## Consigli utili

- Overlap: mantieni 15% per transizioni morbide
- Numero di tagli: non esagerare, 2-4 tagli per asse sono sufficienti
- Performance: lavora con modelli < 2M punti per prestazioni fluide
- Salvataggio: fai sempre un backup del modello originale

---

## Errori comuni

| Errore | Soluzione |
|--------|-----------|
| FileNotFoundError | Controlla il percorso di COLMAP |
| Port 8050 already in use | Cambia porta nel codice |
| ImportError: No module named... | Installa le dipendenze mancanti |

---

## Tecnologie utilizzate

- Python: linguaggio principale
- Dash + Plotly: interfaccia 3D interattiva
- NumPy: calcoli geometrici e PCA
- Struct: lettura/scrittura file binari COLMAP
