# Kidney CT: YOLO11n + ResNet50

Codice sperimentale della tesi **“Metodologie di Deep Learning per la classificazione automatica di tumori renali da imaging medico”** di Aurora Macali.

Repository ufficiale: <https://github.com/macaliaurora62/Kidney-Thesis>

La pipeline localizza le regioni renali con YOLO11n, costruisce due pannelli da 224×224 pixel e confronta:

1. una ResNet50 baseline sull'immagine affiancata da 448×224 pixel;
2. una ResNet50 a due rami con encoder condiviso, maschera di validità e masked max pooling.

## Risultati principali

| Esperimento | Risultato |
| --- | ---: |
| YOLO11n Precision | 0,9820 |
| YOLO11n Recall | 0,9459 |
| YOLO11n mAP@50 | 0,9619 |
| YOLO11n mAP@50–95 | 0,7692 |
| Baseline Macro-F1 validation | 0,8389 |
| Baseline Accuracy test | 0,8216 |
| Baseline Macro-F1 test | 0,7652 |
| Due rami Macro-F1 validation | 0,8937 |
| Due rami Accuracy test | 0,9129 |
| Due rami Macro-F1 test | 0,8991 |

Questi valori coincidono con quelli riportati nella tesi. I checkpoint dei classificatori sono selezionati esclusivamente mediante il Macro-F1 sul validation set; il test set viene utilizzato soltanto per la valutazione finale.

## Struttura

```text
.
├── configs/experiment.yaml
├── data/
│   ├── README.md
│   └── annotations/
│       ├── data.yaml
│       ├── labels/{train,val,test}/
│       └── metadata/split_manifest.csv
├── notebooks/complete_pipeline_colab.ipynb
├── results/
│   ├── baseline/
│   │   ├── history_phase1.csv
│   │   ├── history_phase2.csv
│   │   └── test_metrics.json
│   ├── two_branch/
│   │   ├── history_phase1.csv
│   │   ├── history_phase2.csv
│   │   └── test_metrics.json
│   ├── yolo/
│   │   ├── test_metrics.json
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix_normalized.png
│   │   └── Box*_curve.png
│   ├── dataset_distribution.csv
│   ├── final_metrics.json
│   └── model_comparison.csv
├── scripts/
│   ├── train_yolo.py
│   ├── validate_annotations.py
│   └── verify_thesis_consistency.py
├── src/
│   ├── cropping.py
│   ├── models.py
│   └── reproducibility.py
├── THESIS_CONSISTENCY.md
└── requirements.txt
```

## Dati richiesti

Le immagini non sono incluse nella repository. Sono invece incluse le 600 label YOLO, il manifest dello split group-aware e `data.yaml`, nella cartella `data/annotations/`.

Il dataset di classificazione è disponibile su Kaggle: [CT Kidney Dataset: Normal, Cyst, Tumor and Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone).

Per riprodurre l'intera pipeline, il notebook si aspetta nella cartella principale di Google Drive:

- `Kidney_YOLO_Group_Split.zip`: 600 immagini annotate, label YOLO e `metadata/split_manifest.csv`;
- `kidney_yolo11n_best.zip`: checkpoint YOLO11n selezionato sul validation set;
- `full_dataset_kidney_kaggle.zip`: dataset Kaggle completo da 12.441 immagini.

Se un file ricevuto ha un suffisso automatico, ad esempio `Kidney_YOLO_Group_Split(3).zip`, rinominarlo con il nome atteso prima dell'esecuzione. La struttura dettagliata è descritta in [data/README.md](data/README.md).

## Controlli prima dell'esecuzione

Installare le dipendenze:

```bash
python -m pip install -r requirements.txt
```

Verificare manifest, label e assenza di leakage:

```bash
python scripts/validate_annotations.py
```

Verificare che configurazione e risultati versionati coincidano con i valori finali della tesi:

```bash
python scripts/verify_thesis_consistency.py
```

Il controllo accetta come finali i risultati dei classificatori calcolati sull'intero test set di **1.917 immagini**. Gli output preliminari dei classificatori calcolati sulle sole 96 immagini annotate per YOLO non fanno parte della repository finale.

## Esecuzione in Google Colab

1. Caricare i tre file richiesti nella cartella principale di Google Drive.
2. Aprire `notebooks/complete_pipeline_colab.ipynb` in Google Colab.
3. Selezionare un runtime GPU; gli esperimenti originali sono stati eseguiti su NVIDIA Tesla T4.
4. Eseguire **Runtime → Run all** senza cambiare l'ordine delle celle.

Per riaddestrare YOLO11n da zero, impostare nel notebook:

```python
TRAIN_YOLO_FROM_SCRATCH = True
```

La configurazione utilizza seed 42, 100 epoche massime, immagini 640×640, batch size 16 e patience 20.

## Protocollo sperimentale

- dataset: Cyst, Normal, Stone e Tumor;
- split group-aware, senza gruppi condivisi tra training, validation e test;
- seed 42 per Python, NumPy, PyTorch, DataLoader e YOLO;
- massimo due detection per immagine, ordinate da sinistra a destra;
- margine del 5% intorno a ciascuna bounding box;
- pannello nero quando è disponibile un solo crop;
- checkpoint selezionati mediante validation Macro-F1;
- Accuracy, Balanced Accuracy, Macro-Precision, Macro-Recall, Macro-F1, ROC-AUC e matrice di confusione sul test.

## Ambiente originale

Python 3.12.13, PyTorch 2.11.0+cu128, torchvision 0.28.0, Ultralytics 8.4.117, scikit-learn 1.9.0, pandas 3.0.5 e CUDA 12.8.

Il codice costituisce un prototipo sperimentale e non uno strumento diagnostico per uso clinico.
