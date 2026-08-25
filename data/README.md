# Preparazione dei dati

La repository non distribuisce le immagini TC né i checkpoint addestrati. Distribuisce invece le annotazioni YOLO e il manifest group-aware in `data/annotations/`. Il notebook usa tre archivi nella cartella principale di Google Drive.

## 1. Dataset annotato

Nome richiesto: `Kidney_YOLO_Group_Split.zip`

```text
Kidney_YOLO_Group_Split/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── metadata/
    └── split_manifest.csv
```

Il manifest deve contenere almeno le colonne:

```text
category,group,filename,split,boxes
```

Il sottoinsieme contiene 600 immagini bilanciate, 150 per classe, suddivise per gruppo in 411 training, 93 validation e 96 test.

Le annotazioni versionate nella repository contengono 600 file di label: 411 training, 93 validation e 96 test. Due file sono intenzionalmente vuoti perché il manifest riporta zero bounding box; le altre 598 immagini consentono la generazione dei crop. Il numero totale di box è 782/179/185 per training/validation/test.

Per usare direttamente `data/annotations/data.yaml`, copiare le immagini nelle directory `data/annotations/images/train`, `data/annotations/images/val` e `data/annotations/images/test`, mantenendo i nomi presenti nel manifest.

## 2. Dataset completo

Nome richiesto: `full_dataset_kidney_kaggle.zip`

L'archivio deve contenere la directory `Grouped images` con le cartelle `Cyst`, `Normal`, `Stone` e `Tumor`, organizzate in gruppi. Il dataset iniziale contiene 12.441 immagini.

## 3. Checkpoint YOLO11n

Nome richiesto: `kidney_yolo11n_best.zip`

Nel progetto originale si tratta di un checkpoint PyTorch `.pt` rinominato con estensione `.zip`; il notebook lo copia localmente come `kidney_yolo11n_best.pt`. In alternativa è possibile impostare `TRAIN_YOLO_FROM_SCRATCH = True` e rigenerarlo a partire da `yolo11n.pt`.

## Controllo del leakage

Il notebook verifica che ogni combinazione `category/group` appartenga a un solo split. Le immagini correlate dello stesso gruppo non possono comparire contemporaneamente in training, validation e test.

Il manifest incluso contiene 217 combinazioni `category/group`: 150 nel training set, 33 nel validation set e 34 nel test set, senza sovrapposizioni.
