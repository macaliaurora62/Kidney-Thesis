# Verifica di coerenza con la tesi

Il confronto è stato effettuato con l'elaborato `Tesi_Aurora_Macali(1).docx` e con gli output finali degli esperimenti.

## Valori verificati

| Esperimento | Voce | Tesi | Repository | Esito |
| --- | --- | ---: | ---: | --- |
| YOLO11n | Precision | 0,9820 | 0,9820 | coerente |
| YOLO11n | Recall | 0,9459 | 0,9459 | coerente |
| YOLO11n | mAP@50 | 0,9619 | 0,9619 | coerente |
| YOLO11n | mAP@50–95 | 0,7692 | 0,7692 | coerente |
| ResNet50 baseline | Macro-F1 validation | 0,8389 | 0,8389 | coerente |
| ResNet50 baseline | Accuracy test | 0,8216 | 0,8216 | coerente |
| ResNet50 baseline | Macro-F1 test | 0,7652 | 0,7652 | coerente |
| ResNet50 a due rami | Macro-F1 validation | 0,8937 | 0,8937 | coerente |
| ResNet50 a due rami | Accuracy test | 0,9129 | 0,9129 | coerente |
| ResNet50 a due rami | Macro-F1 test | 0,8991 | 0,8991 | coerente |

Sono inoltre verificati:

- dataset completo: 12.441 immagini prima delle esclusioni;
- immagini escluse: 134;
- immagini utilizzate: 12.307;
- split dei classificatori: 8.498 training, 1.892 validation e 1.917 test;
- dataset YOLO: 600 immagini e 1.146 bounding box;
- split YOLO: 411/93/96 immagini e 782/179/185 bounding box;
- assenza di gruppi condivisi tra training, validation e test;
- configurazione: seed 42, YOLO11n, 100 epoche, 640 px, batch 16, confidence 0,25, NMS IoU 0,70, margine crop 5%, pannelli 224×224 e input combinato 448×224;
- architetture e fasi di addestramento dei due classificatori.

## Criterio per i risultati finali

La valutazione finale dei classificatori deve contenere 1.917 campioni. I file preliminari ottenuti valutando i classificatori sulle sole 96 immagini annotate del detector sono stati esclusi dalla repository finale. Il notebook include i controlli sulla numerosità del test set e rigenera gli output a partire dai checkpoint selezionati sul validation set.

Il comando seguente ripete i controlli numerici e strutturali:

```bash
python scripts/verify_thesis_consistency.py
```
