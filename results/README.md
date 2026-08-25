# Risultati versionati

Questa cartella contiene esclusivamente risultati finali coerenti con i valori riportati nella tesi.

- `yolo/test_metrics.json`: valutazione YOLO sulle 96 immagini annotate del test set, contenenti 185 istanze renali.
- `yolo/confusion_matrix*.png` e `yolo/Box*_curve.png`: grafici della medesima valutazione YOLO finale.
- `baseline/test_metrics.json`: valutazione finale della ResNet50 baseline sulle 1.917 immagini del test set.
- `two_branch/test_metrics.json`: valutazione finale della ResNet50 a due rami sulle 1.917 immagini del test set.
- `baseline/history_phase*.csv` e `two_branch/history_phase*.csv`: curve di addestramento delle due fasi.
- `dataset_distribution.csv`: distribuzione del dataset completo dopo le esclusioni descritte nella tesi.
- `final_metrics.json`: riepilogo normalizzato con i valori numerici completi.
- `model_comparison.csv`: confronto sintetico dei due classificatori.

Gli output preliminari dei classificatori con 96 righe non sono inclusi: 96 è la numerosità del test annotato del detector YOLO, non del test finale dei classificatori. I report per classe, le predizioni e le matrici di confusione dei classificatori sono rigenerati dal notebook a partire dai checkpoint finali e devono avere 1.917 campioni.

Eseguire `python scripts/verify_thesis_consistency.py` dalla radice della repository per controllare automaticamente i dati versionati.
