import pandas as pd

mapping = {
  'Celiac': 1,
  'Non-Celiac': 0,
  'tCD-TG+': 1,
  'tCD-TG-': 1,
  'Untreated CD': 1,
  'Healthy': 0,
}

metadata = pd.read_csv('../metadata.csv', sep=';')[['Run', 'Class']]

metadata['Class'] = metadata['Class'].map(mapping)

metadata.to_csv('labels.csv', sep=';', index=False)
