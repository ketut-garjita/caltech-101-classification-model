## Note:

Data is downloaded in the train_model.py process.

```
tfds_dir = "data/" 
dataset, info = tfds.load(
    "caltech101", as_supervised=True, with_info=True, data_dir=tfds_dir
)
```
