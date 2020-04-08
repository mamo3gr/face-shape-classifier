# face-shape-classifier
classification task for https://www.kaggle.com/niten19/face-shape-dataset

## requirements
- [poetry](https://python-poetry.org/)

## setup
```bash
$ poetry install
```

## crop face region
```bash
$ poetry run python batch_crop.py \
root_dir_in=/home/user/datasets/face_shape \
root_dir_out=/home/user/datasets/face_shape/cropped
```
