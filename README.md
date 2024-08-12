# gust
A modularized version of https://www.kaggle.com/code/siniuho/gust0811


### Install Kaggle API
[Ref1](https://kai-huang.medium.com/%E6%9C%80%E9%BD%8A%E5%85%A8%E7%9A%84-kaggle-api-%E4%B8%AD%E6%96%87%E6%89%8B%E5%86%8A-%E4%B8%80-84c01174deb5)


Goto https://www.kaggle.com/settings to get your `kaggle.json`

```
# Windows
pip install kaggle
mkdir C:\Users\user\.kaggle
mv C:\Users\user\Downloads\kaggle.json C:\Users\user\kaggle.json
chmod 600 C:\Users\user\kaggle.json
```

```
# Linux & MacOS
mkdir ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

###　Download Kaggle
```shell
kaggle kernels output gusthema/asl-fingerspelling-recognition-w-tensorflow -p /path/to/dest
```

`/path/to/dest` is the path to your download destination, for example, `D:\gust-1`.

**Bold** = building blocks
- [ ] **`tranformer.py`**
- [ ] **`preprocess.py`**
- [x] `feature_labels.py`
- [x] `characters.py`
- [x] `auxilaries.py` (not executable)
- [x] `setting.py` (not executable)