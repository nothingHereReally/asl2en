Testing the model v14



Clone this repo, better to use `bash` if you have it, compared to `cmd`.
Installing git by defualt also installs bash.
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch only_1st_10gloss https://github.com/nothingHereReally/asl2en.git
```
on cmd is
```bash
set GIT_LFS_SKIP_SMUDGE=1
git clone --branch only_1st_10gloss https://github.com/nothingHereReally/asl2en.git
```
The `GIT_LFS_SKIP_SMUDGE=1` is to skip downloading the lfs files,
due to this repo creator( me, nothingHereReally ) is not a pro
user and as a non-pro user github limits how many time the
lfs( large files ) files can be included on clones or on
git pull request.


Then on github go to `asl2en/model/aslvid2gloss_v14.keras`
or just go to this [link to model v14](https://github.com/nothingHereReally/asl2en/blob/only_1st_10gloss/model/aslvid2gloss_v14.keras)
then manually download the aslvid2gloss_v14.keras.
After manual download, overwrite the asl2en/model/aslvid2gloss_v14.keras
( on MS Windows that's asl2en\model\aslvid2gloss_v14.keras ) with
what you just manually downloaded


The [WLASL](https://github.com/dxli94/WLASL) dataset is also needed move it to asl2en/dataset,
where inside it has wlasl_dataset, or simply has this structure
asl2en/dataset/wlasl_dataset


create a python environment
```bash
cd asl2en
python -m venv .venv
```


activate it, via bash
```bash
source .venv/bin/activate
```
or via cmd
```bash
.\.venv\Scripts\activate.bat
```


install the needed python modules, ignore the .requirements.txt
we don't need them all for testing or running model v14,
below is for nvidia GPU enabled tensorflow, for GPU
tensorflow you also need to install cuda
and others, for more info please
see the [tensorflow installation guide for nvidia GPU](https://www.tensorflow.org/install/pip)
```bash
pip install "tensorflow[and-cuda]"
pip install opencv-python
pip install mediapipe
```
or for CPU only
```bash
pip install tensorflow
pip install opencv-python
pip install mediapipe
```


to get the latest pip version
```bash
pip install --upgrade pip
```


now test( please use **`bash`** to run this )
```bash
python -m src_asl2gloss.checks.test_g10_model
```

