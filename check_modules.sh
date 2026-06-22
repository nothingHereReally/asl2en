#!/bin/bash



if [ "$(pwd)/.venv/bin/pip" != "$(which pip)" ]; then
    source .venv/bin/activate
fi


installed="$(pip freeze)"


echo "$installed" | grep -iE "^mediapipe="
echo "mediapipe==0.10.21"

echo "$installed" | grep -iE "^opencv-python="
echo "opencv-python==4.12.0.88"

echo "$installed" | grep -iE "^tensorflow="
echo "tensorflow==2.19.0"

echo "$installed" | grep -iE "^torch="
echo "torch==2.7.1"

echo "$installed" | grep -iE "^torchaudio="
echo "torchaudio==2.7.1"

echo "$installed" | grep -iE "^torchvision="
echo "torchvision==0.22.1"

echo "$installed" | grep -iE "^happytransformer="
echo "happytransformer==3.0.0"

echo "$installed" | grep -iE "^tf_keras="
echo "tf_keras==2.19.0"

echo "$installed" | grep -iE "^keras="
echo "keras==3.11.2"

echo "$installed" | grep -iE "^pygame="
echo "pygame==2.6.1"
