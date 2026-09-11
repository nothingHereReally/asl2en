#!/bin/bash



if [ "$(pwd)/.venv/bin/pip" != "$(which pip)" ]; then
    source .venv/bin/activate
fi
installed="$(pip freeze)"
check() {
    local module="$1"
    local version="$2"

    actual=$(echo "$installed" | grep -iE "^${module}=")

    if [[ "$actual" == "${module}==${version}" ]]; then
        echo "--OK----> ${module}==${version}"
    else
       echo "__NOT_EQUAL__ ${actual} != ${module}==${version} <-- expected"
    fi
}


# 14 modules, for mediapipe, tensorflow and keras to work properly
check happytransformer      "3.0.0"
check keras                 "3.15.1"
check keras-hub             "0.31.1"
check mediapipe             "0.10.21"
check numpy                 "2.3.2"
check opencv-contrib-python "4.11.0.86"
check opencv-python         "4.12.0.88"
check protobuf              "4.25.9"
check pygame                "2.6.1"
check tensorflow            "2.19.1"
check tf_keras              "2.19.0"
check torch                 "2.7.1"
check torchaudio            "2.7.1"
check torchvision           "0.22.1"


# -- 10 modules -- old --
# check mediapipe        "0.10.21"
# check opencv-python    "4.12.0.88"
# check tensorflow       "2.19.0"
# check torch            "2.7.1"
# check torchaudio       "2.7.1"
# check torchvision      "0.22.1"
# check happytransformer "3.0.0"
# check tf_keras         "2.19.0"
# check keras            "3.11.2"
# check pygame           "2.6.1"
