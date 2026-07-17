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


# 10 modules
check mediapipe        "0.10.21"
check opencv-python    "4.12.0.88"
check tensorflow       "2.19.0"
check torch            "2.7.1"
check torchaudio       "2.7.1"
check torchvision      "0.22.1"
check happytransformer "3.0.0"
check tf_keras         "2.19.0"
check keras            "3.11.2"
check pygame           "2.6.1"
