#!/usr/bin/env bash

if [ ! -f python/timer/bin/active ]; then
    virtualenv -p python3.6 python/timer
fi

source python/timer/bin/activate

pip install -r requirements.txt
pip install pytest

pytest
