#!/bin/sh
params="$@"
echo parameters are $params
python3.6 process.py /input /output $params