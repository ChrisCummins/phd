#!/usr/bin/env bash

while true; do
    timeout 1800 ./mine.py 1000 glsl
    timeout 1800 ./mine.py 1000 javascript
    timeout 1800 ./mine.py 1000 c
    timeout 1800 ./mine.py 1000 haskell
    timeout 1800 ./mine.py 1000 go
    timeout 1800 ./mine.py 1000 java
    timeout 1800 ./mine.py 1000 php
    timeout 1800 ./mine.py 1000 python
    timeout 1800 ./mine.py 1000 rust
    timeout 1800 ./mine.py 1000 swift
done
