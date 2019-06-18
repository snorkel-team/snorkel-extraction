#!/usr/bin/env bash

sudo yum groupinstall "Development Tools"

sudo python3 -m pip install --upgrade pip

git clone https://github.com/HazyResearch/snorkel
cd snorkel
sudo python3 -m pip install .
