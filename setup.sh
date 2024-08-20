#!/bin/bash

git submodule init
git submodule update
git submodule foreach git lfs pull
