#!/bin/bash
set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"
${PIP_INSTALL} --upgrade pip
${PIP_INSTALL} --upgrade setuptools
${PIP_INSTALL} --upgrade torchada
