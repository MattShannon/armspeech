#!/bin/bash
PYTHONPATH=. /usr/bin/python -u armspeech/test_armspeech.py "$@"
PYTHONPATH=. /usr/bin/python -u test_codedep.py "$@"
