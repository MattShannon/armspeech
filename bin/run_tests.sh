#!/bin/bash
PYTHONPATH=. /usr/bin/python -u bisque/test_bisque.py "$@"
PYTHONPATH=. /usr/bin/python -u armspeech/test_armspeech.py "$@"
