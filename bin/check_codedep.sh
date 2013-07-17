#!/bin/bash
# Checks whether code-level dependencies are correctly declared.
#
# This script is just a wrapper around check_deps.py which calls vimdiff so you
# can see the proposed changes more easily.
# It also allows many modules to be checked at the same time.
#
# This script is a customized version of bin/check_codedep.sh from the codedep
# package.

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

set -e
set -u
set -o pipefail

tmpDir=`mktemp -d`
echo "(using temporary dir $tmpDir)"

for pyFile in "$@"; do
    moduleName="`echo "$pyFile" | sed -r 's%^\./%%;s%/%.%g;s/\.py$//'`"
    echo
    echo "(moduleName = $moduleName, pyFile = $pyFile)"
    echo
    pyFileNew="$tmpDir"/"`basename "$pyFile"`"

    PYTHONPATH=. python -m "codedep.check_deps" . "$moduleName" > "$pyFileNew"

    # below stanza is to ensure eval_local receives correct special treatment
    diffLineExpected=`{ grep -n '^def eval_local(' "$pyFile" || true; } | sed -r 's/:.*//'`
    if [[ "$diffLineExpected" == "" ]]; then
        diffExpected=""
    else
        diffExpected="`{ echo "$(( diffLineExpected - 1 ))a$diffLineExpected"; echo "> @codeDeps()"; }`"
    fi

    if [[ "`diff "$pyFile" "$pyFileNew"`" != "$diffExpected" ]]; then
        # (below works around vim's "Input is not from a terminal" warning)
        { echo "$pyFile"; echo "$pyFileNew"; } | xargs -d '\n' bash -c '</dev/tty vimdiff "$@"' ignoreme
    else
        echo "(no change)"
    fi

    rm -f "$pyFileNew"
    echo
done

rmdir "$tmpDir" && echo "(cleaned-up temporary dir $tmpDir)"
