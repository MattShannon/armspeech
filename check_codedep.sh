#!/bin/bash
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

    if [[ "`diff -q "$pyFile" "$pyFileNew"`" != "" ]]; then
        # (below works around vim's "Input is not from a terminal" warning)
        { echo "$pyFile"; echo "$pyFileNew"; } | xargs -d '\n' bash -c '</dev/tty vimdiff "$@"' ignoreme
    else
        echo "(no change)"
    fi

    rm -f "$pyFileNew"
    echo
done

rmdir "$tmpDir" && echo "(cleaned-up temporary dir $tmpDir)"
