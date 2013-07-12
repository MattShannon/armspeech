#!/bin/bash
set -e
set -u
set -o pipefail

if [[ "`dirname "$0"`" != "." ]]; then
    echo "check_codedep.sh: this tool is designed to be run from the current directory" 1>&2
    exit 1
fi

tmpDir=`mktemp -d`
echo "(using temporary dir $tmpDir)"

for pyFile in "$@"; do
    moduleName="`echo "$pyFile" | sed -r 's%^\./%%;s%/%.%g;s/\.py$//'`"
    echo
    echo "(moduleName = $moduleName, pyFile = $pyFile)"
    echo
    pyFileNew="$tmpDir"/"`basename "$pyFile"`"

    PYTHONPATH=. python check_codedep.py "$moduleName" > "$pyFileNew"

    if [[ "`diff -q "$pyFile" "$pyFileNew"`" != "" ]]; then
        vimdiff "$pyFile" "$pyFileNew"
    else
        echo "(no change)"
    fi

    rm -f "$pyFileNew"
    echo
done

rmdir "$tmpDir" && echo "(cleaned-up temporary dir $tmpDir)"
