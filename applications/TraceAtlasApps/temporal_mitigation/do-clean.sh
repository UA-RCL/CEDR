#!/bin/sh

if ls *.bc 1>/dev/null 2>&1; then
    rm *.bc;
fi

if ls *.out 1>/dev/null 2>&1; then
    rm *.out;
fi

if ls *.ll 1>/dev/null 2>&1; then
    rm *.ll
fi

if ls *.so 1>/dev/null 2>&1; then
    rm *.so
fi

if ls *.json 1>/dev/null 2>&1; then
    rm *.json
fi

if ls *.trc 1>/dev/null 2>&1; then
    rm *.trc
fi
