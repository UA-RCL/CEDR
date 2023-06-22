## Input Files needed
- input_phdata.txt - Placed in $DASH_DATA

## Comamnds to compile and execute
`clang -o BP_SAR BP_SAR.c -lgsl -lgslcblas -lm`
`./BP_SAR 1e-2 600 8000 128 512 0 800 2000 400 81 1e10 6e8`
