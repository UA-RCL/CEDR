# Pulse Doppler Radar

## Input Files needed
- input_pd_pulse.txt - Needs to be located in same folder as code
- input_pd_ps.txt - Needs to be located in same folder as code


## Comamnds to compile and execute
- `clang -o pulse_doppler pulse_doppler.c -lfftw3 -lm`
- `./pulse_doppler 256 128 1.27e-4`
