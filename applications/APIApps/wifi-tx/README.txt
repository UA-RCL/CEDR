bin/ - contains the Makefile and the build results
src/ - contains the source code, along with parent source file that emulates CEDR

to build, go to bin/ and run
make

this produces the shared object file that can be used with the dummy runtime and/or CEDR

to build a CEDR-free version (standalone), run
make standalone

which produces wifi-tx-x86.out and run it as, i.e., ./wifi-tx-x86.out

The file bin/txdata_reference-output.txt contains the reference output for verification.
