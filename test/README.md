# CEDR - Tests

This directory contains automated tests for validating the functionality of CEDR.

Unit tests are provided that use the [Catch2](https://github.com/catchorg/Catch2) framework to test individual components while end-to-end tests will eventually be provided via shell scripts.

## Executing unit tests
Execute unit tests by building the test binary as follows
```bash
make all
```
and running with
```bash
./test-runner
```

The directory can then be cleaned with
```bash
make clean
```
and while it's a bit overkill now given that there's only one test binary to delete, might be nice in the future.

## Executing End-to-End Tests

TBD!
