#!/bin/sh

PROG_NAME=`basename "$0"`
C_FILE=""
INLINE=false
SEM_OPT=false
SKIP_TRACE=false
SKIP_KERNEL_DETECTION=false
SKIP_FINAL_COMPILATION=false
TRACEHOME=/localhome/jmack2545/rcl/DASH-SoC/TraceAtlas/build
COMPILERRT=/localhome/jmack2545/rcl/DASH-SoC/llvm/llvm-project/compiler-rt/build/lib/linux/libclang_rt.builtins-aarch64.a

ARCH=""
ARCH_FLAGS=""

print_usage() {
  echo "Usage:"
cat << EOF
  ${PROG_NAME} [-h|--help]
  ${PROG_NAME} [options] -a|--arch {x86,aarch64} file_to_process.c"

    Required Arguments:
      -a|--arch
        The architecture to build for. Supported values are x86 and aarch64. These flags are only used in final shared object creation.
      file_to_process.c
        The main input source file that is to be instrumented, compiled, traced, and refactored to a DAG-based shared object

    Options:
      [-i|--inline]
        Attempt to inline all functions from the original source file before tracing
      [--semantic-opt]
        Attempt to utilize "semantic optimization" substitutions in the final binary creation
      [--no-trace]
        Skip the trace instrumentation and trace collection steps
      [--no-kernel-detection]
        Skip the trace processing and kernel detection/DAG extraction steps
      [--no-final-processing]
        Skip the application refactoring and final shared object compilation steps
      [-l<lib_suffix>]
        Add a library that is required to compile this application for tracing and in shared object creation
        Ex: -lm for libmath
      [-d|--dependency <dep_file>]
        Add a source file dependency that is required for compilation for tracing and in shared object creation
      [-t|--trace-home <TraceAtlasDir>]
        Define the build directory of TraceAtlas
      [--compiler-rt <compiler-rt lib>]
        Define the full path to LLVM's compiler-rt library to be statically linked in during shared object creation
EOF
}

while (( "$#" )); do
  case "$1" in
    -i|--inline)
      INLINE=true
      shift 1
      ;;
    --semantic-opt)
      SEM_OPT=true
      shift 1
      ;;
    --no-trace)
      SKIP_TRACE=true
      shift 1
      ;;
    --no-kernel-detection)
      SKIP_KERNEL_DETECTION=true
      shift 1
      ;;
    --no-final-processing)
      SKIP_FINAL_COMPILATION=true
      shift 1
      ;;
    -l*)
      LIBS="$LIBS $1"
      shift 1
      ;;
    -d|--dependency)
      DEPS="$DEPS $2"
      shift 2
      ;;
    -t|--trace-home)
      if [ -n "$TRACEHOME" ]; then
        echo "Error: multiple usages of --trace-home flag, using the last one specified" >&2
      fi
      TRACEHOME="$2"
      shift 2
      ;;
    --compiler-rt)
      if [ -n "$COMPILERRT" ]; then
        echo "Error: multiple usages of --compiler-rt flag, using the last one specified" >&2
      fi
      COMPILERRT="$2"
      shift 2
      ;;
    -a|--arch)
      if [ -n "$ARCH" ]; then
        echo "Error: multiple usages of --arch flag, using the last one specified" >&2
      fi
      case "$2" in
        x86)
          ARCH="x86"
          ARCH_FLAGS=""
          ;;
        aarch64)
          ARCH="aarch64"
          ARCH_FLAGS="-march=armv8-a -target aarch64-linux-gnu"
          ;;
        *)
          echo "Error: unsupported architecture specified" >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    -*|--*)
      echo "Error: Unsupported flag $1" >&2
      print_usage
      exit 1
      ;;
    *)
      if [ -z "$C_FILE" ]; then
        C_FILE="$1"
        shift 1
      else
        echo "Error: Only one C file can be passed as the target to be processed. Pass in other dependencies via -d/--dependency." >&2
        exit 1
      fi
      ;;
  esac
done

if [ -z "$C_FILE" ]; then
  echo "Requires main C file to process" >&2
  print_usage
  exit 1
fi

if [ -z "$ARCH" ]; then
  echo "Target architecture must be specified" >&2
  print_usage
  exit 1
fi

IFS=' ' read -r -a DEPS <<< "$DEPS"
IFS=' ' read -r -a LIBS <<< "$LIBS"

if [ "$SKIP_TRACE" = false ]; then
  echo "Stage: Initial compilation"
  clang-9 -S -flto -fPIC -static -fuse-ld=lld-9 ${C_FILE} -o output-${C_FILE%.c}.ll
  # Note: this stage might need all necessary functions to have the "always inline" attribute
  if [ "$INLINE" = true ]; then
    echo "Stage: Inlining function calls"
    opt-9 -always-inline output-${C_FILE%.c}.ll -S -o output-${C_FILE%.c}.ll
  fi
  echo "Stage: Encoded annotation"
  opt-9 -load $TRACEHOME/lib/AtlasPasses.so -EncodedAnnotate output-${C_FILE%.c}.ll -S -o output-${C_FILE%.c}-annotate.ll
  echo "Stage: Encoded trace instrumentation"
  opt-9 -load $TRACEHOME/lib/AtlasPasses.so -EncodedTrace output-${C_FILE%.c}.ll -S -o output-${C_FILE%.c}-opt.ll
fi

if [ "$SKIP_KERNEL_DETECTION" = false ]; then
  # ${LIBS[@]}
  echo "Stage: Tracer binary compilation"
  clang++-9 -static -fuse-ld=lld-9 \
            -lpthread -lz $TRACEHOME/lib/libAtlasBackend.a \
            ${DEPS[@]} output-${C_FILE%.c}-opt.ll \
            -o ${C_FILE%.c}-tracer.out

  echo "Stage: Trace collection"
  ./${C_FILE%.c}-tracer.out

  if [ -f outfile.txt ]; then
    echo "Removing previous outfile.txt"
    rm outfile.txt
  fi
  echo "Stage: Kernel extraction"
  $TRACEHOME/bin/cartographer -i raw.trc -k kernel-${C_FILE%.c}.json
  echo "Stage: DAG extraction"
  $TRACEHOME/bin/dagExtractor -t raw.trc -k kernel-${C_FILE%.c}.json -o kernel-${C_FILE%.c}-dagExtractor.json
fi

if [ "$SKIP_FINAL_COMPILATION" = false ]; then
  echo "Stage: Application refactoring/region outlining"
  $TRACEHOME/bin/kwrap -semantic-opt=${SEM_OPT} -a output-${C_FILE%.c}-annotate.ll -k kernel-${C_FILE%.c}.json -d kernel-${C_FILE%.c}-dagExtractor.json -n ${C_FILE%.c}-${ARCH} -o output-${C_FILE%.c}-extracted.ll -o2 ${C_FILE%.c}-${ARCH}.json
  echo "Stage: Shared object compilation"
  if [ "$ARCH" = "x86" ]; then
    clang++-9 ${ARCH_FLAGS} -shared -fPIC -fuse-ld=lld-9 ${LIBS[@]} ${DEPS[@]} output-${C_FILE%.c}-extracted.ll -o ${C_FILE%.c}-${ARCH}.so
  else
    clang++-9 ${ARCH_FLAGS} -shared -fPIC -fuse-ld=lld-9 ${LIBS[@]} $COMPILERRT ${DEPS[@]} output-${C_FILE%.c}-extracted.ll -o ${C_FILE%.c}-${ARCH}.so
  fi
fi
echo "Complete!"
