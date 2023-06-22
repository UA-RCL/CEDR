#!/bin/bash

#set -x
PROG_NAME=`basename "$0"`
C_FILE=""
C_FILE_NO_EXT=""
INLINE=false
SEM_OPT=false
SINGLE_NODE=false
RELAX_LOOPS=false
UNROLL_NONKERNELS=false
LOOP_PARTITION=false;
AUTO_PARALLEL=false;
SEED_WITH_JR=false
SKIP_TRACE=false
SKIP_TRACE_ANALYSIS=false
SKIP_FINAL_COMPILATION=false
CXX_DEBUG_ARGS=""
DEBUG_ENABLE=false
CPP=false
CC=clang
COMP_LEVEL=9
TRC_NAME="raw.trc"
TRACEHOME=`realpath ../../TraceAtlas/build`
COMPILERRT=`realpath ../lib/libclang_rt.builtins-aarch64.a`

LLVM_HOME="/usr/lib/llvm-9/bin"

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
      [--single-node]
        Produce an output testing binary with a single node consisting of all non-initialization basic blocks
      [--relax-loops]
        Utilize loop relaxation to repair split loops in final binary creation
      [--unroll-nonkernels]
        Attempt to unroll any loops that are not themselves kernels
      [--loop-partition]
        Partition the main function based on top level loops present. If a top level loop matches a kernel, it will be treated as such
      [--auto-parallelize]
        Attempt to parallelize nodes in the output JSON based on whether DagExtractor detects them as independent
      [--seed-with-jr]
        Seed the kernels in DAG generation with the output from JR rather than cartographer
      [--skip-trace]
        Skip the trace instrumentation and trace collection steps
      [--skip-trace-analysis]
        Skip the trace processing and kernel detection/DAG extraction steps
      [--skip-final-processing]
        Skip the application refactoring and final shared object compilation steps
      [--debug]
        Attempt to maintain debug symbols throughout the full transformation process and enable additional debug printing
      [--cpp]
        Process input files as with clang++ rather than clang
      [-l<lib_suffix>]
        Add a library that is required to compile this application for tracing and in shared object creation
        Ex: -lm for libmath
      [-tl<lib_suffix>]
        Add a library that is required to compile this application only for tracing, not shared object creation
      [-I<include_dir>]
        Add a path to the includes search path used during compilation identically to the -I flag on gcc/clang
      [--trace-compression <level>]
        Set the trace compression level to be used as an integer between 0 and 9. The default is 9.
      [--trace-name <name>]
        Set the name to be used for the trace file. Default is raw.trc.
      [-d|--dependency <dep_file>]
        Add a source file dependency that is required for compilation for tracing and in shared object creation
      [-od|--output-dependency <dep_file>]
        Add a source file dependency that is only required in shared object creation (i.e. arch specific libraries)
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
    --single-node)
      SINGLE_NODE=true
      shift 1
      ;;
    --relax-loops)
      RELAX_LOOPS=true
      shift 1
      ;;
    --unroll-nonkernels)
      UNROLL_NONKERNELS=true
      shift 1
      ;;
    --loop-partition)
      LOOP_PARTITION=true
      shift 1
      ;;
    --auto-parallelize)
      AUTO_PARALLEL=true
      shift 1;
      ;;
    --seed-with-jr)
      SEED_WITH_JR=true
      shift 1;
      ;;
    --skip-trace)
      SKIP_TRACE=true
      shift 1
      ;;
    --skip-trace-analysis)
      SKIP_TRACE_ANALYSIS=true
      shift 1
      ;;
    --skip-final-processing)
      SKIP_FINAL_COMPILATION=true
      shift 1
      ;;
    --debug)
      CXX_DEBUG_ARGS="-g3 -ggdb"
      DEBUG_ENABLE=true
      shift 1
      ;;
    --cpp)
      CC=clang++
      CPP=true
      shift 1
      ;;
    -l*)
      LIBS="$LIBS $1"
      shift 1
      ;;
    -tl*)
      TRACE_LIBS="${TRACE_LIBS} `echo $1 | sed 's/-tl/-l/g'`"
      shift 1
      ;;
    -I*)
      INCLUDES="${INCLUDES} $1"
      shift 1
      ;;
    --trace-compression)
      COMP_LEVEL=$2
      shift 2
      ;;
    --trace-name)
      TRC_NAME="$2" 
      shift 2
      ;;
    -d|--dependency)
      DEPS="$DEPS $2"
      shift 2
      ;;
    -od|--output-dependency)
      OUT_DEPS="${OUT_DEPS} $2"
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
          ARCH_FLAGS="-march=armv8-a -target aarch64-linux-gnu -I /usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu -I /usr/aarch64-linux-gnu/include"
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
        if [ ${CPP} = true ]; then
          C_FILE_NO_EXT="${1%.cpp}"
        else
          C_FILE_NO_EXT="${1%.c}"
        fi
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
IFS=' ' read -r -a OUT_DEPS <<< "${OUT_DEPS}"
IFS=' ' read -r -a LIBS <<< "$LIBS"
IFS=' ' read -r -a TRACE_LIBS <<< "${TRACE_LIBS}"
IFS=' ' read -r -a INCLUDES <<< "${INCLUDES}"

if [ "$SKIP_TRACE" = false ]; then
  echo "Stage: Initial compilation"
  ${LLVM_HOME}/${CC} -DENABLE_TRACING ${INCLUDES[@]} -S ${CXX_DEBUG_ARGS} -flto -fPIC -static ${C_FILE} -o output-${C_FILE_NO_EXT}.ll
  # Note: this stage might need all necessary functions to have the "always inline" attribute
  if [ "$INLINE" = true ]; then
    echo "Stage: Inlining function calls"
    ${LLVM_HOME}/opt -always-inline output-${C_FILE_NO_EXT}.ll -S -o output-${C_FILE_NO_EXT}.ll
  fi
  echo "Stage: Splitting allocas into standalone BB"
  ${LLVM_HOME}/opt -load $TRACEHOME/lib/AtlasPasses.so -SplitAllocas output-${C_FILE_NO_EXT}.ll -S -o output-${C_FILE_NO_EXT}.ll
  echo "Stage: Splitting Kernel exit/enters to ensure non-intersecting kernels"
  ${LLVM_HOME}/opt -load $TRACEHOME/lib/AtlasPasses.so -SplitKernExitEnter output-${C_FILE_NO_EXT}.ll -S -o output-${C_FILE_NO_EXT}.ll
  echo "Stage: Encoded annotation"
  ${LLVM_HOME}/opt -load $TRACEHOME/lib/AtlasPasses.so -EncodedAnnotate output-${C_FILE_NO_EXT}.ll -S -o output-${C_FILE_NO_EXT}-annotate.ll
  echo "Stage: Encoded trace instrumentation"
  ${LLVM_HOME}/opt -load $TRACEHOME/lib/AtlasPasses.so -EncodedTrace output-${C_FILE_NO_EXT}.ll -S -o output-${C_FILE_NO_EXT}-opt.ll
  
  echo "Stage: Tracer binary compilation"
  ${LLVM_HOME}/${CC} -static -fuse-ld=lld-9 \
            -lpthread -lz ${LIBS[@]} ${TRACE_LIBS[@]} $TRACEHOME/lib/libAtlasBackend.a \
            ${DEPS[@]} output-${C_FILE_NO_EXT}-opt.ll \
            -o ${C_FILE_NO_EXT}-tracer.out

  echo "Stage: Trace collection"
  TRACE_NAME=${TRC_NAME} TRACE_COMPRESSION=${COMP_LEVEL} ./${C_FILE_NO_EXT}-tracer.out
fi

if [ "$SKIP_TRACE_ANALYSIS" = false ]; then
  if [ -f outfile.txt ]; then
    echo "Removing previous outfile.txt"
    rm outfile.txt
  fi

  # Only run either cartographer or JR depending on which option was selected
  if [ "$SEED_WITH_JR" = false ]; then
    echo "Stage: Kernel extraction"
    $TRACEHOME/bin/cartographer -i ${TRC_NAME} -b output-${C_FILE_NO_EXT}-annotate.ll -k kernel-${C_FILE_NO_EXT}.json -L=true
    echo "Stage: Skipping JR execution"
    echo "{}" > kernel-jr-${C_FILE_NO_EXT}.json
    echo "Stage: DAG extraction"
    $TRACEHOME/bin/dagExtractor -t ${TRC_NAME} -k kernel-${C_FILE_NO_EXT}.json -o kernel-${C_FILE_NO_EXT}-dagExtractor.json
  else
    echo "Stage: Skipping kernel extraction"
    echo "{}" > kernel-${C_FILE_NO_EXT}.json
    echo "Stage: JR execution"
    $TRACEHOME/bin/JR -i ${TRC_NAME} -o kernel-jr-${C_FILE_NO_EXT}.json
    echo "Stage: Skipping DAG extraction"
    echo "{}" > kernel-${C_FILE_NO_EXT}-dagExtractor.json
  fi
fi

if [ "$SKIP_FINAL_COMPILATION" = false ]; then
  echo "Stage: Application refactoring/region outlining"
  #$TRACEHOME/bin/tik -f LLVM -t LLVM -j kernel-${C_FILE_NO_EXT}.json -o tik-${C_FILE_NO_EXT}.bc output-${C_FILE_NO_EXT}.ll
  $TRACEHOME/bin/kwrap -semantic-opt=${SEM_OPT} \
                       -single-node=${SINGLE_NODE} \
                       -relax-loops=${RELAX_LOOPS} \
                       -unroll-nonkernels=${UNROLL_NONKERNELS} \
                       -loop-partition=${LOOP_PARTITION} \
                       -auto-parallelize=${AUTO_PARALLEL} \
                       -seed-with-jr=${SEED_WITH_JR} \
                       -print-debug=${DEBUG_ENABLE} \
                       -a output-${C_FILE_NO_EXT}-annotate.ll \
                       -k kernel-${C_FILE_NO_EXT}.json \
                       -j kernel-jr-${C_FILE_NO_EXT}.json \
                       -d kernel-${C_FILE_NO_EXT}-dagExtractor.json \
                       -n ${C_FILE_NO_EXT}-${ARCH} \
                       -o output-${C_FILE_NO_EXT}-extracted.ll \
                       -o2 ${C_FILE_NO_EXT}-${ARCH}.json
  echo "Stage: Shared object compilation"
  if [ "$ARCH" = "x86" ]; then
    ${LLVM_HOME}/${CC} ${ARCH_FLAGS} ${CXX_DEBUG_ARGS} -shared -fPIC -fuse-ld=lld-9 ${LIBS[@]} ${DEPS[@]} output-${C_FILE_NO_EXT}-extracted.ll -o ${C_FILE_NO_EXT}-${ARCH}.so
  else
    ${LLVM_HOME}/${CC} ${ARCH_FLAGS} ${CXX_DEBUG_ARGS} -Woverride-module -shared -fPIC -fuse-ld=lld-9 ${LIBS[@]} $COMPILERRT ${DEPS[@]} ${OUT_DEPS[@]} output-${C_FILE_NO_EXT}-extracted.ll -o ${C_FILE_NO_EXT}-${ARCH}.so
  fi
fi
echo "Complete!"
