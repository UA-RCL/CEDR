#ifdef __cplusplus
extern "C" {
#endif
/*
 * nn_types.h
 * A pre-trained time-predictable artificial neural network, generated by Keras2C.py.
 * Based on ann.h written by keyan
 */

#ifndef NN_TYPES_H_
#define NN_TYPES_H_

#pragma once

// includes
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// NOTE: only values marked "*" may be changed with defined behaviour

// macros
#define DO_PRAGMA_(x) _Pragma(#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)

#define MAX_TEST_COUNT 10000 // * maximum number of test cases in input files
#define LINEAR_A 1           // * integer multiplier in linear activation

// Available data types used for NN arithmetic (enum)
#define NN_FLOAT 0
#define NN_DOUBLE 1
#define NN_FIX16 2

// Data type selection (modify this macro to change entire NN data type)
#define NN_TYPE NN_FLOAT // *

// Operand macros
#if NN_TYPE == NN_FLOAT
#define NN_NUM_TYPE float
#elif NN_TYPE == NN_DOUBLE
#define NN_NUM_TYPE double
#elif NN_TYPE == NN_FIX16
#include "libfixmath/fix16.h"
#define NN_NUM_TYPE fix16_t
#endif

// Operator macros
#if NN_TYPE == NN_FLOAT || NN_TYPE == NN_DOUBLE // builtin data types use the same rules
#define ADD(a, b) ((a) + (b))
#define SUB(a, b) ((a) - (b))
#define MUL(a, b) ((a) * (b))
#define DIV(a, b) ((a) / (b))
#define EXP(x) exp(x)
#define FROM_DBL(x) ((NN_NUM_TYPE)(x))
#define TO_DBL(x) ((double)(x))
#define FROM_INT(x) ((NN_NUM_TYPE)(x))

// custom activations
#define tanh(x) tanh(x) // use instance in math.h
#elif NN_TYPE == NN_FIX16
#define ADD(a, b) fix16_add(a, b)
#define SUB(a, b) fix16_sub(a, b)
#define MUL(a, b) fix16_mul(a, b)
#define DIV(a, b) fix16_div(a, b)
#define EXP(x) fix16_exp(x)
#define FROM_DBL(x) fix16_from_dbl(x)
#define TO_DBL(x) fix16_to_dbl(x)
#define FROM_INT(x) fix16_from_int(x)
#endif

// activation functions (used as fallback, if no type-specific versions were defined)
#ifndef tanh
#define tanh(x) DIV(SUB(FROM_INT(1), EXP(MUL(FROM_INT(-2), x))), ADD(FROM_INT(1), EXP(MUL(FROM_INT(-2), x))))
#endif
#ifndef sigmoid
#define sigmoid(x) DIV(FROM_INT(1), ADD(FROM_INT(1), EXP(MUL(FROM_INT(-1), x))))
#endif
#ifndef relu
#define relu(x) ((x) > 0 ? (x) : 0)
#endif
#ifndef linear
#define linear(x) MUL(FROM_INT(LINEAR_A), x)
#endif

// Activation enum
#define ACT_ENUM_SIGMOID 0
#define ACT_ENUM_TANH 1
#define ACT_ENUM_RELU 2
#define ACT_ENUM_LINEAR 3
#define ACT_ENUM_SOFTMAX 4

#endif /* NN_TYPES_H_ */

#ifdef __cplusplus
}
#endif