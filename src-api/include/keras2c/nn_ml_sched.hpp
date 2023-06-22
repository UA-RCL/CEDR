#ifdef __cplusplus
extern "C" {
#endif
/*
 * nn_ml_sched.h
 * A pre-trained time-predictable artificial neural network, generated by Keras2C.py.
 * Based on ann.h written by keyan
 */

#ifndef NN_ML_SCHED_H_
#define NN_ML_SCHED_H_

#pragma once

// includes
#include "nn_types.hpp"

// NOTE: only values marked "*" may be changed with defined behaviour

// Network input defines
#define ML_SCHED_INPUT_DIM_COUNT 1
#define ML_SCHED_INPUT_DIM_0 19

// Network output defines
#define ML_SCHED_MAX_PL_LEN 1
#define ML_SCHED_OUTPUT_DIM_COUNT 1
#define ML_SCHED_OUTPUT_DIM_0 6

// Dense layer defines
#define ML_SCHED_L0_DENSE_NEURON_COUNT 16
#define ML_SCHED_L0_DENSE_USE_BIAS 1
#define ML_SCHED_L0_DENSE_ACTIVATION ACT_ENUM_RELU // *
#define ML_SCHED_L0_DENSE_DIM_COUNT 1
#define ML_SCHED_L0_DENSE_DIM_0 16

// Dense layer defines
#define ML_SCHED_L1_DENSE_NEURON_COUNT 16
#define ML_SCHED_L1_DENSE_USE_BIAS 1
#define ML_SCHED_L1_DENSE_ACTIVATION ACT_ENUM_RELU // *
#define ML_SCHED_L1_DENSE_DIM_COUNT 1
#define ML_SCHED_L1_DENSE_DIM_0 16

// Dense layer defines
#define ML_SCHED_L2_DENSE_NEURON_COUNT 16
#define ML_SCHED_L2_DENSE_USE_BIAS 1
#define ML_SCHED_L2_DENSE_ACTIVATION ACT_ENUM_RELU // *
#define ML_SCHED_L2_DENSE_DIM_COUNT 1
#define ML_SCHED_L2_DENSE_DIM_0 16

// Dense layer defines
#define ML_SCHED_L3_DENSE_NEURON_COUNT 6
#define ML_SCHED_L3_DENSE_USE_BIAS 1
#define ML_SCHED_L3_DENSE_ACTIVATION ACT_ENUM_SOFTMAX // *
#define ML_SCHED_L3_DENSE_DIM_COUNT 1
#define ML_SCHED_L3_DENSE_DIM_0 6

// NN weights struct
typedef struct {
  // Dense layer unit
  NN_NUM_TYPE ml_sched_l0_dense_weights[ML_SCHED_INPUT_DIM_0][ML_SCHED_L0_DENSE_NEURON_COUNT];
  NN_NUM_TYPE ml_sched_l0_dense_bias[ML_SCHED_L0_DENSE_NEURON_COUNT];
  // Dense layer unit
  NN_NUM_TYPE ml_sched_l1_dense_weights[ML_SCHED_L0_DENSE_DIM_0][ML_SCHED_L1_DENSE_NEURON_COUNT];
  NN_NUM_TYPE ml_sched_l1_dense_bias[ML_SCHED_L1_DENSE_NEURON_COUNT];
  // Dense layer unit
  NN_NUM_TYPE ml_sched_l2_dense_weights[ML_SCHED_L1_DENSE_DIM_0][ML_SCHED_L2_DENSE_NEURON_COUNT];
  NN_NUM_TYPE ml_sched_l2_dense_bias[ML_SCHED_L2_DENSE_NEURON_COUNT];
  // Dense layer unit
  NN_NUM_TYPE ml_sched_l3_dense_weights[ML_SCHED_L2_DENSE_DIM_0][ML_SCHED_L3_DENSE_NEURON_COUNT];
  NN_NUM_TYPE ml_sched_l3_dense_bias[ML_SCHED_L3_DENSE_NEURON_COUNT];
} NN_ML_SCHED;

// Static storage
extern NN_ML_SCHED nn_weights_ml_sched;
// Instance
typedef struct {
  NN_NUM_TYPE inputs[ML_SCHED_INPUT_DIM_0];
  NN_NUM_TYPE ml_sched_l0_dense_outputs[ML_SCHED_L0_DENSE_DIM_0];
  NN_NUM_TYPE ml_sched_l1_dense_outputs[ML_SCHED_L1_DENSE_DIM_0];
  NN_NUM_TYPE ml_sched_l2_dense_outputs[ML_SCHED_L2_DENSE_DIM_0];
  NN_NUM_TYPE outputs[ML_SCHED_MAX_PL_LEN][ML_SCHED_L3_DENSE_DIM_0];

  // Output pipeline
  int pl_index;
} NN_DATA_ML_SCHED;

// For the neural network
extern NN_DATA_ML_SCHED nn_data;
// Functions
void nn_init_ml_sched();
void run_ml_sched_l0_dense();
void run_ml_sched_l1_dense();
void run_ml_sched_l2_dense();
void run_ml_sched_l3_dense();

void nn_run_ml_sched(NN_DATA_ML_SCHED *nn_data);

#endif /* NN_ML_SCHED_H_ */

#ifdef __cplusplus
}
#endif