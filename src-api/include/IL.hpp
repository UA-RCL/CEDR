#pragma once

#include "config_manager.hpp"
#include "header.hpp"
#include "keras2c/nn_ml_sched.hpp"
#include "keras2c/nn_types.hpp"

// Previous signature accepted a "node" argument, updated signature does not
int predict(float features[19], int node);
// int predict(float features[19]);

extern "C" {
int nn_max_ml_sched(int dim_0, NN_NUM_TYPE *array);

void print_array_1d(int dim_0, NN_NUM_TYPE *array);

void normalizeExeTimes(NN_DATA_ML_SCHED *nn_data, int model);

void getPredecessor_PE_ID(int *ft, task_nodes *task, NN_DATA_ML_SCHED *nn_data, int model, int feature_index);

void getPredecessor_ID(int *ft, task_nodes *task, NN_DATA_ML_SCHED *nn_data, int model, int feature_index);

int getPrediction(ConfigManager &cedr_config, NN_DATA_ML_SCHED *nn_data, task_nodes *task, worker_thread *wt, int model);
}
