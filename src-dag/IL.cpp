#include "IL.hpp"

extern "C" {

// Get resource by doing max of probabilities of the softmax layer at the end of the DNN
int nn_max_ml_sched(int dim_0, NN_NUM_TYPE *array) {

  double *write_var = (double *)malloc(sizeof(double *) * ML_SCHED_OUTPUT_DIM_0);
  double tmp_max = 0;
  int ret = 0;
  for (int dim_0_index = 0; dim_0_index < dim_0; dim_0_index++) {
    write_var[dim_0_index] = TO_DBL(array[dim_0_index]);
    if (write_var[dim_0_index] > tmp_max) {
      tmp_max = write_var[dim_0_index];
      if (dim_0_index == 0) {
        ret = 0;
      } else if (dim_0_index == 1) {
        ret = 1;
      } else if (dim_0_index == 2) {
        ret = 2;
      } else if (dim_0_index == 3) {
        ret = 3;
      } else if (dim_0_index == 4) {
        ret = 4;
      } else if (dim_0_index == 5) {
        ret = 5;
      }
    }
  }
  // free(write_var);
  // printf("\n");
  return ret;
}

// Function to print values of an array for DNN
void print_array_1d(int dim_0, NN_NUM_TYPE *array) {
  double write_var;

  printf("(%d)\n", dim_0);
  for (int dim_0_index = 0; dim_0_index < dim_0; dim_0_index++) {
    write_var = TO_DBL(array[dim_0_index]);
    printf("%lf ", write_var);
  }
  printf("\n");
}

// Function to normalize execution times for IL-scheduler
// Currently, not in use
void normalizeExeTimes(NN_DATA_ML_SCHED *nn_data, int model) {
  float range = 0;
  float max = 0;
  float min = 0;
  int cnt = 0;
  for (int i = 0; i < 3; i++) {
    if (nn_data->inputs[6 + i] == 0) {
      nn_data->inputs[6 + i] = (float)10000;
    } else {
      cnt++;
    }
  }
  float valid[cnt];
  cnt = 0;
  for (int i = 0; i < 3; i++) {
    if (nn_data->inputs[6 + i] != 10000) {
      valid[cnt] = nn_data->inputs[6 + i];
      cnt++;
    }
  }
  max = *std::max_element(valid, valid + cnt);
  // max = std::max((float)max, nn_data->inputs[8]);
  min = *std::min_element(valid, valid + cnt);
  range = max - min;
  if (range == 0) {
    for (int i = 0; i < 3; i++) {
      nn_data->inputs[6 + i] = nn_data->inputs[6 + i] / min;
    }
  } else {
    for (int i = 0; i < 3; i++) {
      nn_data->inputs[6 + i] = (nn_data->inputs[6 + i] - min) / range;
    }
  }
  for (int i = 0; i < 3; i++) {
    if (nn_data->inputs[6 + i] > 1) {
      nn_data->inputs[6 + i] = (float)10;
    }
  }
}

/*
 * Returns the predecessor PE ID (if any) as the following format:
 * (2,10,10,10,10). Notice 10 is used to represent no predecessor ID
 */
void getPredecessor_PE_ID(int *ft, task_nodes *task, NN_DATA_ML_SCHED *nn_data, int model, int feature_index) {

  int i;

  for (i = 0; i < task->pred_count; i++) {
    if (task->pred[i]->assigned_resource_name == "Core 1") {
      ft[feature_index + i] = 0;
    } else if (task->pred[i]->assigned_resource_name == "Core 2") {
      ft[feature_index + i] = 1;
    } else if (task->pred[i]->assigned_resource_name == "Core 3") {
      ft[feature_index + i] = 2;
    } else if (task->pred[i]->assigned_resource_name == "MMULT 1") {
      ft[feature_index + i] = 5;
    } else if (task->pred[i]->assigned_resource_name == "FFT 1") {
      ft[feature_index + i] = 3;
    } else if (task->pred[i]->assigned_resource_name == "FFT 2") {
      ft[feature_index + i] = 4;
    }

    nn_data->inputs[feature_index + i] = FROM_DBL(ft[feature_index + i]);
  }
  for (i = task->pred_count; i < 1; i++) {
    ft[feature_index + i] = 10;
    nn_data->inputs[feature_index + i] = FROM_DBL(ft[feature_index + i]);
  }

  // for (i=0;i<5;i++)
  //     printf("[DEBUG] getPredecessor_PE_ID #%d: %d\n",14+i, ft[14+i]);
}

/*
 * Returns the predecessor ID (if any) as the following format:
 * (pred_id,10000,10000,10000,10000). Notice 10000 is used to represent no predecessor
 */
void getPredecessor_ID(int *ft, task_nodes *task, NN_DATA_ML_SCHED *nn_data, int model, int feature_index) {
  int i;

  for (i = 0; i < task->pred_count; i++) {
    ft[feature_index + i] = task->pred[i]->task_id;
    nn_data->inputs[feature_index + i] = FROM_DBL(ft[feature_index + i]);
  }
  for (i = task->pred_count; i < 1; i++) {
    ft[feature_index + i] = 50;
    nn_data->inputs[feature_index + i] = FROM_DBL(ft[feature_index + i]);
  }

  // for (i=0;i<5;i++)
  //   printf("[DEBUG] getPredecessor_ID #%d: %d\n",8+i, ft[8+i]);
}

/*
 * Sets all the input model features and runs inference of the. Returns the CPU assigned
 * for that task
 */
int getPrediction(ConfigManager &cedr_config, NN_DATA_ML_SCHED *nn_data, task_nodes *task, worker_thread *wt, int model) {
  int ret = 0;
  struct timespec curr_sys_time, aux1, aux2;
  long long tmp, sch_time;
  int *feature_data = (int *)malloc(sizeof(int) * ML_SCHED_INPUT_DIM_0);
  int feature_index = 0;

  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_sys_time);
  tmp = (long long)curr_sys_time.tv_sec * SEC2NANOSEC + (long long)curr_sys_time.tv_nsec;
  // PEs available times,. Units should be in microseconds
  // TODO: We can use cedr_config.getTotalResources() here, but that has the risk of breaking IL in the future if
  // another resource type is added
  for (feature_index = 0; feature_index < cedr_config.getResourceArray()[(uint8_t) resource_type::cpu] + cedr_config.getResourceArray()[(uint8_t) resource_type::fft] +
                                              cedr_config.getResourceArray()[(uint8_t) resource_type::mmult];
       feature_index++) {
    (tmp > (long long)wt[feature_index].thread_avail_time) ? (feature_data[feature_index] = 0)
                                                           : (feature_data[feature_index] = ((int)(wt[feature_index].thread_avail_time - tmp)) * 10);
    nn_data->inputs[feature_index] = FROM_DBL(feature_data[feature_index]);
    // nn_data->inputs[feature_index] = wt[feature_index].todo_task_dequeue.size();
  }

  // Populate task execution time
  if (task->supported_resources.size() == 1) {
    nn_data->inputs[feature_index + 0] = 0;
    nn_data->inputs[feature_index + 1] = 10;
    nn_data->inputs[feature_index + 2] = 10;
  } else {
    nn_data->inputs[feature_index + 0] = 1;
    if (task->supported_resources.count(resource_type::fft) != 0) {
      nn_data->inputs[feature_index + 1] = 0;
      nn_data->inputs[feature_index + 2] = 10;
    } else {
      nn_data->inputs[feature_index + 1] = 10;
      nn_data->inputs[feature_index + 2] = 0;
    }
  }
  feature_index = feature_index + 3;

  // feature_data[6] = (int)task->estimated_execution[ON_CPU];
  // nn_data->inputs[6] = FROM_DBL(feature_data[6]);

  // feature_data[7] = (int)task->estimated_execution[ON_ACC];
  // nn_data->inputs[7] = FROM_DBL(feature_data[7]);
  // 	// Need to be fixed after adding MMULT
  // 	feature_data[8] = 0;
  //
  // 	nn_data->inputs[8] = FROM_DBL(feature_data[8]);
  //
  // 	normalizeExeTimes(nn_data, model);

  // Predecessor/s ID/s
  getPredecessor_ID(feature_data, task, nn_data, model, feature_index);

  // Predecessor/s assigned PE/s ID/s
  getPredecessor_PE_ID(feature_data, task, nn_data, model, feature_index);
  if (model == 0) {
    nn_init_ml_sched();
    clock_gettime(CLOCK_MONOTONIC_RAW, &aux1);
    nn_run_ml_sched(nn_data);
    clock_gettime(CLOCK_MONOTONIC_RAW, &aux2);
    sch_time = (((long long)aux2.tv_sec * SEC2NANOSEC + (long long)aux2.tv_nsec)) - (((long long)aux1.tv_sec * SEC2NANOSEC + (long long)aux1.tv_nsec));
    LOG_DEBUG << "DNN scheduling overhead: " << sch_time << "ns";
    ret = nn_max_ml_sched(ML_SCHED_OUTPUT_DIM_0, nn_data->outputs[0]);
    LOG_DEBUG << "PE assigned to task " << std::string(task->task_name) << " is: " << ret;
  } else {
    clock_gettime(CLOCK_MONOTONIC_RAW, &aux1);
    // Previous IL function signature accepted a "node" input
    ret = predict(nn_data->inputs, 0);
    // Updated IL predict signature
    // ret = predict(nn_data->inputs);
    clock_gettime(CLOCK_MONOTONIC_RAW, &aux2);
    sch_time = (((long long)aux2.tv_sec * SEC2NANOSEC + (long long)aux2.tv_nsec)) - (((long long)aux1.tv_sec * SEC2NANOSEC + (long long)aux1.tv_nsec));
    LOG_DEBUG << "DT scheduling overhead: " << sch_time << "ns";
    LOG_DEBUG << "PE assigned to task " << std::string(task->task_name) << " is: " << ret;
  }

  free(feature_data);
  return ret;
}

}; // END of extern "C"
