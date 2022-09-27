#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "dma.h"
#include "zip.h"

#define SEC2NANOSEC 1000000000

static volatile unsigned int* zip_control_base_addr[NUM_ZIPS];
static volatile unsigned int* dma_control_base_addr[NUM_ZIPS];
static volatile unsigned int* udmabuf_base_addr;
static uint64_t               udmabuf_phys_addr;

void __attribute__((constructor)) setup_zip(void) {
  LOG("[zip] Running ZIP constructor\n");

  for(uint8_t i = 0; i < NUM_ZIPS; i++){
    LOG("[zip] Initializing ZIP DMA (0x%x)\n", ZIP_DMA_CTRL_BASE_ADDRS[i]);
    dma_control_base_addr[i] = init_dma(ZIP_DMA_CTRL_BASE_ADDRS[i]);
    reset_dma(dma_control_base_addr[i]);

    LOG("[zip] Initializing ZIP\n");
    zip_control_base_addr[i] = init_zip(ZIP_CONTROL_BASE_ADDRS[i]);
  }
  
  LOG("[zip] Initializing udmabuf\n");
  init_udmabuf(ZIP_UDAMBUF_NUM, ZIP_UDAMBUF_SIZE, &udmabuf_base_addr, &udmabuf_phys_addr);
  LOG("[zip] udmabuf base address is 0x%p\n", udmabuf_base_addr);

  LOG("[zip] ZIP constructor complete!\n");
}

void __attribute__((destructor)) teardown_zip(void) {
  LOG("[zip] Running ZIP destructor\n");
  close_udmabuf(udmabuf_base_addr, ZIP_UDAMBUF_SIZE);

  for (uint8_t i = 0; i < NUM_ZIPS; i++){
    close_zip(zip_control_base_addr[i]);
    close_dma(dma_control_base_addr[i]);
  }
  LOG("[zip] ZIP destructor complete!\n");
}

void zip_accel(zip_re_type* input0, zip_re_type* input1, zip_re_type* output, size_t size, int op, uint8_t resource_idx) {
  LOG("[zip-%u] Running zip for vector size %ld on the FPGA\n", resource_idx, size);
  struct timespec start_accel {};
  struct timespec end_accel {};

  volatile unsigned int *dma_control_base = dma_control_base_addr[resource_idx];
  volatile unsigned int *zip_control_base = zip_control_base_addr[resource_idx];

  volatile unsigned int *udmabuf_base = udmabuf_base_addr + (resource_idx * (UDMABUF_PARTITION_SIZE / sizeof(unsigned int)));
  uint64_t udmabuf_phys = udmabuf_phys_addr + (resource_idx * UDMABUF_PARTITION_SIZE);

  LOG("[zip-%u] zip control reg is %p\n", resource_idx, zip_control_base);
  LOG("[zip-%u] dma control reg is %p\n", resource_idx, dma_control_base);
  LOG("[zip-%u] udmabuf base address is 0x%x\n", resource_idx, (unsigned int) *udmabuf_base);
  LOG("[zip-%u] udmabuf phys address is 0x%lx\n", resource_idx, udmabuf_phys);

  size_t i;

  LOG("[zip-%u] Resetting DMA engine\n", resource_idx);
  reset_dma(dma_control_base);
  LOG("[zip-%u] Resetting DMA engine done!\n", resource_idx);

  switch(op){
    case 0: // ZIP_ADD
      LOG("[zip-%u] Configuring ZIP as ADD\n", resource_idx);
      break;
    case 1: // ZIP_SUB
      LOG("[zip-%u] Configuring ZIP as SUB\n", resource_idx);
      break;
    case 2: // ZIP_MULT
      LOG("[zip-%u] Configuring ZIP as MULT\n", resource_idx);
      break;
    case 3: // ZIP_DIV
      LOG("[zip-%u] Configuring ZIP as DIV\n", resource_idx);
      break;
    case 4: // ZIP_COMP_MULT
      LOG("[zip-%u] Configuring ZIP as COMP_MULT\n", resource_idx);
      size = size*2;
      break;
  }
  config_zip_op(zip_control_base, op);

  LOG("[zip-%u] Configuring ZIP with size %ld\n", resource_idx, size);
  config_zip_size(zip_control_base, size);

  LOG("[zip-%u] Copying input0 buffer to udmabuf (udmabuf_base: %p, input0: %p)\n", resource_idx, udmabuf_base, input0);
  //memcpy((float* ) udmabuf0_base, input0, size * sizeof(float));
  for (size_t i = 0; i < size; i++) {
    ((float*)udmabuf_base)[i] = (float) input0[i];
  }
  for (size_t i = 0; i < size; i++) {
    ((float*)udmabuf_base)[size+i] = (float) input1[i]; 
  }

  LOG("[zip-%u] Calling setup_rx for dma\n", resource_idx);
  fflush(stdout);
  setup_rx(dma_control_base, udmabuf_phys + (2 * size * sizeof(float)), size * sizeof(float));

  clock_gettime(CLOCK_MONOTONIC_RAW, &start_accel);

  LOG("[zip-%u] Calling setup_tx for dma\n", resource_idx);
  setup_tx(dma_control_base, udmabuf_phys, 2 * size * sizeof(float));
  
  LOG("[zip-%u] Waiting for RX to complete for dma0\n", resource_idx);
  dma_wait_for_rx_complete(dma_control_base);
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_accel);
  
  LOG("[zip-%u] ZIP accelerator execution time (ns): %lf\n", resource_idx,
         ((double)end_accel.tv_sec * SEC2NANOSEC + (double)end_accel.tv_nsec) - ((double)start_accel.tv_sec * SEC2NANOSEC + (double)start_accel.tv_nsec));

  LOG("[zip-%u] Memcpy output back\n", resource_idx);
  //memcpy(output,(unsigned int *) &((float*)udmabuf0_base)[size], size * sizeof(float));
//  memcpy(output, ((unsigned int *)((char*)udmabuf_base + (2 * size * sizeof(float)))), size * sizeof(float));
  for (size_t i = 0; i < size; i++) {
    output[i] = (float) (((float*)udmabuf_base)[2*size+i]);
  }
 
  LOG("[zip-%u] Finished ZIP on the FPGA\n", resource_idx);
}

extern "C" void DASH_ZIP_flt_zip(dash_re_flt_type** x, dash_re_flt_type** y, dash_re_flt_type** z, size_t* size, int* op, uint8_t resource_idx){
  if(sizeof(dash_re_flt_type) == sizeof(zip_re_type)){
    zip_accel(*x, *y, *z, *size, *op, resource_idx);
  }
  else{
    zip_re_type* inp0 = (zip_re_type*) malloc((*size) * sizeof(zip_re_type));
    zip_re_type* inp1 = (zip_re_type*) malloc((*size) * sizeof(zip_re_type));
    zip_re_type* out = (zip_re_type*) malloc((*size) * sizeof(zip_re_type));

    for (size_t i = 0; i < (*size); i++) {
      inp0[i] = (zip_re_type) ((*x)[i]);
      inp1[i] = (zip_re_type) ((*y)[i]);
    }

    zip_accel(inp0, inp1, out, *size, *op, resource_idx);

    for (size_t i = 0; i < (*size); i++) {
      (*z)[i] = (zip_re_type) (out[i]);
    }

    free(inp0);
    free(inp1);
    free(out);
  }
}

#if defined(__ZIP_ENABLE_MAIN)
uint32_t _check_zip_result_(zip_re_type * a, zip_re_type *b, zip_re_type *zip_actual, zip_re_type *zip_expected, size_t size) {
  int error_count = 0;
  double diff;
  float c, d;

  LOG("[zip] Checking actual versus expected output\n");
  for (size_t i = 0; i < size; i++) {
    c = zip_expected[i];
    d = (zip_re_type) zip_actual[i];

    diff = std::abs(c - d) / c * 100;

    if (diff > 0.01) {
      LOG("[zip] ERROR %ld - In0 = %lf, In1 = %lf, Expected = %lf, Hardware ZIP = %lf, size = %ld\n", i, a[i], b[i], c, d, size);
      error_count++;
    }
  }

  if (error_count == 0) {
    LOG("[zip] ZIP Passed!\n");
    return 0;
  } else {
    fprintf(stderr, "[zip] ZIP Failed!\n");
    return 1;
  }
}

void _generate_zip_test_values_(zip_re_type *zip_input0, zip_re_type *zip_input1, zip_re_type *zip_expected, size_t h_len, int op) {
  for(int i = 0; i < h_len; i++){
    switch(op){
      case 0: // ZIP_ADD
        zip_expected[i] = (zip_re_type) (zip_input0[i] + zip_input1[i]);
        break;
      case 1: // ZIP_SUB
        zip_expected[i] = (zip_re_type) (zip_input0[i] - zip_input1[i]);
        break;
      case 2: // ZIP_MULT
        zip_expected[i] = (zip_re_type) (zip_input0[i] * zip_input1[i]);
        break;
      case 3: // ZIP_DIV
        zip_expected[i] = (zip_re_type) (zip_input0[i] / zip_input1[i]);
        break;
      case 4: // ZIP_COMP_MULT
        zip_expected[i] = (zip_re_type) (zip_input0[i]*zip_input1[i] - zip_input0[i+1]*zip_input1[i+1]);
        zip_expected[i+1] = (zip_re_type) (zip_input0[i+1]*zip_input1[i] + zip_input0[i]*zip_input1[i+1]);
        i++;
        break;
    }
  }
}

int main() {
  zip_re_type *A;
  zip_re_type *B;
  zip_re_type *C;
  zip_re_type *C_expected;
  int op;
  
  size_t test_len = 1024;

  A = (zip_re_type*) calloc(test_len*2, sizeof(zip_re_type));
  B = (zip_re_type*) calloc(test_len*2, sizeof(zip_re_type));
  C = (zip_re_type*) calloc(test_len*2, sizeof(zip_re_type));
  C_expected = (zip_re_type*) calloc(test_len*2, sizeof(zip_re_type));

  int base = 4;
  for (int j=0; j<100; j++){
  for (uint8_t zip_num = 0; zip_num < NUM_ZIPS; zip_num++) {
  for (int i = 0; i < test_len*2; i++){
    A[i] = (zip_re_type) base*(i+1)/100;
    B[i] = (zip_re_type) (base*2)*(i+1)/100;
  }
    LOG("[zip-%u] Beginning tests on accelerator %u\n", zip_num, zip_num);
    // Test ZIP add
/*    op = 0;
    LOG("[zip-%u] Testing ZIP Add...\n", zip_num);
    _generate_zip_test_values_(A, B, C_expected, test_len, 0);
    DASH_ZIP_zip(&A, &B, &C, &test_len, &op, zip_num);
    _check_zip_result_(A, B, C, C_expected, test_len);

     // Test ZIP sub
    op = 1;
    LOG("[zip-%u] Testing ZIP Sub...\n", zip_num);
    _generate_zip_test_values_(A, B, C_expected, test_len, 1);
    DASH_ZIP_zip(&A, &B, &C, &test_len, &op, zip_num);
    _check_zip_result_(A, B, C, C_expected, test_len);

    // Test ZIP mult
    op = 2;
    LOG("[zip-%u] Testing ZIP Mult...\n", zip_num);
    _generate_zip_test_values_(A, B, C_expected, test_len, 2);
    DASH_ZIP_zip(&A, &B, &C, &test_len, &op, zip_num);
    _check_zip_result_(A, B, C, C_expected, test_len);

    // Test ZIP div
    op = 3;
    LOG("[zip-%u] Testing ZIP Div...\n", zip_num);
    _generate_zip_test_values_(A, B, C_expected, test_len, 3);
    DASH_ZIP_zip(&A, &B, &C, &test_len, &op, zip_num);
    _check_zip_result_(A, B, C, C_expected, test_len);
*/
    // Test ZIP comp mult
    op = 4;
    LOG("[zip-%u] Testing ZIP Comp Mult...\n", zip_num);
    _generate_zip_test_values_(A, B, C_expected, test_len*2, 4);
    DASH_ZIP_flt_zip(&A, &B, &C, &test_len, &op, zip_num);
    _check_zip_result_(A, B, C, C_expected, test_len*2);
  }
  base = base*2;
  }
  free(A); free(B); free(C); free(C_expected);
}
#endif
