#include        <stdio.h>
#include        <stdint.h>
#include        <stdlib.h>
#include        <fcntl.h>
#include        <string.h>
#include        <time.h>
#include        <sys/types.h>
#include        <sys/mman.h>
#include        <math.h>
#include        <time.h>

#define TYPE     unsigned int
#define NUM_BYTES_TO_TRANSFER 1026 * 4
#define SEC2NANOSEC 1000000000

static char   attr[NUM_BYTES_TO_TRANSFER];

int           dma_control_fd;
unsigned int *dma_control_base_addr;

int           fd_udmabuf0;
unsigned long udmabuf0_phys_addr;

int           fd_udmabuf1;
unsigned long udmabuf1_phys_addr;

TYPE         *udmabuf0_base_addr;
TYPE         *udmabuf1_base_addr;

// Include File Containing DMA and udmabuf Initialization Routines
#include      "dma.h"
int N = 4;
int M = 64;
#define FIXED_POINT_FRACTIONAL_BITS 5

void hermitian(float *S, float *Si, float *Shermitian, float *Shermitianimag);
void hermitian_inverse(float *S, float *Si, float *Shermitian, float *Shermitianimag);

//###################################################################################
// Main Function
//###################################################################################

void __attribute__((constructor)) setup(void) {
  printf("[mmult] Initializing DMA buffers...\n");

  // Virtual Address to DMA Control Slave
  init_dma();

  // Virtual Address to udmabuf Buffer
  udmabuf0_base_addr = init_udmabuf0(fd_udmabuf0);
  udmabuf1_base_addr = init_udmabuf1(fd_udmabuf1);
  printf("[mmult] udmabuf0 and udmabuf1 initialized...\n");
}

void display(float *A, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", A[i * n + j]);
    }
    printf("\n");
  }
}

void mmult_fpga_kern(int *i, int *j, float *res1, float *res2, float *res3, float* res4, int *k, float **A, float **Binput, float *term1, float **Ai, float **Binputimag, float *term2, float *term3, float *term4, float **output, float **output_i) {

  printf("Dispatching to mmult accelerator...\n");

  float *B, *Bi;

  B = (float*) malloc(M * N * sizeof(float));
  Bi = (float*) malloc(M * N * sizeof(float));
  hermitian_inverse(*Binput, *Binputimag, B, Bi);

  // Clear buffers before application to avoid data corruption
  //for(int i = 0; i < NUM_BYTES_TO_TRANSFER / 4; i++) {
  //  udmabuf0_base_addr[i] = 0x0;
  //  udmabuf1_base_addr[i] = 0x0;
  //}

  // Reset DMA
  dma_write_reg(DMA_OFFSET_MM2S_CONTROL, 0x4);
  dma_write_reg(DMA_OFFSET_S2MM_CONTROL, 0x4);
  dma_wait_for_tx_idle();
  dma_wait_for_rx_idle();

  // Generate Inputs and Write it to DMA Source Buffer;
  TYPE *base_addr     = udmabuf0_base_addr;
  TYPE *base_addr_out = udmabuf1_base_addr;

  for (int i = 0; i < (NUM_BYTES_TO_TRANSFER /4-2)/4; i++) {
    base_addr[i*2] = (unsigned int) ((*A)[i]*1000);
    base_addr[i*2+1] = (unsigned int)(B[i]*1000);
    base_addr[512+i*2] = (unsigned int)((*Ai)[i]*1000);
    base_addr[512+i*2+1] = (unsigned int)(Bi[i]*1000);
  }

  base_addr[NUM_BYTES_TO_TRANSFER / 4-2] = 0;
  base_addr[NUM_BYTES_TO_TRANSFER / 4-1] = 0;

  //printf("before set up dma\n" );
  dma_write_reg(DMA_OFFSET_MM2S_SRCLWR,  udmabuf0_phys_addr);
  dma_write_reg(DMA_OFFSET_MM2S_CONTROL, 0x3);
  dma_write_reg(DMA_OFFSET_MM2S_LENGTH,  NUM_BYTES_TO_TRANSFER);
  //printf("before wait dma_wait_for_rx_complete\n");
  while ( (dma_control_base_addr[DMA_OFFSET_MM2S_STATUS] & 0x03) != 0x02);

  printf("[mmult] DMA sent data to IP ...\n");

  dma_write_reg(DMA_OFFSET_S2MM_SRCLWR,  udmabuf1_phys_addr);
  dma_write_reg(DMA_OFFSET_S2MM_CONTROL, 0x3);
  dma_write_reg(DMA_OFFSET_S2MM_LENGTH,  NUM_BYTES_TO_TRANSFER);

  while ( (dma_control_base_addr[DMA_OFFSET_S2MM_STATUS] & 0x03) != 0x02);

  printf("[mmult] IP sent data to DMA ...\n");

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      (*output_i)[i+j*4] =  -((float)((int)base_addr_out[997+i*4+j]))/1000000;
      (*output)[i+j*4] =  ((float)((int)base_addr_out[977+i*4+j]))/1000000;
    }
    //printf("\n");
  }

  printf("Accelerator execution complete!\n");
}

//void mmult_fpga(float A[64 * 4], float Ai[64 * 4], float B[64 * 4], float Bi[64 * 4], float output[16], float output_i[100]) {
void mmult_fpga(float *A, float *Ai, float *Binput, float *Binputimag, float *output, float *output_i) {
//void mmult_fpga(float *A, float *Ai, float *B, float *Bi, float *output, float *output_i) {

  int          fd_mm2s_buffer;
  int          fd_s2mm_buffer;
  unsigned int *mm2s_buffer_base_addr;
  unsigned int *s2mm_buffer_base_addr;

  TYPE         *udmabuf0_base_addr;
  TYPE         *udmabuf1_base_addr;

  TYPE         *ref_output;

  float *B, *Bi;

  B = (float*) malloc(M * N * sizeof(float));
  Bi = (float*) malloc(M * N * sizeof(float));
  //hermitian_inverse(Binput, Binputimag, B, Bi);
  hermitian_inverse(Binput, Binputimag, B, Bi);

  // Virtual Address to DMA Control Slave
  init_dma();

  // Virtual Address to udmabuf Buffer
  udmabuf0_base_addr = init_udmabuf0(fd_udmabuf0);
  udmabuf1_base_addr = init_udmabuf1(fd_udmabuf1);

  // Clear buffers before application to avoid data corruption
  for(int i = 0; i < NUM_BYTES_TO_TRANSFER / 4; i++) {
    udmabuf0_base_addr[i] = 0x0;
    udmabuf1_base_addr[i] = 0x0;
  }

  // Reset DMA
  dma_write_reg(DMA_OFFSET_MM2S_CONTROL, 0x4);
  dma_write_reg(DMA_OFFSET_S2MM_CONTROL, 0x4);
  dma_wait_for_tx_idle();
  dma_wait_for_rx_idle();

  // Generate Inputs and Write it to DMA Source Buffer;
  TYPE *base_addr     = udmabuf0_base_addr;
  TYPE *base_addr_out = udmabuf1_base_addr;

  for (int i = 0; i < (NUM_BYTES_TO_TRANSFER /4-2)/4; i++) {
    base_addr[i*2] = (unsigned int) (A[i]*1000);
    base_addr[i*2+1] = (unsigned int)(B[i]*1000);
    base_addr[512+i*2] = (unsigned int)(Ai[i]*1000);
    base_addr[512+i*2+1] = (unsigned int)(Bi[i]*1000);
  }

  base_addr[NUM_BYTES_TO_TRANSFER / 4-2] = 0;
  base_addr[NUM_BYTES_TO_TRANSFER / 4-1] = 0;

  printf("before set up dma\n" );
  dma_write_reg(DMA_OFFSET_MM2S_SRCLWR,  udmabuf0_phys_addr);
  dma_write_reg(DMA_OFFSET_MM2S_CONTROL, 0x3);
  dma_write_reg(DMA_OFFSET_MM2S_LENGTH,  NUM_BYTES_TO_TRANSFER);
  printf("before wait dma_wait_for_rx_complete\n");
  while ( (dma_control_base_addr[DMA_OFFSET_MM2S_STATUS] & 0x03) != 0x02);

  printf("[ INFO] DMA sent data to IP ...\n");

  dma_write_reg(DMA_OFFSET_S2MM_SRCLWR,  udmabuf1_phys_addr);
  dma_write_reg(DMA_OFFSET_S2MM_CONTROL, 0x3);
  dma_write_reg(DMA_OFFSET_S2MM_LENGTH,  NUM_BYTES_TO_TRANSFER);

  while ( (dma_control_base_addr[DMA_OFFSET_S2MM_STATUS] & 0x03) != 0x02);

  printf("[ INFO] IP sent data to DMA ...\n");

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      output_i[i+j*4] =  -((float)((int)base_addr_out[997+i*4+j]))/1000000;
      output[i+j*4] =  ((float)((int)base_addr_out[977+i*4+j]))/1000000;
    }
    //printf("\n");
  }

  // Clear buffers after application to avoid data corruption
  for(int i = 0; i < NUM_BYTES_TO_TRANSFER / 4; i++) {
    udmabuf0_base_addr [i] = 0x0;
    udmabuf1_base_addr[i] = 0x0;
  }
  clear_address(udmabuf0_base_addr, ref_output);
  clear_address(udmabuf1_base_addr, ref_output);
  //###################################################################################
  // Destroying file descriptors and virtual address maps
  //###################################################################################
  munmap(udmabuf0_base_addr, NUM_BYTES_TO_TRANSFER);
  munmap(udmabuf1_base_addr, NUM_BYTES_TO_TRANSFER);
  close(fd_udmabuf0);
  close(fd_udmabuf1);
  close_dma();
}

void mmult(float A[N * M], float Ai[N * M], float B[M * N], float Bi[M * N], float C[N * N], float Ci[N * N]) {
  int i, j;

  float Abuf[N][M], Aibuf[N][M], Bbuf[M][N], Bibuf[M][N];
#pragma HLS array_partition variable = Abuf block factor = 4 dim = 2
#pragma HLS array_partition variable = Aibuf block factor = 4 dim = 2

#pragma HLS array_partition variable = Bbuf block factor = 4 dim = 1
#pragma HLS array_partition variable = Bibuf block factor = 4 dim = 1

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
#pragma HLS PIPELINE
      Abuf[i][j] = A[i * M + j];
      Aibuf[i][j] = Ai[i * M + j];
      //printf("Abuf [%d %d] %f  %f   " , i , j ,Abuf[i][j],   Aibuf[i][j]);

    }
    printf("\n");
  }

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
#pragma HLS PIPELINE
      Bbuf[i][j] = B[i * N + j];
      Bibuf[i][j] = Bi[i * N + j];
      // printf("Bbuf [%d %d] %f  %f       " , i , j ,Bbuf[i][j],   Bibuf[i][j]);
    }

    printf("\n");
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE
      float result1 = 0, result2 = 0, result3 = 0, result4 = 0;
      for (int k = 0; k < M; k++) {
        float term1 = Abuf[i][k] * Bbuf[k][j];
        result1 += term1;
        float term2 = Aibuf[i][k] * Bibuf[k][j];
        result2 += term2;
        float term3 = Abuf[i][k] * Bibuf[k][j];
        result3 += term3;
        float term4 = Aibuf[i][k] * Bbuf[k][j];
        result4 += term4;
      }
      C[i * N + j] = result1 - result2;
      Ci[i * N + j] = result3 + result4;
    }
  }
}

//void hermitian(float S[N * M], float Si[N * M], float Shermitian[M * N], float Shermitianimag[M * N]) {
void hermitian(float *S, float *Si, float *Shermitian, float *Shermitianimag) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      Shermitian[j * N + i] = S[i * M + j];
      Shermitianimag[j * N + i] = -Si[i * M + j];
    }
  }
}

void hermitian_inverse(float *S, float *Si, float *Shermitian, float *Shermitianimag) {
  int i, j;

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      Shermitian[j * M + i] = S[i * N + j];
      Shermitianimag[j * M + i] = -Si[i * N + j];

    }
  }
}

int main() {
  // Initializing the Z signal which will have 4*64 dimension
  float *Z, *Zi;
  Z = (float *)malloc(N * M * sizeof(float));
  Zi = (float *)malloc(N * M * sizeof(float));

  // Now defining the jammer signal which will have the same dimensions as the message signal , The jammer is denoted
  // by S
  float *S, *Si;
  S = (float *)malloc(N * M * sizeof(float));
  Si = (float *)malloc(N * M * sizeof(float));

  //// Now computing the result from the first multiplication (Z*S^H)--> Result 1
  float *result1, *result1imag;
  result1 = (float *)malloc(N * N * sizeof(float));
  result1imag = (float *)malloc(N * N * sizeof(float));

  float *result1_fpga, *result1imag_fpga;
  result1_fpga = (float *)malloc(N * N * sizeof(float));
  result1imag_fpga = (float *)malloc(N * N * sizeof(float));

  float *Shermitian, *Shermitianimag;
  Shermitian = (float *)malloc(M * N * sizeof(float));
  Shermitianimag = (float *)malloc(M * N * sizeof(float));

//  for (int i = 0; i < N; i++) {
//    for (int j = 0; j < M; j++) {
//      Z[i * M + j] = rand() % 50*0.001;
//      Zi[i * M + j] = rand() % 50*0.001;
//      S[i * M + j] = rand() % 50*0.001;
//      Si[i * M + j] = rand() % 50*0.001;
//    }
//  }

  float minZ = 10000000, maxZ = -10000000, minZi = 10000000, maxZi = -10000000, minSher = 10000000, maxSher = -10000000, minSherimag = 10000000, maxSherimag = -10000000;

  FILE *fp;
  fp = fopen("josh_Z.txt", "r");
  for (int i = 0; i < 256; i++) {
    fscanf(fp, "%f", &Z[i]);
    if (Z[i] < minZ) minZ = Z[i];
    if (Z[i] > maxZ) maxZ = Z[i];
  }
  fclose(fp);
  fp = fopen("josh_Zi.txt", "r");
  for (int i = 0; i < 256; i++) {
    fscanf(fp, "%f", &Zi[i]);
    if (Zi[i] < minZi) minZi = Zi[i];
    if (Zi[i] > maxZi) maxZi = Zi[i];
  }
  fclose(fp);
  fp = fopen("josh_Shermitian.txt", "r");
  for (int i = 0; i < 256; i++) {
    fscanf(fp, "%f", &Shermitian[i]);
    if (Shermitian[i] < minSher) minSher = Shermitian[i];
    if (Shermitian[i] > maxSher) maxSher = Shermitian[i];
  }
  fclose(fp);
  fp = fopen("josh_Shermitianimag.txt", "r");
  for (int i = 0; i < 256; i++) {
    fscanf(fp, "%f", &Shermitianimag[i]);
    if (Shermitianimag[i] < minSherimag) minSherimag = Shermitianimag[i];
    if (Shermitianimag[i] > maxSherimag) maxSherimag = Shermitianimag[i];
  }
  fclose(fp);

  display(Z, 4, 64);
  printf("\n\n");
  display(Zi, 4, 64);
  printf("\n\n");
  display(Shermitian, 64, 4);
  printf("\n\n");
  display(Shermitianimag, 64, 4);
  printf("\n\n");

  printf("Z: [%f, %f], Zi: [%f, %f], Shermitian: [%f, %f], Shermitianimag: [%f, %f]\n", minZ, maxZ, minZi, maxZi, minSher, maxSher, minSherimag, maxSherimag);

  //hermitian(S, Si, Shermitian, Shermitianimag);
  mmult(Z, Zi, Shermitian, Shermitianimag, result1, result1imag);
  //mmult(josh_Z, josh_Zi, josh_Shermitian, josh_Shermitianimag, result1, result1imag);
  //mmult_fpga(Z, Zi, Shermitian, Shermitianimag, result1_fpga, result1imag_fpga);
  //mmult_fpga(Z, Zi, S, Si, result1_fpga, result1imag_fpga);
  mmult_fpga_kern(0, 0, 0, 0, 0, 0, 0, &Z, &Shermitian, 0, &Zi, &Shermitianimag, 0, 0, 0, &result1_fpga, &result1imag_fpga);
  //mmult_fpga_kern(0, 0, 0, 0, 0, 0, 0, (float**) &josh_Z, (float**) &josh_Shermitian, 0, (float**) &josh_Zi, (float**) &josh_Shermitianimag, 0, 0, 0, &result1_fpga, &result1imag_fpga);

  printf("fpga   \n");
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      printf("%f  %f     ",i,j, result1_fpga[i*4+j], result1imag_fpga[i*4+j]);

    }
    printf("\n");
  }

  printf("software  \n");
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      printf(" %f  %f      ", result1[i*4+j], result1imag[i*4+j]);

    }
    printf("\n");
  }
}