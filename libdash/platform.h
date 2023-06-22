#pragma once

#include <cstdlib>
#include <cstdint>

#define DASH_ZCU102_2020_2 0
#define DASH_JETSONAGX     1
#define DASH_VCU128        2
#define DASH_HTG960        3
#define DASH_ZCU102_10FFT_2MMULT_2ZIP_HWSCHEDULER 4

//#define DASH_PLATFORM DASH_ZCU102_2020_2
#define DASH_PLATFORM DASH_ZCU102_10FFT_2MMULT_2ZIP_HWSCHEDULER
#define GLOBAL_UDMABUF_SIZE 1048576

#if DASH_PLATFORM == DASH_ZCU102_2020_2
  #pragma message("=*=*= Building for ZCU102 =*=*=")

  #define FFT_UDMABUF_NUM  0
  #define FFT_UDMABUF_SIZE GLOBAL_UDMABUF_SIZE
  // If defined, we assume that we can configure the FFT IP through AXI GPIO
  // Otherwise, we assume that an fft_axi_config IP is being used
  #define FFT_CONFIG_VIA_GPIO
  #define FFT_GPIO_CONFIG_DELAY 10
  #define FFT_CONTROL_BASE_ADDRS    ((uint32_t[]) {0xA0000000, 0xA0030000})
  #define FFT_DMA_CTRL_BASE_ADDRS   ((uint32_t[]) {0xA0010000, 0xA0040000})
  #define FFT_GPIO_RESET_BASE_ADDRS ((uint32_t[]) {0xA0020000, 0xA0050000})

  #define GEMM_UDMABUF_NUM  1
  #define GEMM_UDMABUF_SIZE GLOBAL_UDMABUF_SIZE
  #define GEMM_DMA_CTRL_BASE_ADDRS  ((uint32_t[]) {0xA0060000, 0xA0070000})

#elif DASH_PLATFORM == DASH_ZCU102_10FFT_2MMULT_2ZIP_HWSCHEDULER
  #pragma message("=*=*= Building for ZCU102 with 10FFTs, 2 MMULTs, 2ZIPs, and HWSCHEDULER =*=*=")

  #define FFT_UDMABUF_NUM  0
  #define FFT_UDMABUF_SIZE GLOBAL_UDMABUF_SIZE
  // If defined, we assume that we can configure the FFT IP through AXI GPIO
  // Otherwise, we assume that an fft_axi_config IP is being used
  #define FFT_CONFIG_VIA_GPIO
  #define FFT_GPIO_CONFIG_DELAY 10
  #define FFT_CONTROL_BASE_ADDRS    ((uint32_t[]) {0xA0020000, 0xA0050000, 0xA0080000, 0xA00B0000, 0xA00E0000, 0xB0020000, 0xB0050000, 0xB0080000, 0xB00B0000, 0xB00E0000})
  #define FFT_DMA_CTRL_BASE_ADDRS   ((uint32_t[]) {0xA0010000, 0xA0040000, 0xA0070000, 0xA00A0000, 0xA00D0000, 0xB0010000, 0xB0040000, 0xB0070000, 0xB00A0000, 0xB00D0000})
  #define FFT_GPIO_RESET_BASE_ADDRS ((uint32_t[]) {0xA0030000, 0xA0060000, 0xA0090000, 0xA00C0000, 0xA00F0000, 0xB0030000, 0xB0060000, 0xB0090000, 0xB00C0000, 0xB00F0000})

  #define GEMM_UDMABUF_NUM  1
  #define GEMM_UDMABUF_SIZE GLOBAL_UDMABUF_SIZE
  #define GEMM_DMA_CTRL_BASE_ADDRS ((uint32_t[]) {0xA0100000, 0xB0100000})

  #define ZIP_UDAMBUF_NUM 4
  #define ZIP_UDAMBUF_SIZE GLOBAL_UDMABUF_SIZE
  #define ZIP_DMA_CTRL_BASE_ADDRS   ((uint32_t[]) {0xA0120000, 0xB0110000})
  #define ZIP_CONTROL_BASE_ADDRS     ((uint32_t[]) {0xA0000000, 0xB0000000})
  // Offsets for ZIP config AXI-Lite ports
  #define ZIP_OP_OFFSET   0x18
  #define ZIP_SIZE_OFFSET 0x10

#elif DASH_PLATFORM == DASH_JETSONAGX
  #pragma message("=*=*= Building for Jetson AGX Xavier =*=*=")
  // No control registers are defined here either, but it's because we don't interact with the GPU over userspace MMIO. Not an issue.
#elif DASH_PLATFORM == DASH_VCU128
  #pragma message("=*=*= Building for VCU-128 =*=*=")
  #error "No accelerator control registers are defined for this platform!"
#elif DASH_PLATFORM == DASH_HTG960
  #pragma message("=*=*= Building for HTG-960 =*=*=")
  #error "No accelerator control registers are defined for this platform!"
#else
  #error "Unknown DASH_PLATFORM selected!";
#endif
