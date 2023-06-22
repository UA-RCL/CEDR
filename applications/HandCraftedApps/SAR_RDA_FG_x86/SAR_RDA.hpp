#pragma once

extern "C" void SAR_node_head(void);
extern "C" void SAR_RDA_LFM_1(void);
extern "C" void SAR_RDA_LFM_1(void);

/***** phase 1 ******/
extern "C" void SAR_RDA_1_FFT_cpu(void);
extern "C" void SAR_RDA_1_FFT_accel(void);
extern "C" void SAR_RDA_1_FFTSHIFT(void);
extern "C" void SAR_RDA_1_Mul(void);
extern "C" void SAR_RDA_1_IFFT_cpu(void);
extern "C" void SAR_RDA_1_IFFT_accel(void);

/***** phase 2 ******/
extern "C" void SAR_RDA_2_FFT_cpu(void);
extern "C" void SAR_RDA_2_FFT_accel(void);
extern "C" void SAR_RDA_2_FFTSHIFT(void);

/***** phase 3 ******/
extern "C" void SAR_RDA_3_Mul(void);
extern "C" void SAR_RDA_3_IFFT_cpu(void);
extern "C" void SAR_RDA_3_IFFT_accel(void);
extern "C" void SAR_RDA_3_FFTSHIFT(void);
extern "C" void SAR_RDA_3_Amplitude(void);
