#ifndef __EQUALIZER_H__
#define __EQUALIZER_H__

void init_equalizer();
void equalization(float pilotdata_rx[], float offt[], float out_fft_c2f[], int mode);

#endif
