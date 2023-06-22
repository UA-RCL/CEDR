#define _GNU_SOURCE
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "baseband_lib.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <math.h>
#include <time.h>
#include "common.h"

#include <fft_hs.h>
#include <sched.h>

unsigned int hex_data;
comp_t tx_ifft_data[128];
comp_t tx_fft_data[128];

void get_tx_ifft_data() {

hex_data = 0xbd174b23; tx_ifft_data[0].real    = *(float *)&hex_data; 
hex_data = 0xbd26d28f; tx_ifft_data[0].imag    = *(float *)&hex_data; 
hex_data = 0xbdd56bed; tx_ifft_data[1].real    = *(float *)&hex_data; 
hex_data = 0x3e382817; tx_ifft_data[1].imag    = *(float *)&hex_data; 
hex_data = 0x3c88c17a; tx_ifft_data[2].real    = *(float *)&hex_data; 
hex_data = 0xbd8bb1ea; tx_ifft_data[2].imag    = *(float *)&hex_data; 
hex_data = 0xbd01ab1c; tx_ifft_data[3].real    = *(float *)&hex_data; 
hex_data = 0xbdebc565; tx_ifft_data[3].imag    = *(float *)&hex_data; 
hex_data = 0x3d3d8aa6; tx_ifft_data[4].real    = *(float *)&hex_data; 
hex_data = 0xbda1475e; tx_ifft_data[4].imag    = *(float *)&hex_data; 
hex_data = 0xbd0118fc; tx_ifft_data[5].real    = *(float *)&hex_data; 
hex_data = 0x3d8e2b42; tx_ifft_data[5].imag    = *(float *)&hex_data; 
hex_data = 0xbbaf1d79; tx_ifft_data[6].real    = *(float *)&hex_data; 
hex_data = 0x3d9062c0; tx_ifft_data[6].imag    = *(float *)&hex_data; 
hex_data = 0x3c1b1582; tx_ifft_data[7].real    = *(float *)&hex_data; 
hex_data = 0xbd9d7b0c; tx_ifft_data[7].imag    = *(float *)&hex_data; 
hex_data = 0x3ddf8737; tx_ifft_data[8].real    = *(float *)&hex_data; 
hex_data = 0xbd45bbcc; tx_ifft_data[8].imag    = *(float *)&hex_data; 
hex_data = 0x3ddf4a55; tx_ifft_data[9].real    = *(float *)&hex_data; 
hex_data = 0x3da49d63; tx_ifft_data[9].imag    = *(float *)&hex_data; 
hex_data = 0x3dc3666b; tx_ifft_data[10].real   = *(float *)&hex_data; 
hex_data = 0x3daf1148; tx_ifft_data[10].imag   = *(float *)&hex_data; 
hex_data = 0xbd0e0f54; tx_ifft_data[11].real   = *(float *)&hex_data; 
hex_data = 0x3c8c5406; tx_ifft_data[11].imag   = *(float *)&hex_data; 
hex_data = 0x3d14cf0c; tx_ifft_data[12].real   = *(float *)&hex_data; 
hex_data = 0x3d46e90f; tx_ifft_data[12].imag   = *(float *)&hex_data; 
hex_data = 0x3cb961c2; tx_ifft_data[13].real   = *(float *)&hex_data; 
hex_data = 0x3d256a61; tx_ifft_data[13].imag   = *(float *)&hex_data; 
hex_data = 0x3d0f0de5; tx_ifft_data[14].real   = *(float *)&hex_data; 
hex_data = 0x3c3b0cf0; tx_ifft_data[14].imag   = *(float *)&hex_data; 
hex_data = 0x3da252dc; tx_ifft_data[15].real   = *(float *)&hex_data; 
hex_data = 0x3d28710c; tx_ifft_data[15].imag   = *(float *)&hex_data; 
hex_data = 0x3e066457; tx_ifft_data[16].real   = *(float *)&hex_data; 
hex_data = 0x3db3e1d3; tx_ifft_data[16].imag   = *(float *)&hex_data; 
hex_data = 0xbaa0c003; tx_ifft_data[17].real   = *(float *)&hex_data; 
hex_data = 0x3c9f1ce6; tx_ifft_data[17].imag   = *(float *)&hex_data; 
hex_data = 0xbd1caf46; tx_ifft_data[18].real   = *(float *)&hex_data; 
hex_data = 0x3b05ed10; tx_ifft_data[18].imag   = *(float *)&hex_data; 
hex_data = 0xbd0f42e4; tx_ifft_data[19].real   = *(float *)&hex_data; 
hex_data = 0xbc5e3ccc; tx_ifft_data[19].imag   = *(float *)&hex_data; 
hex_data = 0xbd95b079; tx_ifft_data[20].real   = *(float *)&hex_data; 
hex_data = 0xbd1429bf; tx_ifft_data[20].imag   = *(float *)&hex_data; 
hex_data = 0xbc941ed7; tx_ifft_data[21].real   = *(float *)&hex_data; 
hex_data = 0x3c7b4735; tx_ifft_data[21].imag   = *(float *)&hex_data; 
hex_data = 0xbd424b9f; tx_ifft_data[22].real   = *(float *)&hex_data; 
hex_data = 0x3d08ce9f; tx_ifft_data[22].imag   = *(float *)&hex_data; 
hex_data = 0x3c10f4d7; tx_ifft_data[23].real   = *(float *)&hex_data; 
hex_data = 0x3de1b971; tx_ifft_data[23].imag   = *(float *)&hex_data; 
hex_data = 0x3d6981fe; tx_ifft_data[24].real   = *(float *)&hex_data; 
hex_data = 0x3db5fc8d; tx_ifft_data[24].imag   = *(float *)&hex_data; 
hex_data = 0xbd7526fd; tx_ifft_data[25].real   = *(float *)&hex_data; 
hex_data = 0x3d9463db; tx_ifft_data[25].imag   = *(float *)&hex_data; 
hex_data = 0xbc06025a; tx_ifft_data[26].real   = *(float *)&hex_data; 
hex_data = 0xbd14d2de; tx_ifft_data[26].imag   = *(float *)&hex_data; 
hex_data = 0xb8b78d2d; tx_ifft_data[27].real   = *(float *)&hex_data; 
hex_data = 0xbaf9bbf4; tx_ifft_data[27].imag   = *(float *)&hex_data; 
hex_data = 0xba94ff43; tx_ifft_data[28].real   = *(float *)&hex_data; 
hex_data = 0x3d05ddbb; tx_ifft_data[28].imag   = *(float *)&hex_data; 
hex_data = 0x3d6ef9fd; tx_ifft_data[29].real   = *(float *)&hex_data; 
hex_data = 0x3da5569b; tx_ifft_data[29].imag   = *(float *)&hex_data; 
hex_data = 0x3c8226f6; tx_ifft_data[30].real   = *(float *)&hex_data; 
hex_data = 0xbc13c451; tx_ifft_data[30].imag   = *(float *)&hex_data; 
hex_data = 0xbbd88223; tx_ifft_data[31].real   = *(float *)&hex_data; 
hex_data = 0x3d20b416; tx_ifft_data[31].imag   = *(float *)&hex_data; 
hex_data = 0x3e2d413c; tx_ifft_data[32].real   = *(float *)&hex_data; 
hex_data = 0xbd800004; tx_ifft_data[32].imag   = *(float *)&hex_data; 
hex_data = 0xbd2bdbd4; tx_ifft_data[33].real   = *(float *)&hex_data; 
hex_data = 0xbcc7b381; tx_ifft_data[33].imag   = *(float *)&hex_data; 
hex_data = 0x3cc0f1b4; tx_ifft_data[34].real   = *(float *)&hex_data; 
hex_data = 0xba40007e; tx_ifft_data[34].imag   = *(float *)&hex_data; 
hex_data = 0x3d425c95; tx_ifft_data[35].real   = *(float *)&hex_data; 
hex_data = 0x3c2e9534; tx_ifft_data[35].imag   = *(float *)&hex_data; 
hex_data = 0xbbfc0970; tx_ifft_data[36].real   = *(float *)&hex_data; 
hex_data = 0xbc4f430f; tx_ifft_data[36].imag   = *(float *)&hex_data; 
hex_data = 0x3d3a8f4e; tx_ifft_data[37].real   = *(float *)&hex_data; 
hex_data = 0xbbef8cc5; tx_ifft_data[37].imag   = *(float *)&hex_data; 
hex_data = 0x3cf5285d; tx_ifft_data[38].real   = *(float *)&hex_data; 
hex_data = 0x3d8e48d3; tx_ifft_data[38].imag   = *(float *)&hex_data; 
hex_data = 0xbd2abc3e; tx_ifft_data[39].real   = *(float *)&hex_data; 
hex_data = 0x3dc78cd6; tx_ifft_data[39].imag   = *(float *)&hex_data; 
hex_data = 0xbd0ff549; tx_ifft_data[40].real   = *(float *)&hex_data; 
hex_data = 0xbc3138d5; tx_ifft_data[40].imag   = *(float *)&hex_data; 
hex_data = 0xbd25c2fa; tx_ifft_data[41].real   = *(float *)&hex_data; 
hex_data = 0x3d366487; tx_ifft_data[41].imag   = *(float *)&hex_data; 
hex_data = 0x3cc15f77; tx_ifft_data[42].real   = *(float *)&hex_data; 
hex_data = 0xbdb3a604; tx_ifft_data[42].imag   = *(float *)&hex_data; 
hex_data = 0xbbaf18ac; tx_ifft_data[43].real   = *(float *)&hex_data; 
hex_data = 0x3cfd1844; tx_ifft_data[43].imag   = *(float *)&hex_data; 
hex_data = 0x3de42ddd; tx_ifft_data[44].real   = *(float *)&hex_data; 
hex_data = 0xbc9e0c9f; tx_ifft_data[44].imag   = *(float *)&hex_data; 
hex_data = 0xbd0a99b4; tx_ifft_data[45].real   = *(float *)&hex_data; 
hex_data = 0xbd44823e; tx_ifft_data[45].imag   = *(float *)&hex_data; 
hex_data = 0xbb80c436; tx_ifft_data[46].real   = *(float *)&hex_data; 
hex_data = 0x3d73ed09; tx_ifft_data[46].imag   = *(float *)&hex_data; 
hex_data = 0xbdc95ed2; tx_ifft_data[47].real   = *(float *)&hex_data; 
hex_data = 0x3cf0a7b8; tx_ifft_data[47].imag   = *(float *)&hex_data; 
hex_data = 0xbd60000b; tx_ifft_data[48].real   = *(float *)&hex_data; 
hex_data = 0xbe4b1406; tx_ifft_data[48].imag   = *(float *)&hex_data; 
hex_data = 0xbc94b1eb; tx_ifft_data[49].real   = *(float *)&hex_data; 
hex_data = 0xbd97501d; tx_ifft_data[49].imag   = *(float *)&hex_data; 
hex_data = 0x3c9f557a; tx_ifft_data[50].real   = *(float *)&hex_data; 
hex_data = 0xbc4b155e; tx_ifft_data[50].imag   = *(float *)&hex_data; 
hex_data = 0xbd4bf8c5; tx_ifft_data[51].real   = *(float *)&hex_data; 
hex_data = 0x3ca0ca87; tx_ifft_data[51].imag   = *(float *)&hex_data; 
hex_data = 0xbd8d1cce; tx_ifft_data[52].real   = *(float *)&hex_data; 
hex_data = 0xbc27a3f9; tx_ifft_data[52].imag   = *(float *)&hex_data; 
hex_data = 0x3cf50951; tx_ifft_data[53].real   = *(float *)&hex_data; 
hex_data = 0xbd5a7a23; tx_ifft_data[53].imag   = *(float *)&hex_data; 
hex_data = 0x3d48a09c; tx_ifft_data[54].real   = *(float *)&hex_data; 
hex_data = 0x3d68c8c3; tx_ifft_data[54].imag   = *(float *)&hex_data; 
hex_data = 0xbd533b5d; tx_ifft_data[55].real   = *(float *)&hex_data; 
hex_data = 0x3d5e8a79; tx_ifft_data[55].imag   = *(float *)&hex_data; 
hex_data = 0x3e583572; tx_ifft_data[56].real   = *(float *)&hex_data; 
hex_data = 0xbe4d76f8; tx_ifft_data[56].imag   = *(float *)&hex_data; 
hex_data = 0xbd42a876; tx_ifft_data[57].real   = *(float *)&hex_data; 
hex_data = 0xbb3962c2; tx_ifft_data[57].imag   = *(float *)&hex_data; 
hex_data = 0x3ca3c3ca; tx_ifft_data[58].real   = *(float *)&hex_data; 
hex_data = 0xbde8dd60; tx_ifft_data[58].imag   = *(float *)&hex_data; 
hex_data = 0xbd476706; tx_ifft_data[59].real   = *(float *)&hex_data; 
hex_data = 0xbb05051a; tx_ifft_data[59].imag   = *(float *)&hex_data; 
hex_data = 0xbd137176; tx_ifft_data[60].real   = *(float *)&hex_data; 
hex_data = 0xbd3ce059; tx_ifft_data[60].imag   = *(float *)&hex_data; 
hex_data = 0xbd18c302; tx_ifft_data[61].real   = *(float *)&hex_data; 
hex_data = 0x3d95e31e; tx_ifft_data[61].imag   = *(float *)&hex_data; 
hex_data = 0x3ce32a96; tx_ifft_data[62].real   = *(float *)&hex_data; 
hex_data = 0xbcd56542; tx_ifft_data[62].imag   = *(float *)&hex_data; 
hex_data = 0x3d058c24; tx_ifft_data[63].real   = *(float *)&hex_data; 
hex_data = 0xbc87bea1; tx_ifft_data[63].imag   = *(float *)&hex_data; 
hex_data = 0xbd96a09e; tx_ifft_data[64].real   = *(float *)&hex_data; 
hex_data = 0xbe0b504f; tx_ifft_data[64].imag   = *(float *)&hex_data; 
hex_data = 0x3c98b403; tx_ifft_data[65].real   = *(float *)&hex_data; 
hex_data = 0x3d8ac0c5; tx_ifft_data[65].imag   = *(float *)&hex_data; 
hex_data = 0x3bad0765; tx_ifft_data[66].real   = *(float *)&hex_data; 
hex_data = 0xbdf30016; tx_ifft_data[66].imag   = *(float *)&hex_data; 
hex_data = 0xbd4ea486; tx_ifft_data[67].real   = *(float *)&hex_data; 
hex_data = 0xbdaa1925; tx_ifft_data[67].imag   = *(float *)&hex_data; 
hex_data = 0x3d8d0ea2; tx_ifft_data[68].real   = *(float *)&hex_data; 
hex_data = 0x3d266e66; tx_ifft_data[68].imag   = *(float *)&hex_data; 
hex_data = 0x3dd4f1e7; tx_ifft_data[69].real   = *(float *)&hex_data; 
hex_data = 0x3d4bb9b9; tx_ifft_data[69].imag   = *(float *)&hex_data; 
hex_data = 0xbd879851; tx_ifft_data[70].real   = *(float *)&hex_data; 
hex_data = 0x3c2bd611; tx_ifft_data[70].imag   = *(float *)&hex_data; 
hex_data = 0xbd3127b4; tx_ifft_data[71].real   = *(float *)&hex_data; 
hex_data = 0xbdc30b11; tx_ifft_data[71].imag   = *(float *)&hex_data; 
hex_data = 0xbe366437; tx_ifft_data[72].real   = *(float *)&hex_data; 
hex_data = 0x3eb68a44; tx_ifft_data[72].imag   = *(float *)&hex_data; 
hex_data = 0xbb1f5292; tx_ifft_data[73].real   = *(float *)&hex_data; 
hex_data = 0xbd4146a5; tx_ifft_data[73].imag   = *(float *)&hex_data; 
hex_data = 0x3dbac934; tx_ifft_data[74].real   = *(float *)&hex_data; 
hex_data = 0x3dbe3a10; tx_ifft_data[74].imag   = *(float *)&hex_data; 
hex_data = 0xbd10a332; tx_ifft_data[75].real   = *(float *)&hex_data; 
hex_data = 0x3b7ffb30; tx_ifft_data[75].imag   = *(float *)&hex_data; 
hex_data = 0xbd87d59f; tx_ifft_data[76].real   = *(float *)&hex_data; 
hex_data = 0xbda5c1c5; tx_ifft_data[76].imag   = *(float *)&hex_data; 
hex_data = 0xbc3c673e; tx_ifft_data[77].real   = *(float *)&hex_data; 
hex_data = 0xbcbbd0a4; tx_ifft_data[77].imag   = *(float *)&hex_data; 
hex_data = 0xbd1e48ef; tx_ifft_data[78].real   = *(float *)&hex_data; 
hex_data = 0x3d611cc0; tx_ifft_data[78].imag   = *(float *)&hex_data; 
hex_data = 0x3d85702f; tx_ifft_data[79].real   = *(float *)&hex_data; 
hex_data = 0xbd1809b0; tx_ifft_data[79].imag   = *(float *)&hex_data; 
hex_data = 0xbdb24635; tx_ifft_data[80].real   = *(float *)&hex_data; 
hex_data = 0xbde12311; tx_ifft_data[80].imag   = *(float *)&hex_data; 
hex_data = 0x3dcc8072; tx_ifft_data[81].real   = *(float *)&hex_data; 
hex_data = 0x3dce1ae4; tx_ifft_data[81].imag   = *(float *)&hex_data; 
hex_data = 0xbbaf0912; tx_ifft_data[82].real   = *(float *)&hex_data; 
hex_data = 0x3d776b55; tx_ifft_data[82].imag   = *(float *)&hex_data; 
hex_data = 0xbd182fbb; tx_ifft_data[83].real   = *(float *)&hex_data; 
hex_data = 0xbbbf4a4e; tx_ifft_data[83].imag   = *(float *)&hex_data; 
hex_data = 0x3d9a0252; tx_ifft_data[84].real   = *(float *)&hex_data; 
hex_data = 0x3d63bf4d; tx_ifft_data[84].imag   = *(float *)&hex_data; 
hex_data = 0x3d3d02d5; tx_ifft_data[85].real   = *(float *)&hex_data; 
hex_data = 0x3c8956ba; tx_ifft_data[85].imag   = *(float *)&hex_data; 
hex_data = 0xbd74542f; tx_ifft_data[86].real   = *(float *)&hex_data; 
hex_data = 0x3cf90f4a; tx_ifft_data[86].imag   = *(float *)&hex_data; 
hex_data = 0xbd834934; tx_ifft_data[87].real   = *(float *)&hex_data; 
hex_data = 0xbbb8194b; tx_ifft_data[87].imag   = *(float *)&hex_data; 
hex_data = 0x3d05144b; tx_ifft_data[88].real   = *(float *)&hex_data; 
hex_data = 0x3cf308d2; tx_ifft_data[88].imag   = *(float *)&hex_data; 
hex_data = 0x3d282294; tx_ifft_data[89].real   = *(float *)&hex_data; 
hex_data = 0xbca32e4e; tx_ifft_data[89].imag   = *(float *)&hex_data; 
hex_data = 0x3db7e560; tx_ifft_data[90].real   = *(float *)&hex_data; 
hex_data = 0xbd0437e4; tx_ifft_data[90].imag   = *(float *)&hex_data; 
hex_data = 0x3ca596a0; tx_ifft_data[91].real   = *(float *)&hex_data; 
hex_data = 0xbd325ff7; tx_ifft_data[91].imag   = *(float *)&hex_data; 
hex_data = 0xbd6340e4; tx_ifft_data[92].real   = *(float *)&hex_data; 
hex_data = 0xbd8f612e; tx_ifft_data[92].imag   = *(float *)&hex_data; 
hex_data = 0x3b5b0e5e; tx_ifft_data[93].real   = *(float *)&hex_data; 
hex_data = 0xbcb3ef6e; tx_ifft_data[93].imag   = *(float *)&hex_data; 
hex_data = 0xbc883618; tx_ifft_data[94].real   = *(float *)&hex_data; 
hex_data = 0xbc67204d; tx_ifft_data[94].imag   = *(float *)&hex_data; 
hex_data = 0xbd25c1b4; tx_ifft_data[95].real   = *(float *)&hex_data; 
hex_data = 0xbbcbe6ea; tx_ifft_data[95].imag   = *(float *)&hex_data; 
hex_data = 0xbe43e1da; tx_ifft_data[96].real   = *(float *)&hex_data; 
hex_data = 0x3dda827e; tx_ifft_data[96].imag   = *(float *)&hex_data; 
hex_data = 0x3d474b01; tx_ifft_data[97].real   = *(float *)&hex_data; 
hex_data = 0x3c509f3d; tx_ifft_data[97].imag   = *(float *)&hex_data; 
hex_data = 0x3c686c8b; tx_ifft_data[98].real   = *(float *)&hex_data; 
hex_data = 0xbcf81fb4; tx_ifft_data[98].imag   = *(float *)&hex_data; 
hex_data = 0x3c7c1059; tx_ifft_data[99].real   = *(float *)&hex_data; 
hex_data = 0xbc055074; tx_ifft_data[99].imag   = *(float *)&hex_data; 
hex_data = 0x3dd8f192; tx_ifft_data[100].real  = *(float *)&hex_data; 
hex_data = 0xbdcd0c6c; tx_ifft_data[100].imag  = *(float *)&hex_data; 
hex_data = 0xbc93681a; tx_ifft_data[101].real  = *(float *)&hex_data; 
hex_data = 0x3dc989d8; tx_ifft_data[101].imag  = *(float *)&hex_data; 
hex_data = 0x3d2a8031; tx_ifft_data[102].real  = *(float *)&hex_data; 
hex_data = 0x3d45ac58; tx_ifft_data[102].imag  = *(float *)&hex_data; 
hex_data = 0xbcbf3866; tx_ifft_data[103].real  = *(float *)&hex_data; 
hex_data = 0x3d372c9a; tx_ifft_data[103].imag  = *(float *)&hex_data; 
hex_data = 0x3d7572cd; tx_ifft_data[104].real  = *(float *)&hex_data; 
hex_data = 0x3dcd723c; tx_ifft_data[104].imag  = *(float *)&hex_data; 
hex_data = 0x3c83e772; tx_ifft_data[105].real  = *(float *)&hex_data; 
hex_data = 0xbc69ea9c; tx_ifft_data[105].imag  = *(float *)&hex_data; 
hex_data = 0x3cb6ebbc; tx_ifft_data[106].real  = *(float *)&hex_data; 
hex_data = 0xbcf38413; tx_ifft_data[106].imag  = *(float *)&hex_data; 
hex_data = 0x3c011cf2; tx_ifft_data[107].real  = *(float *)&hex_data; 
hex_data = 0x3cf7f08c; tx_ifft_data[107].imag  = *(float *)&hex_data; 
hex_data = 0xbc9aff14; tx_ifft_data[108].real  = *(float *)&hex_data; 
hex_data = 0x3d53a0ca; tx_ifft_data[108].imag  = *(float *)&hex_data; 
hex_data = 0x3d0a7cc3; tx_ifft_data[109].real  = *(float *)&hex_data; 
hex_data = 0x3bca6ab1; tx_ifft_data[109].imag  = *(float *)&hex_data; 
hex_data = 0x3bfa9ca1; tx_ifft_data[110].real  = *(float *)&hex_data; 
hex_data = 0x3c7d0e7d; tx_ifft_data[110].imag  = *(float *)&hex_data; 
hex_data = 0xbca485c9; tx_ifft_data[111].real  = *(float *)&hex_data; 
hex_data = 0x3c7a3a13; tx_ifft_data[111].imag  = *(float *)&hex_data; 
hex_data = 0xbd0a09db; tx_ifft_data[112].real  = *(float *)&hex_data; 
hex_data = 0x3e347368; tx_ifft_data[112].imag  = *(float *)&hex_data; 
hex_data = 0xbcc0f0b3; tx_ifft_data[113].real  = *(float *)&hex_data; 
hex_data = 0x3d94c34d; tx_ifft_data[113].imag  = *(float *)&hex_data; 
hex_data = 0x3c84c792; tx_ifft_data[114].real  = *(float *)&hex_data; 
hex_data = 0xbc1292d4; tx_ifft_data[114].imag  = *(float *)&hex_data; 
hex_data = 0xbd6b9eae; tx_ifft_data[115].real  = *(float *)&hex_data; 
hex_data = 0x3c8ef0d4; tx_ifft_data[115].imag  = *(float *)&hex_data; 
hex_data = 0x3cf73fa7; tx_ifft_data[116].real  = *(float *)&hex_data; 
hex_data = 0xbd0fb66c; tx_ifft_data[116].imag  = *(float *)&hex_data; 
hex_data = 0x3dddf940; tx_ifft_data[117].real  = *(float *)&hex_data; 
hex_data = 0xbcdc651d; tx_ifft_data[117].imag  = *(float *)&hex_data; 
hex_data = 0x3d6dff36; tx_ifft_data[118].real  = *(float *)&hex_data; 
hex_data = 0xbc900875; tx_ifft_data[118].imag  = *(float *)&hex_data; 
hex_data = 0x3d839d58; tx_ifft_data[119].real  = *(float *)&hex_data; 
hex_data = 0xbc988a43; tx_ifft_data[119].imag  = *(float *)&hex_data; 
hex_data = 0x3dc6e038; tx_ifft_data[120].real  = *(float *)&hex_data; 
hex_data = 0x3d1b595f; tx_ifft_data[120].imag  = *(float *)&hex_data; 
hex_data = 0x3d6cd593; tx_ifft_data[121].real  = *(float *)&hex_data; 
hex_data = 0x3c2e5292; tx_ifft_data[121].imag  = *(float *)&hex_data; 
hex_data = 0xbd049d23; tx_ifft_data[122].real  = *(float *)&hex_data; 
hex_data = 0xbd62d6fb; tx_ifft_data[122].imag  = *(float *)&hex_data; 
hex_data = 0xbbe6a78e; tx_ifft_data[123].real  = *(float *)&hex_data; 
hex_data = 0x3c95ad90; tx_ifft_data[123].imag  = *(float *)&hex_data; 
hex_data = 0x3cf6b4b1; tx_ifft_data[124].real  = *(float *)&hex_data; 
hex_data = 0x3daae27f; tx_ifft_data[124].imag  = *(float *)&hex_data; 
hex_data = 0xb8bdb980; tx_ifft_data[125].real  = *(float *)&hex_data; 
hex_data = 0xbd28a429; tx_ifft_data[125].imag  = *(float *)&hex_data; 
hex_data = 0xbcdd1b7b; tx_ifft_data[126].real  = *(float *)&hex_data; 
hex_data = 0xbd25910a; tx_ifft_data[126].imag  = *(float *)&hex_data; 
hex_data = 0x3d9fde8c; tx_ifft_data[127].real  = *(float *)&hex_data; 
hex_data = 0xbd4e90bc; tx_ifft_data[127].imag  = *(float *)&hex_data; 

}

void get_tx_fft_data() {

hex_data = 0x3f350508; tx_fft_data[0].real    = *(float *)&hex_data; 
hex_data = 0x3f350492; tx_fft_data[0].imag    = *(float *)&hex_data; 
hex_data = 0x3f350589; tx_fft_data[1].real    = *(float *)&hex_data; 
hex_data = 0x3f350521; tx_fft_data[1].imag    = *(float *)&hex_data; 
hex_data = 0x3f3504bf; tx_fft_data[2].real    = *(float *)&hex_data; 
hex_data = 0xbf3504cf; tx_fft_data[2].imag    = *(float *)&hex_data; 
hex_data = 0xbf350530; tx_fft_data[3].real    = *(float *)&hex_data; 
hex_data = 0xbf35052b; tx_fft_data[3].imag    = *(float *)&hex_data; 
hex_data = 0x3f350518; tx_fft_data[4].real    = *(float *)&hex_data; 
hex_data = 0xbf35050d; tx_fft_data[4].imag    = *(float *)&hex_data; 
hex_data = 0xbf3504e8; tx_fft_data[5].real    = *(float *)&hex_data; 
hex_data = 0xbf3504b9; tx_fft_data[5].imag    = *(float *)&hex_data; 
hex_data = 0xbf350549; tx_fft_data[6].real    = *(float *)&hex_data; 
hex_data = 0x3f3504fa; tx_fft_data[6].imag    = *(float *)&hex_data; 
hex_data = 0xbf80000d; tx_fft_data[7].real    = *(float *)&hex_data; 
hex_data = 0x3f80000c; tx_fft_data[7].imag    = *(float *)&hex_data; 
hex_data = 0xbf3504cf; tx_fft_data[8].real    = *(float *)&hex_data; 
hex_data = 0xbf35053f; tx_fft_data[8].imag    = *(float *)&hex_data; 
hex_data = 0xbf3504c3; tx_fft_data[9].real    = *(float *)&hex_data; 
hex_data = 0x3f3504c3; tx_fft_data[9].imag    = *(float *)&hex_data; 
hex_data = 0xbf35053c; tx_fft_data[10].real   = *(float *)&hex_data; 
hex_data = 0xbf3504e8; tx_fft_data[10].imag   = *(float *)&hex_data; 
hex_data = 0x3f350526; tx_fft_data[11].real   = *(float *)&hex_data; 
hex_data = 0x3f35050a; tx_fft_data[11].imag   = *(float *)&hex_data; 
hex_data = 0xbf35052c; tx_fft_data[12].real   = *(float *)&hex_data; 
hex_data = 0xbf350537; tx_fft_data[12].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504d6; tx_fft_data[13].real   = *(float *)&hex_data; 
hex_data = 0xbf3504c3; tx_fft_data[13].imag   = *(float *)&hex_data; 
hex_data = 0x3f35049f; tx_fft_data[14].real   = *(float *)&hex_data; 
hex_data = 0x3f3504b0; tx_fft_data[14].imag   = *(float *)&hex_data; 
hex_data = 0x3f800039; tx_fft_data[15].real   = *(float *)&hex_data; 
hex_data = 0x3f800012; tx_fft_data[15].imag   = *(float *)&hex_data; 
hex_data = 0xbf350513; tx_fft_data[16].real   = *(float *)&hex_data; 
hex_data = 0x3f3504e0; tx_fft_data[16].imag   = *(float *)&hex_data; 
hex_data = 0x3f350526; tx_fft_data[17].real   = *(float *)&hex_data; 
hex_data = 0x3f3504dc; tx_fft_data[17].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504c2; tx_fft_data[18].real   = *(float *)&hex_data; 
hex_data = 0xbf3504f7; tx_fft_data[18].imag   = *(float *)&hex_data; 
hex_data = 0xbf350506; tx_fft_data[19].real   = *(float *)&hex_data; 
hex_data = 0xbf350529; tx_fft_data[19].imag   = *(float *)&hex_data; 
hex_data = 0xbf350537; tx_fft_data[20].real   = *(float *)&hex_data; 
hex_data = 0x3f3504e6; tx_fft_data[20].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504c3; tx_fft_data[21].real   = *(float *)&hex_data; 
hex_data = 0xbf3504dc; tx_fft_data[21].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504b6; tx_fft_data[22].real   = *(float *)&hex_data; 
hex_data = 0x3f350500; tx_fft_data[22].imag   = *(float *)&hex_data; 
hex_data = 0x3f800022; tx_fft_data[23].real   = *(float *)&hex_data; 
hex_data = 0x3f7ffffc; tx_fft_data[23].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504d2; tx_fft_data[24].real   = *(float *)&hex_data; 
hex_data = 0x3f3504b7; tx_fft_data[24].imag   = *(float *)&hex_data; 
hex_data = 0xbf35051b; tx_fft_data[25].real   = *(float *)&hex_data; 
hex_data = 0x3f3504d8; tx_fft_data[25].imag   = *(float *)&hex_data; 
hex_data = 0x3f350520; tx_fft_data[26].real   = *(float *)&hex_data; 
hex_data = 0x3f3504b5; tx_fft_data[26].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504cf; tx_fft_data[27].real   = *(float *)&hex_data; 
hex_data = 0x3f350535; tx_fft_data[27].imag   = *(float *)&hex_data; 
hex_data = 0x3f35049b; tx_fft_data[28].real   = *(float *)&hex_data; 
hex_data = 0x3f350536; tx_fft_data[28].imag   = *(float *)&hex_data; 
hex_data = 0x3f35051d; tx_fft_data[29].real   = *(float *)&hex_data; 
hex_data = 0xbf3504cf; tx_fft_data[29].imag   = *(float *)&hex_data; 
hex_data = 0x3f350516; tx_fft_data[30].real   = *(float *)&hex_data; 
hex_data = 0x3f3504ed; tx_fft_data[30].imag   = *(float *)&hex_data; 
hex_data = 0x3f80000a; tx_fft_data[31].real   = *(float *)&hex_data; 
hex_data = 0x3f7fffd9; tx_fft_data[31].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504d3; tx_fft_data[32].real   = *(float *)&hex_data; 
hex_data = 0xbf35052b; tx_fft_data[32].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504f4; tx_fft_data[33].real   = *(float *)&hex_data; 
hex_data = 0x3f350534; tx_fft_data[33].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504d4; tx_fft_data[34].real   = *(float *)&hex_data; 
hex_data = 0xbf3504fe; tx_fft_data[34].imag   = *(float *)&hex_data; 
hex_data = 0xbf350510; tx_fft_data[35].real   = *(float *)&hex_data; 
hex_data = 0xbf3504f9; tx_fft_data[35].imag   = *(float *)&hex_data; 
hex_data = 0x3f35050e; tx_fft_data[36].real   = *(float *)&hex_data; 
hex_data = 0x3f3504e0; tx_fft_data[36].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504c0; tx_fft_data[37].real   = *(float *)&hex_data; 
hex_data = 0xbf3504b5; tx_fft_data[37].imag   = *(float *)&hex_data; 
hex_data = 0x3f35050a; tx_fft_data[38].real   = *(float *)&hex_data; 
hex_data = 0x3f350506; tx_fft_data[38].imag   = *(float *)&hex_data; 
hex_data = 0xbf800014; tx_fft_data[39].real   = *(float *)&hex_data; 
hex_data = 0x3f800008; tx_fft_data[39].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504bf; tx_fft_data[40].real   = *(float *)&hex_data; 
hex_data = 0xbf35051a; tx_fft_data[40].imag   = *(float *)&hex_data; 
hex_data = 0x3f35050b; tx_fft_data[41].real   = *(float *)&hex_data; 
hex_data = 0x3f35053f; tx_fft_data[41].imag   = *(float *)&hex_data; 
hex_data = 0xbf350520; tx_fft_data[42].real   = *(float *)&hex_data; 
hex_data = 0xbf350561; tx_fft_data[42].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504f6; tx_fft_data[43].real   = *(float *)&hex_data; 
hex_data = 0x3f35050b; tx_fft_data[43].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504b9; tx_fft_data[44].real   = *(float *)&hex_data; 
hex_data = 0xbf35052f; tx_fft_data[44].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504d7; tx_fft_data[45].real   = *(float *)&hex_data; 
hex_data = 0xbf3504da; tx_fft_data[45].imag   = *(float *)&hex_data; 
hex_data = 0xbf350538; tx_fft_data[46].real   = *(float *)&hex_data; 
hex_data = 0xbf3504d8; tx_fft_data[46].imag   = *(float *)&hex_data; 
hex_data = 0x3f7fffe5; tx_fft_data[47].real   = *(float *)&hex_data; 
hex_data = 0x3f80000d; tx_fft_data[47].imag   = *(float *)&hex_data; 
hex_data = 0x3f35048a; tx_fft_data[48].real   = *(float *)&hex_data; 
hex_data = 0x3f350538; tx_fft_data[48].imag   = *(float *)&hex_data; 
hex_data = 0xbf350464; tx_fft_data[49].real   = *(float *)&hex_data; 
hex_data = 0xbf3504ea; tx_fft_data[49].imag   = *(float *)&hex_data; 
hex_data = 0x3f35054a; tx_fft_data[50].real   = *(float *)&hex_data; 
hex_data = 0x3f350480; tx_fft_data[50].imag   = *(float *)&hex_data; 
hex_data = 0xbf350503; tx_fft_data[51].real   = *(float *)&hex_data; 
hex_data = 0xbf3504c7; tx_fft_data[51].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504ec; tx_fft_data[52].real   = *(float *)&hex_data; 
hex_data = 0x3f350508; tx_fft_data[52].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504c2; tx_fft_data[53].real   = *(float *)&hex_data; 
hex_data = 0xbf3504af; tx_fft_data[53].imag   = *(float *)&hex_data; 
hex_data = 0x3f350442; tx_fft_data[54].real   = *(float *)&hex_data; 
hex_data = 0xbf3504fd; tx_fft_data[54].imag   = *(float *)&hex_data; 
hex_data = 0x3f7fff8f; tx_fft_data[55].real   = *(float *)&hex_data; 
hex_data = 0x3f7fffec; tx_fft_data[55].imag   = *(float *)&hex_data; 
hex_data = 0xbf350544; tx_fft_data[56].real   = *(float *)&hex_data; 
hex_data = 0xbf350539; tx_fft_data[56].imag   = *(float *)&hex_data; 
hex_data = 0x3f350494; tx_fft_data[57].real   = *(float *)&hex_data; 
hex_data = 0x3f35051f; tx_fft_data[57].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504ad; tx_fft_data[58].real   = *(float *)&hex_data; 
hex_data = 0xbf3504b4; tx_fft_data[58].imag   = *(float *)&hex_data; 
hex_data = 0x3f350526; tx_fft_data[59].real   = *(float *)&hex_data; 
hex_data = 0x3f350507; tx_fft_data[59].imag   = *(float *)&hex_data; 
hex_data = 0xbf350513; tx_fft_data[60].real   = *(float *)&hex_data; 
hex_data = 0xbf3504a1; tx_fft_data[60].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504e1; tx_fft_data[61].real   = *(float *)&hex_data; 
hex_data = 0x3f3504d0; tx_fft_data[61].imag   = *(float *)&hex_data; 
hex_data = 0xbf350528; tx_fft_data[62].real   = *(float *)&hex_data; 
hex_data = 0xbf350470; tx_fft_data[62].imag   = *(float *)&hex_data; 
hex_data = 0x3f80001c; tx_fft_data[63].real   = *(float *)&hex_data; 
hex_data = 0x3f7fffe4; tx_fft_data[63].imag   = *(float *)&hex_data; 
hex_data = 0x3f350481; tx_fft_data[64].real   = *(float *)&hex_data; 
hex_data = 0xbf350492; tx_fft_data[64].imag   = *(float *)&hex_data; 
hex_data = 0xbf350508; tx_fft_data[65].real   = *(float *)&hex_data; 
hex_data = 0xbf3504e0; tx_fft_data[65].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504ec; tx_fft_data[66].real   = *(float *)&hex_data; 
hex_data = 0x3f3504d0; tx_fft_data[66].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504fb; tx_fft_data[67].real   = *(float *)&hex_data; 
hex_data = 0xbf3504ab; tx_fft_data[67].imag   = *(float *)&hex_data; 
hex_data = 0x3f350525; tx_fft_data[68].real   = *(float *)&hex_data; 
hex_data = 0xbf3504e6; tx_fft_data[68].imag   = *(float *)&hex_data; 
hex_data = 0xbf350482; tx_fft_data[69].real   = *(float *)&hex_data; 
hex_data = 0xbf3504cf; tx_fft_data[69].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504cf; tx_fft_data[70].real   = *(float *)&hex_data; 
hex_data = 0xbf35055c; tx_fft_data[70].imag   = *(float *)&hex_data; 
hex_data = 0xbf7ffff0; tx_fft_data[71].real   = *(float *)&hex_data; 
hex_data = 0x3f7fffe6; tx_fft_data[71].imag   = *(float *)&hex_data; 
hex_data = 0xbf350520; tx_fft_data[72].real   = *(float *)&hex_data; 
hex_data = 0xbf3504f2; tx_fft_data[72].imag   = *(float *)&hex_data; 
hex_data = 0x3f350545; tx_fft_data[73].real   = *(float *)&hex_data; 
hex_data = 0xbf3504a9; tx_fft_data[73].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504f4; tx_fft_data[74].real   = *(float *)&hex_data; 
hex_data = 0xbf35052b; tx_fft_data[74].imag   = *(float *)&hex_data; 
hex_data = 0x3f350528; tx_fft_data[75].real   = *(float *)&hex_data; 
hex_data = 0x3f350523; tx_fft_data[75].imag   = *(float *)&hex_data; 
hex_data = 0xbf35049e; tx_fft_data[76].real   = *(float *)&hex_data; 
hex_data = 0xbf35051f; tx_fft_data[76].imag   = *(float *)&hex_data; 
hex_data = 0xbf350497; tx_fft_data[77].real   = *(float *)&hex_data; 
hex_data = 0xbf35054c; tx_fft_data[77].imag   = *(float *)&hex_data; 
hex_data = 0xbf350536; tx_fft_data[78].real   = *(float *)&hex_data; 
hex_data = 0xbf3504e9; tx_fft_data[78].imag   = *(float *)&hex_data; 
hex_data = 0x3f80001f; tx_fft_data[79].real   = *(float *)&hex_data; 
hex_data = 0x3f800022; tx_fft_data[79].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504ff; tx_fft_data[80].real   = *(float *)&hex_data; 
hex_data = 0xbf35051f; tx_fft_data[80].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504fd; tx_fft_data[81].real   = *(float *)&hex_data; 
hex_data = 0xbf350547; tx_fft_data[81].imag   = *(float *)&hex_data; 
hex_data = 0xbf350496; tx_fft_data[82].real   = *(float *)&hex_data; 
hex_data = 0xbf350527; tx_fft_data[82].imag   = *(float *)&hex_data; 
hex_data = 0xbf350511; tx_fft_data[83].real   = *(float *)&hex_data; 
hex_data = 0xbf35052e; tx_fft_data[83].imag   = *(float *)&hex_data; 
hex_data = 0x3f35053d; tx_fft_data[84].real   = *(float *)&hex_data; 
hex_data = 0x3f350525; tx_fft_data[84].imag   = *(float *)&hex_data; 
hex_data = 0xbf35050c; tx_fft_data[85].real   = *(float *)&hex_data; 
hex_data = 0xbf3504f1; tx_fft_data[85].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504e0; tx_fft_data[86].real   = *(float *)&hex_data; 
hex_data = 0xbf3504ee; tx_fft_data[86].imag   = *(float *)&hex_data; 
hex_data = 0x3f800033; tx_fft_data[87].real   = *(float *)&hex_data; 
hex_data = 0x3f7fffdb; tx_fft_data[87].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504a3; tx_fft_data[88].real   = *(float *)&hex_data; 
hex_data = 0xbf3504f3; tx_fft_data[88].imag   = *(float *)&hex_data; 
hex_data = 0x3f3504c4; tx_fft_data[89].real   = *(float *)&hex_data; 
hex_data = 0xbf3504bf; tx_fft_data[89].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504ba; tx_fft_data[90].real   = *(float *)&hex_data; 
hex_data = 0xbf3504c4; tx_fft_data[90].imag   = *(float *)&hex_data; 
hex_data = 0x3f35052c; tx_fft_data[91].real   = *(float *)&hex_data; 
hex_data = 0xbf35048a; tx_fft_data[91].imag   = *(float *)&hex_data; 
hex_data = 0xbf350529; tx_fft_data[92].real   = *(float *)&hex_data; 
hex_data = 0xbf35045c; tx_fft_data[92].imag   = *(float *)&hex_data; 
hex_data = 0x3f350503; tx_fft_data[93].real   = *(float *)&hex_data; 
hex_data = 0xbf350530; tx_fft_data[93].imag   = *(float *)&hex_data; 
hex_data = 0xbf350511; tx_fft_data[94].real   = *(float *)&hex_data; 
hex_data = 0xbf3504da; tx_fft_data[94].imag   = *(float *)&hex_data; 
hex_data = 0x3f800012; tx_fft_data[95].real   = *(float *)&hex_data; 
hex_data = 0x3f7fff97; tx_fft_data[95].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504d3; tx_fft_data[96].real   = *(float *)&hex_data; 
hex_data = 0x3f35052b; tx_fft_data[96].imag   = *(float *)&hex_data; 
hex_data = 0xbf350549; tx_fft_data[97].real   = *(float *)&hex_data; 
hex_data = 0xbf3504c7; tx_fft_data[97].imag   = *(float *)&hex_data; 
hex_data = 0xbf3504f9; tx_fft_data[98].real   = *(float *)&hex_data; 
hex_data = 0x3f35051d; tx_fft_data[98].imag   = *(float *)&hex_data; 
hex_data = 0xbf350504; tx_fft_data[99].real   = *(float *)&hex_data; 
hex_data = 0xbf350518; tx_fft_data[99].imag   = *(float *)&hex_data; 
hex_data = 0xbf35050d; tx_fft_data[100].real  = *(float *)&hex_data; 
hex_data = 0x3f3504c7; tx_fft_data[100].imag  = *(float *)&hex_data; 
hex_data = 0xbf350589; tx_fft_data[101].real  = *(float *)&hex_data; 
hex_data = 0xbf3504e9; tx_fft_data[101].imag  = *(float *)&hex_data; 
hex_data = 0x3f350535; tx_fft_data[102].real  = *(float *)&hex_data; 
hex_data = 0x3f3504dc; tx_fft_data[102].imag  = *(float *)&hex_data; 
hex_data = 0xbf80000f; tx_fft_data[103].real  = *(float *)&hex_data; 
hex_data = 0x3f7fffba; tx_fft_data[103].imag  = *(float *)&hex_data; 
hex_data = 0xbf35052f; tx_fft_data[104].real  = *(float *)&hex_data; 
hex_data = 0xbf350516; tx_fft_data[104].imag  = *(float *)&hex_data; 
hex_data = 0xbf350502; tx_fft_data[105].real  = *(float *)&hex_data; 
hex_data = 0x3f3504e5; tx_fft_data[105].imag  = *(float *)&hex_data; 
hex_data = 0xbf350492; tx_fft_data[106].real  = *(float *)&hex_data; 
hex_data = 0xbf350484; tx_fft_data[106].imag  = *(float *)&hex_data; 
hex_data = 0x3f3504a7; tx_fft_data[107].real  = *(float *)&hex_data; 
hex_data = 0x3f350520; tx_fft_data[107].imag  = *(float *)&hex_data; 
hex_data = 0xbf3504c0; tx_fft_data[108].real  = *(float *)&hex_data; 
hex_data = 0xbf3504e6; tx_fft_data[108].imag  = *(float *)&hex_data; 
hex_data = 0xbf3504d8; tx_fft_data[109].real  = *(float *)&hex_data; 
hex_data = 0x3f35053c; tx_fft_data[109].imag  = *(float *)&hex_data; 
hex_data = 0xbf350521; tx_fft_data[110].real  = *(float *)&hex_data; 
hex_data = 0xbf3504c9; tx_fft_data[110].imag  = *(float *)&hex_data; 
hex_data = 0x3f800014; tx_fft_data[111].real  = *(float *)&hex_data; 
hex_data = 0x3f7ffffd; tx_fft_data[111].imag  = *(float *)&hex_data; 
hex_data = 0x3f350501; tx_fft_data[112].real  = *(float *)&hex_data; 
hex_data = 0x3f350490; tx_fft_data[112].imag  = *(float *)&hex_data; 
hex_data = 0xbf35053a; tx_fft_data[113].real  = *(float *)&hex_data; 
hex_data = 0xbf350502; tx_fft_data[113].imag  = *(float *)&hex_data; 
hex_data = 0x3f35049f; tx_fft_data[114].real  = *(float *)&hex_data; 
hex_data = 0xbf350527; tx_fft_data[114].imag  = *(float *)&hex_data; 
hex_data = 0xbf3504e4; tx_fft_data[115].real  = *(float *)&hex_data; 
hex_data = 0xbf350504; tx_fft_data[115].imag  = *(float *)&hex_data; 
hex_data = 0x3f3504f4; tx_fft_data[116].real  = *(float *)&hex_data; 
hex_data = 0x3f3504c9; tx_fft_data[116].imag  = *(float *)&hex_data; 
hex_data = 0xbf35055e; tx_fft_data[117].real  = *(float *)&hex_data; 
hex_data = 0xbf3504b3; tx_fft_data[117].imag  = *(float *)&hex_data; 
hex_data = 0x3f35050a; tx_fft_data[118].real  = *(float *)&hex_data; 
hex_data = 0x3f35051a; tx_fft_data[118].imag  = *(float *)&hex_data; 
hex_data = 0x3f800003; tx_fft_data[119].real  = *(float *)&hex_data; 
hex_data = 0x3f800022; tx_fft_data[119].imag  = *(float *)&hex_data; 
hex_data = 0xbf35053a; tx_fft_data[120].real  = *(float *)&hex_data; 
hex_data = 0xbf3504e0; tx_fft_data[120].imag  = *(float *)&hex_data; 
hex_data = 0xbf3504d1; tx_fft_data[121].real  = *(float *)&hex_data; 
hex_data = 0xbf3504a3; tx_fft_data[121].imag  = *(float *)&hex_data; 
hex_data = 0xbf3504fe; tx_fft_data[122].real  = *(float *)&hex_data; 
hex_data = 0xbf3504de; tx_fft_data[122].imag  = *(float *)&hex_data; 
hex_data = 0xbf350520; tx_fft_data[123].real  = *(float *)&hex_data; 
hex_data = 0x3f3504f4; tx_fft_data[123].imag  = *(float *)&hex_data; 
hex_data = 0xbf3504ce; tx_fft_data[124].real  = *(float *)&hex_data; 
hex_data = 0xbf3504e7; tx_fft_data[124].imag  = *(float *)&hex_data; 
hex_data = 0x3f3504fc; tx_fft_data[125].real  = *(float *)&hex_data; 
hex_data = 0xbf35051f; tx_fft_data[125].imag  = *(float *)&hex_data; 
hex_data = 0xbf350503; tx_fft_data[126].real  = *(float *)&hex_data; 
hex_data = 0xbf3504fe; tx_fft_data[126].imag  = *(float *)&hex_data; 
hex_data = 0x3f7fffc8; tx_fft_data[127].real  = *(float *)&hex_data; 
hex_data = 0x3f7fffd3; tx_fft_data[127].imag  = *(float *)&hex_data; 

}

#define SEC2NANOSEC  1000000000
extern int           dma1_control_fd;
extern unsigned int *dma1_control_base_addr;
extern int           fd_udmabuf1;
extern int           fft1_control_fd;
extern unsigned int *fft1_control_base_addr;
extern unsigned int  udmabuf1_phys_addr;
extern TYPE         *udmabuf1_base_addr;
extern int           dma2_control_fd;
extern unsigned int *dma2_control_base_addr;
extern int           fd_udmabuf2;
extern int           fft2_control_fd;
extern unsigned int *fft2_control_base_addr;
extern unsigned int  udmabuf2_phys_addr;
extern TYPE         *udmabuf2_base_addr;

int           dma1_control_fd;
unsigned int *dma1_control_base_addr;
int           fd_udmabuf1;
int           fft1_control_fd;
unsigned int *fft1_control_base_addr;
unsigned int  udmabuf1_phys_addr;
TYPE         *udmabuf1_base_addr;
int           dma2_control_fd;
unsigned int *dma2_control_base_addr;
int           fd_udmabuf2;
int           fft2_control_fd;
unsigned int *fft2_control_base_addr;
unsigned int  udmabuf2_phys_addr;
TYPE         *udmabuf2_base_addr;

#include "dma.h"
#include "fft_hwa.h"

#include "../../../include/DashExtras.h"

double PI = 3.141593;
typedef double complex cplx;
 
void _fft_hs(cplx buf[], cplx out[], int n, int step) {
    int i;

    if (step < n) {
        _fft_hs(out, buf, n, step * 2);
        _fft_hs(out + step, buf + step, n, step * 2);
        for (i = 0; i < n; i += 2 * step) {
	    cplx t = cexp(-I * PI * i / n) * out[i + step];
	    buf[i / 2]     = out[i] + t;
	    buf[(i + n)/2] = out[i] - t;
        }
    }
}

#ifndef THREAD_PER_TASK
void fft_hs(int fft_id, comp_t fdata[], int n, int hw_fft_busy) {
#else
void* fft_hs(void *input) {

    int fft_id = ((struct args_fft *)input)->fft_id;
    comp_t *fdata = ((struct args_fft *)input)->fdata;
    int n = ((struct args_fft *)input)->n;
    int hw_fft_busy = ((struct args_fft *)input)->hw_fft_busy;
#endif
    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]",n));

    #ifdef ACC_RX_FFT
        memcpy(fdata, tx_fft_data, 128*sizeof(comp_t));
        return;
    #endif

    #ifdef DISPLAY_CPU_ASSIGNMENT
        printf("[INFO] RX-FFT assigned to CPU: %d\n", sched_getcpu());
    #endif

    int i;
    cplx out[n], buf[n];
    float fft_hw[DIM * 2];

    float        *udmabuf_base_addr;
    unsigned int *dma_control_base_addr;
    unsigned int *fft_control_base_addr;
    unsigned int udmabuf_phys_addr;

    if (hw_fft_busy == 1) {

        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        printf("[INFO] FFT running on A53\n");
        #endif

        for (i = 0; i < n; i++) buf[i] = (double)fdata[i].real + (double)fdata[i].imag * I;
        for (i = 0; i < n; i++) out[i] = buf[i];

        _fft_hs(buf, out, n, 1);

        for (i = 0; i < n; i++) {
            fdata[i].real = (float)creal(buf[i]);
            fdata[i].imag = (float)cimag(buf[i]);
        }
    } else {

        if (fft_id == 1) {
            udmabuf_base_addr     = udmabuf1_base_addr;
            dma_control_base_addr = dma1_control_base_addr;
            udmabuf_phys_addr     = udmabuf1_phys_addr;
            fft_control_base_addr = fft1_control_base_addr;
        } else {
            udmabuf_base_addr     = udmabuf2_base_addr;
            dma_control_base_addr = dma2_control_base_addr;
            udmabuf_phys_addr     = udmabuf2_phys_addr;
            fft_control_base_addr = fft2_control_base_addr;
        }

        config_fft(fft_control_base_addr, 7);

        memcpy(udmabuf_base_addr, fdata, sizeof(float) * DIM * 2);

        // Setup RX over DMA
        setup_rx(dma_control_base_addr, udmabuf_phys_addr, n);

        // Transfer Matrix A over the DMA
        setup_tx(dma_control_base_addr, udmabuf_phys_addr, n);

        //dma_wait_for_tx_complete();

        // Wait for DMA to complete transfer to destination buffer
        dma_wait_for_rx_complete(dma_control_base_addr);

        memcpy(fft_hw, &udmabuf_base_addr[DIM * 2], sizeof(float) * DIM * 2);

        //for (i = 0; i < n; i++) {
        //    fdata[i].real = (float) fft_hw[(i * 2)];
        //    fdata[i].imag = (float) fft_hw[(i * 2) + 1];
        //}
        
        // Compare SW and HW results
        // check_result(fdata, fft_hw);

    }

    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]",n));
}

#ifndef THREAD_PER_TASK
void ifft_hs(int fft_id, comp_t fdata[], int n, int hw_fft_busy) {
#else
void * ifft_hs(void *input) {

    int fft_id = ((struct args_ifft*)input)->fft_id;
    comp_t *fdata = ((struct args_ifft*)input)->fdata;
    int n = ((struct args_ifft*)input)->n;
    int hw_fft_busy = ((struct args_ifft*)input)->hw_fft_busy;
#endif

    KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]",n));
    #ifdef ACC_TX_IFFT
        memcpy(fdata, tx_ifft_data, 128*sizeof(comp_t));
        return;
    #endif

    #ifdef DISPLAY_CPU_ASSIGNMENT
        printf("[INFO] TX-IFFT assigned to CPU: %d\n", sched_getcpu());
    #endif

    int i, n2;
    cplx out[n], buf[n];
    cplx tmp;
    float fft_hw[DIM * 2];

    float        *udmabuf_base_addr;
    unsigned int *dma_control_base_addr;
    unsigned int *fft_control_base_addr;
    unsigned int udmabuf_phys_addr;

    if (hw_fft_busy == 1) {

        #ifdef PRINT_BLOCK_EXECUTION_TIMES
        printf("[INFO] IFFT running on A53\n");
        #endif

        for (i = 0; i < n; i++) buf[i] = (double)fdata[i].real + (double)fdata[i].imag * I;
        for (i = 0; i < n; i++) out[i] = buf[i];

        _fft_hs(buf, out, n, 1);

        n2 = n/2;
        buf[0] = buf[0]/n;
        buf[n2] = buf[n2]/n;
        for(i=1; i<n2; i++) {
          tmp = buf[i]/n;
          buf[i] = buf[n-i]/n;
          buf[n-i] = tmp;
        }

        for (i = 0; i < n; i++) {
            fdata[i].real = (float)creal(buf[i]);
            fdata[i].imag = (float)cimag(buf[i]);;
        }
        
    } else {

        if (fft_id == 1) {
            udmabuf_base_addr     = udmabuf1_base_addr;
            dma_control_base_addr = dma1_control_base_addr;
            udmabuf_phys_addr     = udmabuf1_phys_addr;
            fft_control_base_addr = fft1_control_base_addr;
        } else {
            udmabuf_base_addr     = udmabuf2_base_addr;
            dma_control_base_addr = dma2_control_base_addr;
            udmabuf_phys_addr     = udmabuf2_phys_addr;
            fft_control_base_addr = fft2_control_base_addr;
        }

        config_ifft(fft_control_base_addr, 7);

        memcpy(udmabuf_base_addr, fdata, sizeof(float) * DIM * 2);

        // Setup RX over DMA
        setup_rx(dma_control_base_addr, udmabuf_phys_addr, n);

        // Transfer Matrix A over the DMA
        setup_tx(dma_control_base_addr, udmabuf_phys_addr, n);

        //dma_wait_for_tx_complete();

        // Wait for DMA to complete transfer to destination buffer
        dma_wait_for_rx_complete(dma_control_base_addr);

        memcpy(fft_hw, &udmabuf_base_addr[DIM * 2], sizeof(float) * DIM * 2);

        //for (i = 0; i < n; i++) {
        //    fdata[i].real = (float) fft_hw[(i * 2)] / 128;
        //    fdata[i].imag = (float) fft_hw[(i * 2) + 1] / 128;
        //}

        // Compare SW and HW results
        //check_result(fdata, fft_hw);

    }
    KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]",n));
   
}
