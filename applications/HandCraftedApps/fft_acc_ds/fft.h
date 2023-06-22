//###################################################################################
// Function to initialize memory maps to FFT
//###################################################################################
void init_fft() {

	// Open device memory in order to get access to DMA control slave
    int fft_control_fd = open("/dev/mem", O_RDWR|O_SYNC);
    if(fft_control_fd < 0) {
      printf("[ERROR] Can't open /dev/mem. Exiting ...\n");
      exit(1);
    }

    printf("[ INFO] Successfully opened /dev/mem ...\n");

	// Obtain virtual address to DMA control slave through mmap
    fft_control_base_addr = (unsigned int*) mmap(0, getpagesize(), PROT_READ|PROT_WRITE, MAP_SHARED, fft_control_fd, FFT_CONTROL_BASE_ADDR);

    if(fft_control_base_addr == MAP_FAILED) {
       printf("[ERROR] Can't obtain memory map to FFT control slave. Exiting ...\n");
       exit(1);
    }

    printf("[ INFO] Successfully obtained virtual address to FFT control slave ...\n");
}

//###################################################################################
// Function to Write Data to FFT Control Register
//###################################################################################
void fft_write_reg(unsigned int *base, unsigned int offset, int data) {

    *(base + offset) = data;
}

//###################################################################################
// Function to initialize memory maps to FFT 
//###################################################################################
void config_ifft(unsigned int *base, unsigned int size) {
    
    fft_write_reg(base, 0x0, size);
    //printf("[ INFO] Configured FFT IP ...\n");
}

//###################################################################################
// Function to initialize memory maps to FFT 
//###################################################################################
void config_fft(unsigned int *base, unsigned int size) {
    
    fft_write_reg(base, 0x0, (0x7 << 8 | size));
    //printf("[ INFO] Configured FFT IP ...\n");
}

//###################################################################################
// Function to generate Matrix A
//###################################################################################
void gen_input_fft (TYPE *fft_input, TYPE *fft_input_ref) {

unsigned int a, b;

    for (int i = 0; i < DIM * 2; i++) {
       fft_input[i] = (TYPE) i;
       fft_input_ref[i] = (TYPE) i;
    }

//    a  =  0xbdc48c72  ;  b  =  0xbd174b40  ;  *(fft_input  +  1)    =  *(float *)&a;  *(fft_input  +  0)    =  *(float *)&b;
//    a  =  0x3d118091  ;  b  =  0xbd7d7f52  ;  *(fft_input  +  3)    =  *(float *)&a;  *(fft_input  +  2)    =  *(float *)&b;
//    a  =  0x3d039bcc  ;  b  =  0x3db2c732  ;  *(fft_input  +  5)    =  *(float *)&a;  *(fft_input  +  4)    =  *(float *)&b;
//    a  =  0x3d1bf594  ;  b  =  0xbc6d6fdb  ;  *(fft_input  +  7)    =  *(float *)&a;  *(fft_input  +  6)    =  *(float *)&b;
//    a  =  0xbd95caf3  ;  b  =  0xbd66d267  ;  *(fft_input  +  9)    =  *(float *)&a;  *(fft_input  +  8)    =  *(float *)&b;
//    a  =  0x3cc20f6f  ;  b  =  0x3cd66cf4  ;  *(fft_input  +  11)   =  *(float *)&a;  *(fft_input  +  10)   =  *(float *)&b;
//    a  =  0x3d8d21bc  ;  b  =  0x3d980fdc  ;  *(fft_input  +  13)   =  *(float *)&a;  *(fft_input  +  12)   =  *(float *)&b;
//    a  =  0xbd870b07  ;  b  =  0xbcd1c7de  ;  *(fft_input  +  15)   =  *(float *)&a;  *(fft_input  +  14)   =  *(float *)&b;
//    a  =  0xbda6b378  ;  b  =  0x3e1f92b0  ;  *(fft_input  +  17)   =  *(float *)&a;  *(fft_input  +  16)   =  *(float *)&b;
//    a  =  0x3d9ad107  ;  b  =  0x3b82f0e1  ;  *(fft_input  +  19)   =  *(float *)&a;  *(fft_input  +  18)   =  *(float *)&b;
//    a  =  0xbcd0507a  ;  b  =  0xbca4fca4  ;  *(fft_input  +  21)   =  *(float *)&a;  *(fft_input  +  20)   =  *(float *)&b;
//    a  =  0xbda56e69  ;  b  =  0xbd6db1ea  ;  *(fft_input  +  23)   =  *(float *)&a;  *(fft_input  +  22)   =  *(float *)&b;
//    a  =  0xbc7b8f58  ;  b  =  0x3dc20ab7  ;  *(fft_input  +  25)   =  *(float *)&a;  *(fft_input  +  24)   =  *(float *)&b;
//    a  =  0x3daa91da  ;  b  =  0x3c95a4ad  ;  *(fft_input  +  27)   =  *(float *)&a;  *(fft_input  +  26)   =  *(float *)&b;
//    a  =  0x3cf38e6d  ;  b  =  0x3d39799e  ;  *(fft_input  +  29)   =  *(float *)&a;  *(fft_input  +  28)   =  *(float *)&b;
//    a  =  0x3d97121b  ;  b  =  0xbcfe64f5  ;  *(fft_input  +  31)   =  *(float *)&a;  *(fft_input  +  30)   =  *(float *)&b;
//    a  =  0x3d9d4174  ;  b  =  0x3e19231c  ;  *(fft_input  +  33)   =  *(float *)&a;  *(fft_input  +  32)   =  *(float *)&b;
//    a  =  0x3da45671  ;  b  =  0x3d83fc44  ;  *(fft_input  +  35)   =  *(float *)&a;  *(fft_input  +  34)   =  *(float *)&b;
//    a  =  0x3de3ab86  ;  b  =  0x3b3b1af4  ;  *(fft_input  +  37)   =  *(float *)&a;  *(fft_input  +  36)   =  *(float *)&b;
//    a  =  0x3d45e918  ;  b  =  0x3ce3497b  ;  *(fft_input  +  39)   =  *(float *)&a;  *(fft_input  +  38)   =  *(float *)&b;
//    a  =  0xbbe65387  ;  b  =  0xbd9eebb3  ;  *(fft_input  +  41)   =  *(float *)&a;  *(fft_input  +  40)   =  *(float *)&b;
//    a  =  0x3cd141a7  ;  b  =  0xbc5c5d64  ;  *(fft_input  +  43)   =  *(float *)&a;  *(fft_input  +  42)   =  *(float *)&b;
//    a  =  0x3d4da4cf  ;  b  =  0xbd88f755  ;  *(fft_input  +  45)   =  *(float *)&a;  *(fft_input  +  44)   =  *(float *)&b;
//    a  =  0xbd714014  ;  b  =  0xbd0af923  ;  *(fft_input  +  47)   =  *(float *)&a;  *(fft_input  +  46)   =  *(float *)&b;
//    a  =  0x3d523e5c  ;  b  =  0x3dac6ac2  ;  *(fft_input  +  49)   =  *(float *)&a;  *(fft_input  +  48)   =  *(float *)&b;
//    a  =  0x3cc63cfb  ;  b  =  0xbd445565  ;  *(fft_input  +  51)   =  *(float *)&a;  *(fft_input  +  50)   =  *(float *)&b;
//    a  =  0xbc848e04  ;  b  =  0x3d877920  ;  *(fft_input  +  53)   =  *(float *)&a;  *(fft_input  +  52)   =  *(float *)&b;
//    a  =  0x3d86e547  ;  b  =  0x3d692c49  ;  *(fft_input  +  55)   =  *(float *)&a;  *(fft_input  +  54)   =  *(float *)&b;
//    a  =  0x3d919e73  ;  b  =  0x3d99b995  ;  *(fft_input  +  57)   =  *(float *)&a;  *(fft_input  +  56)   =  *(float *)&b;
//    a  =  0x3da589ad  ;  b  =  0x3ccd5b68  ;  *(fft_input  +  59)   =  *(float *)&a;  *(fft_input  +  58)   =  *(float *)&b;
//    a  =  0x3d215db3  ;  b  =  0xbd71d1d4  ;  *(fft_input  +  61)   =  *(float *)&a;  *(fft_input  +  60)   =  *(float *)&b;
//    a  =  0x3ca1c68f  ;  b  =  0xbda2107b  ;  *(fft_input  +  63)   =  *(float *)&a;  *(fft_input  +  62)   =  *(float *)&b;
//    a  =  0xbdf1233e  ;  b  =  0x3da57d9e  ;  *(fft_input  +  65)   =  *(float *)&a;  *(fft_input  +  64)   =  *(float *)&b;
//    a  =  0x3d366517  ;  b  =  0x3cfbb949  ;  *(fft_input  +  67)   =  *(float *)&a;  *(fft_input  +  66)   =  *(float *)&b;
//    a  =  0x3d9faa04  ;  b  =  0x3ccffc98  ;  *(fft_input  +  69)   =  *(float *)&a;  *(fft_input  +  68)   =  *(float *)&b;
//    a  =  0xbd98c75c  ;  b  =  0x3bfb6dca  ;  *(fft_input  +  71)   =  *(float *)&a;  *(fft_input  +  70)   =  *(float *)&b;
//    a  =  0x3b1b413a  ;  b  =  0xbd44d445  ;  *(fft_input  +  73)   =  *(float *)&a;  *(fft_input  +  72)   =  *(float *)&b;
//    a  =  0x3d0fbca1  ;  b  =  0x3dc82f0e  ;  *(fft_input  +  75)   =  *(float *)&a;  *(fft_input  +  74)   =  *(float *)&b;
//    a  =  0x3d8b48d4  ;  b  =  0xbdfd82fd  ;  *(fft_input  +  77)   =  *(float *)&a;  *(fft_input  +  76)   =  *(float *)&b;
//    a  =  0x3d21c151  ;  b  =  0x3d5c3372  ;  *(fft_input  +  79)   =  *(float *)&a;  *(fft_input  +  78)   =  *(float *)&b;
//    a  =  0xbde19008  ;  b  =  0xbda084a5  ;  *(fft_input  +  81)   =  *(float *)&a;  *(fft_input  +  80)   =  *(float *)&b;
//    a  =  0xbba6223e  ;  b  =  0xbd4bc59c  ;  *(fft_input  +  83)   =  *(float *)&a;  *(fft_input  +  82)   =  *(float *)&b;
//    a  =  0xbbce8102  ;  b  =  0xbd02ecaf  ;  *(fft_input  +  85)   =  *(float *)&a;  *(fft_input  +  84)   =  *(float *)&b;
//    a  =  0xbd557796  ;  b  =  0x3d5101b0  ;  *(fft_input  +  87)   =  *(float *)&a;  *(fft_input  +  86)   =  *(float *)&b;
//    a  =  0xbd9b8130  ;  b  =  0x3d01d5c3  ;  *(fft_input  +  89)   =  *(float *)&a;  *(fft_input  +  88)   =  *(float *)&b;
//    a  =  0xbd88b439  ;  b  =  0x3d3c6da4  ;  *(fft_input  +  91)   =  *(float *)&a;  *(fft_input  +  90)   =  *(float *)&b;
//    a  =  0x3b9ab29e  ;  b  =  0x3d1e4e6a  ;  *(fft_input  +  93)   =  *(float *)&a;  *(fft_input  +  92)   =  *(float *)&b;
//    a  =  0x3da49a13  ;  b  =  0x3ce1e710  ;  *(fft_input  +  95)   =  *(float *)&a;  *(fft_input  +  94)   =  *(float *)&b;
//    a  =  0xbdff876a  ;  b  =  0x3d2f87ad  ;  *(fft_input  +  97)   =  *(float *)&a;  *(fft_input  +  96)   =  *(float *)&b;
//    a  =  0xbccdf04e  ;  b  =  0xbd1f1cfc  ;  *(fft_input  +  99)   =  *(float *)&a;  *(fft_input  +  98)   =  *(float *)&b;
//    a  =  0x3d144242  ;  b  =  0xbd0100e7  ;  *(fft_input  +  101)  =  *(float *)&a;  *(fft_input  +  100)  =  *(float *)&b;
//    a  =  0x3cbf1412  ;  b  =  0x3cd4de7f  ;  *(fft_input  +  103)  =  *(float *)&a;  *(fft_input  +  102)  =  *(float *)&b;
//    a  =  0x3d1b24ea  ;  b  =  0xbd4d887f  ;  *(fft_input  +  105)  =  *(float *)&a;  *(fft_input  +  104)  =  *(float *)&b;
//    a  =  0x3d391b3f  ;  b  =  0xbd864452  ;  *(fft_input  +  107)  =  *(float *)&a;  *(fft_input  +  106)  =  *(float *)&b;
//    a  =  0x3be94ee4  ;  b  =  0xbd36f9fd  ;  *(fft_input  +  109)  =  *(float *)&a;  *(fft_input  +  108)  =  *(float *)&b;
//    a  =  0x3d7d230c  ;  b  =  0x3d0b7e4e  ;  *(fft_input  +  111)  =  *(float *)&a;  *(fft_input  +  110)  =  *(float *)&b;
//    a  =  0xbde57e24  ;  b  =  0x3da0e842  ;  *(fft_input  +  113)  =  *(float *)&a;  *(fft_input  +  112)  =  *(float *)&b;
//    a  =  0xbe1ad081  ;  b  =  0xbd5013a9  ;  *(fft_input  +  115)  =  *(float *)&a;  *(fft_input  +  114)  =  *(float *)&b;
//    a  =  0x3be514c2  ;  b  =  0x3c53f9e8  ;  *(fft_input  +  117)  =  *(float *)&a;  *(fft_input  +  116)  =  *(float *)&b;
//    a  =  0xbe148451  ;  b  =  0x3cc0d0ab  ;  *(fft_input  +  119)  =  *(float *)&a;  *(fft_input  +  118)  =  *(float *)&b;
//    a  =  0x3bcc0bdd  ;  b  =  0x3d4aff6d  ;  *(fft_input  +  121)  =  *(float *)&a;  *(fft_input  +  120)  =  *(float *)&b;
//    a  =  0xbd3825e1  ;  b  =  0x3bf74470  ;  *(fft_input  +  123)  =  *(float *)&a;  *(fft_input  +  122)  =  *(float *)&b;
//    a  =  0xbce67f91  ;  b  =  0x3d679fed  ;  *(fft_input  +  125)  =  *(float *)&a;  *(fft_input  +  124)  =  *(float *)&b;
//    a  =  0xbde75ab8  ;  b  =  0xbd080410  ;  *(fft_input  +  127)  =  *(float *)&a;  *(fft_input  +  126)  =  *(float *)&b;
//    a  =  0xbe43e1ca  ;  b  =  0xbd96a0dc  ;  *(fft_input  +  129)  =  *(float *)&a;  *(fft_input  +  128)  =  *(float *)&b;
//    a  =  0x3da03eea  ;  b  =  0xbd62c882  ;  *(fft_input  +  131)  =  *(float *)&a;  *(fft_input  +  130)  =  *(float *)&b;
//    a  =  0x3c075d57  ;  b  =  0xbd9b0037  ;  *(fft_input  +  133)  =  *(float *)&a;  *(fft_input  +  132)  =  *(float *)&b;
//    a  =  0xbbf76e61  ;  b  =  0xbcae31d7  ;  *(fft_input  +  135)  =  *(float *)&a;  *(fft_input  +  134)  =  *(float *)&b;
//    a  =  0x3e0d0e99  ;  b  =  0x3c3bcb1d  ;  *(fft_input  +  137)  =  *(float *)&a;  *(fft_input  +  136)  =  *(float *)&b;
//    a  =  0x3bba27af  ;  b  =  0xbd2a1944  ;  *(fft_input  +  139)  =  *(float *)&a;  *(fft_input  +  138)  =  *(float *)&b;
//    a  =  0xbd40d0ab  ;  b  =  0x3cbb5e0f  ;  *(fft_input  +  141)  =  *(float *)&a;  *(fft_input  +  140)  =  *(float *)&b;
//    a  =  0x3cc128bf  ;  b  =  0xbe23b70f  ;  *(fft_input  +  143)  =  *(float *)&a;  *(fft_input  +  142)  =  *(float *)&b;
//    a  =  0x3e59fa98  ;  b  =  0xbd544135  ;  *(fft_input  +  145)  =  *(float *)&a;  *(fft_input  +  144)  =  *(float *)&b;
//    a  =  0x3d76f6d7  ;  b  =  0x3dcf0e0b  ;  *(fft_input  +  147)  =  *(float *)&a;  *(fft_input  +  146)  =  *(float *)&b;
//    a  =  0xbb7e4799  ;  b  =  0x3cf4afd5  ;  *(fft_input  +  149)  =  *(float *)&a;  *(fft_input  +  148)  =  *(float *)&b;
//    a  =  0xbd60e94f  ;  b  =  0x3d916c1e  ;  *(fft_input  +  151)  =  *(float *)&a;  *(fft_input  +  150)  =  *(float *)&b;
//    a  =  0xbdb1df76  ;  b  =  0x3cb7aa26  ;  *(fft_input  +  153)  =  *(float *)&a;  *(fft_input  +  152)  =  *(float *)&b;
//    a  =  0x3da6783a  ;  b  =  0x3cf631b6  ;  *(fft_input  +  155)  =  *(float *)&a;  *(fft_input  +  154)  =  *(float *)&b;
//    a  =  0x3db0a3d7  ;  b  =  0xbd38dd61  ;  *(fft_input  +  157)  =  *(float *)&a;  *(fft_input  +  156)  =  *(float *)&b;
//    a  =  0x3b4308ff  ;  b  =  0xbc97b310  ;  *(fft_input  +  159)  =  *(float *)&a;  *(fft_input  +  158)  =  *(float *)&b;
//    a  =  0xbdf7c3d7  ;  b  =  0xbe46644e  ;  *(fft_input  +  161)  =  *(float *)&a;  *(fft_input  +  160)  =  *(float *)&b;
//    a  =  0xbb1edbf9  ;  b  =  0x3da51c19  ;  *(fft_input  +  163)  =  *(float *)&a;  *(fft_input  +  162)  =  *(float *)&b;
//    a  =  0x3d2cbb80  ;  b  =  0x3d276ea4  ;  *(fft_input  +  165)  =  *(float *)&a;  *(fft_input  +  164)  =  *(float *)&b;
//    a  =  0xbd8fa1e4  ;  b  =  0x3d50d068  ;  *(fft_input  +  167)  =  *(float *)&a;  *(fft_input  +  166)  =  *(float *)&b;
//    a  =  0x3d2f2c73  ;  b  =  0x3ddb79d9  ;  *(fft_input  +  169)  =  *(float *)&a;  *(fft_input  +  168)  =  *(float *)&b;
//    a  =  0xbbe58a33  ;  b  =  0xbcc49dbf  ;  *(fft_input  +  171)  =  *(float *)&a;  *(fft_input  +  170)  =  *(float *)&b;
//    a  =  0xbb7e36d2  ;  b  =  0xbd0db9c7  ;  *(fft_input  +  173)  =  *(float *)&a;  *(fft_input  +  172)  =  *(float *)&b;
//    a  =  0xbc3ba55d  ;  b  =  0xbd738a3b  ;  *(fft_input  +  175)  =  *(float *)&a;  *(fft_input  +  174)  =  *(float *)&b;
//    a  =  0x3d7d48cb  ;  b  =  0x3be17a03  ;  *(fft_input  +  177)  =  *(float *)&a;  *(fft_input  +  176)  =  *(float *)&b;
//    a  =  0xbba89763  ;  b  =  0x3d969db6  ;  *(fft_input  +  179)  =  *(float *)&a;  *(fft_input  +  178)  =  *(float *)&b;
//    a  =  0x3b06594b  ;  b  =  0x3d8ad602  ;  *(fft_input  +  181)  =  *(float *)&a;  *(fft_input  +  180)  =  *(float *)&b;
//    a  =  0x3d39cd81  ;  b  =  0x3d5b5c7d  ;  *(fft_input  +  183)  =  *(float *)&a;  *(fft_input  +  182)  =  *(float *)&b;
//    a  =  0x3d3061c8  ;  b  =  0xbddfc116  ;  *(fft_input  +  185)  =  *(float *)&a;  *(fft_input  +  184)  =  *(float *)&b;
//    a  =  0xbd79d2bf  ;  b  =  0xbc82c6ef  ;  *(fft_input  +  187)  =  *(float *)&a;  *(fft_input  +  186)  =  *(float *)&b;
//    a  =  0x3d801712  ;  b  =  0xbc2caff7  ;  *(fft_input  +  189)  =  *(float *)&a;  *(fft_input  +  188)  =  *(float *)&b;
//    a  =  0xbda1b973  ;  b  =  0x391b3073  ;  *(fft_input  +  191)  =  *(float *)&a;  *(fft_input  +  190)  =  *(float *)&b;
//    a  =  0x3dc3e20d  ;  b  =  0xbe7122fb  ;  *(fft_input  +  193)  =  *(float *)&a;  *(fft_input  +  192)  =  *(float *)&b;
//    a  =  0x3cfd2631  ;  b  =  0x3c78a090  ;  *(fft_input  +  195)  =  *(float *)&a;  *(fft_input  +  194)  =  *(float *)&b;
//    a  =  0xbcce8102  ;  b  =  0x3cfee2ca  ;  *(fft_input  +  197)  =  *(float *)&a;  *(fft_input  +  196)  =  *(float *)&b;
//    a  =  0xbb92f6e8  ;  b  =  0x3da57fb7  ;  *(fft_input  +  199)  =  *(float *)&a;  *(fft_input  +  198)  =  *(float *)&b;
//    a  =  0x3caf661f  ;  b  =  0xbc6147ae  ;  *(fft_input  +  201)  =  *(float *)&a;  *(fft_input  +  200)  =  *(float *)&b;
//    a  =  0xbcea5508  ;  b  =  0x3d0c0ce9  ;  *(fft_input  +  203)  =  *(float *)&a;  *(fft_input  +  202)  =  *(float *)&b;
//    a  =  0x3d668901  ;  b  =  0xbd65d6bf  ;  *(fft_input  +  205)  =  *(float *)&a;  *(fft_input  +  204)  =  *(float *)&b;
//    a  =  0xbe02a6f4  ;  b  =  0xbce1975f  ;  *(fft_input  +  207)  =  *(float *)&a;  *(fft_input  +  206)  =  *(float *)&b;
//    a  =  0x3cbb44e5  ;  b  =  0x3e1ac4b5  ;  *(fft_input  +  209)  =  *(float *)&a;  *(fft_input  +  208)  =  *(float *)&b;
//    a  =  0x3d98e087  ;  b  =  0xbd81f7d7  ;  *(fft_input  +  211)  =  *(float *)&a;  *(fft_input  +  210)  =  *(float *)&b;
//    a  =  0x3d19ee89  ;  b  =  0x3c0e4fb9  ;  *(fft_input  +  213)  =  *(float *)&a;  *(fft_input  +  212)  =  *(float *)&b;
//    a  =  0x3cb4e54f  ;  b  =  0x3daf71a8  ;  *(fft_input  +  215)  =  *(float *)&a;  *(fft_input  +  214)  =  *(float *)&b;
//    a  =  0xbc196b76  ;  b  =  0xbdd65dc0  ;  *(fft_input  +  217)  =  *(float *)&a;  *(fft_input  +  216)  =  *(float *)&b;
//    a  =  0x3c9a75cd  ;  b  =  0x3da9d170  ;  *(fft_input  +  219)  =  *(float *)&a;  *(fft_input  +  218)  =  *(float *)&b;
//    a  =  0xbd66527a  ;  b  =  0x3d189481  ;  *(fft_input  +  221)  =  *(float *)&a;  *(fft_input  +  220)  =  *(float *)&b;
//    a  =  0x3dc8711d  ;  b  =  0x3d84ed70  ;  *(fft_input  +  223)  =  *(float *)&a;  *(fft_input  +  222)  =  *(float *)&b;
//    a  =  0x3dff876a  ;  b  =  0x3aafa2f0  ;  *(fft_input  +  225)  =  *(float *)&a;  *(fft_input  +  224)  =  *(float *)&b;
//    a  =  0x3cc8a58b  ;  b  =  0x3d1f2885  ;  *(fft_input  +  227)  =  *(float *)&a;  *(fft_input  +  226)  =  *(float *)&b;
//    a  =  0x3c9c7150  ;  b  =  0x3c03b603  ;  *(fft_input  +  229)  =  *(float *)&a;  *(fft_input  +  228)  =  *(float *)&b;
//    a  =  0x3c3ed30f  ;  b  =  0x3d909f1f  ;  *(fft_input  +  231)  =  *(float *)&a;  *(fft_input  +  230)  =  *(float *)&b;
//    a  =  0x3c720ea6  ;  b  =  0x3d1f676f  ;  *(fft_input  +  233)  =  *(float *)&a;  *(fft_input  +  232)  =  *(float *)&b;
//    a  =  0xbcd3e2d6  ;  b  =  0x3cb2d0a2  ;  *(fft_input  +  235)  =  *(float *)&a;  *(fft_input  +  234)  =  *(float *)&b;
//    a  =  0xbcbac92a  ;  b  =  0x3c7e974a  ;  *(fft_input  +  237)  =  *(float *)&a;  *(fft_input  +  236)  =  *(float *)&b;
//    a  =  0x3d85b4ab  ;  b  =  0x3b830a0b  ;  *(fft_input  +  239)  =  *(float *)&a;  *(fft_input  +  238)  =  *(float *)&b;
//    a  =  0x3e06a0dc  ;  b  =  0xbda665e0  ;  *(fft_input  +  241)  =  *(float *)&a;  *(fft_input  +  240)  =  *(float *)&b;
//    a  =  0xbd1b5b70  ;  b  =  0x3c3a0a52  ;  *(fft_input  +  243)  =  *(float *)&a;  *(fft_input  +  242)  =  *(float *)&b;
//    a  =  0x3d694035  ;  b  =  0xbd38151a  ;  *(fft_input  +  245)  =  *(float *)&a;  *(fft_input  +  244)  =  *(float *)&b;
//    a  =  0xbd88e68e  ;  b  =  0x3d23fcca  ;  *(fft_input  +  247)  =  *(float *)&a;  *(fft_input  +  246)  =  *(float *)&b;
//    a  =  0x3d896feb  ;  b  =  0x3cec2acc  ;  *(fft_input  +  249)  =  *(float *)&a;  *(fft_input  +  248)  =  *(float *)&b;
//    a  =  0xbde872b0  ;  b  =  0x3d9470eb  ;  *(fft_input  +  251)  =  *(float *)&a;  *(fft_input  +  250)  =  *(float *)&b;
//    a  =  0x3d1db552  ;  b  =  0xbcd8537e  ;  *(fft_input  +  253)  =  *(float *)&a;  *(fft_input  +  252)  =  *(float *)&b;
//    a  =  0xbd7940ff  ;  b  =  0x3daf6cf0  ;  *(fft_input  +  255)  =  *(float *)&a;  *(fft_input  +  254)  =  *(float *)&b;
//
//    a  =  0xbdc48c72  ;  b  =  0xbd174b40  ;  *(fft_input_ref  +  1)    = *(float *) &a;  *(fft_input_ref  +  0)    =  *(float *) &b;
//    a  =  0x3d118091  ;  b  =  0xbd7d7f52  ;  *(fft_input_ref  +  3)    = *(float *) &a;  *(fft_input_ref  +  2)    =  *(float *) &b;
//    a  =  0x3d039bcc  ;  b  =  0x3db2c732  ;  *(fft_input_ref  +  5)    = *(float *) &a;  *(fft_input_ref  +  4)    =  *(float *) &b;
//    a  =  0x3d1bf594  ;  b  =  0xbc6d6fdb  ;  *(fft_input_ref  +  7)    = *(float *) &a;  *(fft_input_ref  +  6)    =  *(float *) &b;
//    a  =  0xbd95caf3  ;  b  =  0xbd66d267  ;  *(fft_input_ref  +  9)    = *(float *) &a;  *(fft_input_ref  +  8)    =  *(float *) &b;
//    a  =  0x3cc20f6f  ;  b  =  0x3cd66cf4  ;  *(fft_input_ref  +  11)   = *(float *) &a;  *(fft_input_ref  +  10)   =  *(float *) &b;
//    a  =  0x3d8d21bc  ;  b  =  0x3d980fdc  ;  *(fft_input_ref  +  13)   = *(float *) &a;  *(fft_input_ref  +  12)   =  *(float *) &b;
//    a  =  0xbd870b07  ;  b  =  0xbcd1c7de  ;  *(fft_input_ref  +  15)   = *(float *) &a;  *(fft_input_ref  +  14)   =  *(float *) &b;
//    a  =  0xbda6b378  ;  b  =  0x3e1f92b0  ;  *(fft_input_ref  +  17)   = *(float *) &a;  *(fft_input_ref  +  16)   =  *(float *) &b;
//    a  =  0x3d9ad107  ;  b  =  0x3b82f0e1  ;  *(fft_input_ref  +  19)   = *(float *) &a;  *(fft_input_ref  +  18)   =  *(float *) &b;
//    a  =  0xbcd0507a  ;  b  =  0xbca4fca4  ;  *(fft_input_ref  +  21)   = *(float *) &a;  *(fft_input_ref  +  20)   =  *(float *) &b;
//    a  =  0xbda56e69  ;  b  =  0xbd6db1ea  ;  *(fft_input_ref  +  23)   = *(float *) &a;  *(fft_input_ref  +  22)   =  *(float *) &b;
//    a  =  0xbc7b8f58  ;  b  =  0x3dc20ab7  ;  *(fft_input_ref  +  25)   = *(float *) &a;  *(fft_input_ref  +  24)   =  *(float *) &b;
//    a  =  0x3daa91da  ;  b  =  0x3c95a4ad  ;  *(fft_input_ref  +  27)   = *(float *) &a;  *(fft_input_ref  +  26)   =  *(float *) &b;
//    a  =  0x3cf38e6d  ;  b  =  0x3d39799e  ;  *(fft_input_ref  +  29)   = *(float *) &a;  *(fft_input_ref  +  28)   =  *(float *) &b;
//    a  =  0x3d97121b  ;  b  =  0xbcfe64f5  ;  *(fft_input_ref  +  31)   = *(float *) &a;  *(fft_input_ref  +  30)   =  *(float *) &b;
//    a  =  0x3d9d4174  ;  b  =  0x3e19231c  ;  *(fft_input_ref  +  33)   = *(float *) &a;  *(fft_input_ref  +  32)   =  *(float *) &b;
//    a  =  0x3da45671  ;  b  =  0x3d83fc44  ;  *(fft_input_ref  +  35)   = *(float *) &a;  *(fft_input_ref  +  34)   =  *(float *) &b;
//    a  =  0x3de3ab86  ;  b  =  0x3b3b1af4  ;  *(fft_input_ref  +  37)   = *(float *) &a;  *(fft_input_ref  +  36)   =  *(float *) &b;
//    a  =  0x3d45e918  ;  b  =  0x3ce3497b  ;  *(fft_input_ref  +  39)   = *(float *) &a;  *(fft_input_ref  +  38)   =  *(float *) &b;
//    a  =  0xbbe65387  ;  b  =  0xbd9eebb3  ;  *(fft_input_ref  +  41)   = *(float *) &a;  *(fft_input_ref  +  40)   =  *(float *) &b;
//    a  =  0x3cd141a7  ;  b  =  0xbc5c5d64  ;  *(fft_input_ref  +  43)   = *(float *) &a;  *(fft_input_ref  +  42)   =  *(float *) &b;
//    a  =  0x3d4da4cf  ;  b  =  0xbd88f755  ;  *(fft_input_ref  +  45)   = *(float *) &a;  *(fft_input_ref  +  44)   =  *(float *) &b;
//    a  =  0xbd714014  ;  b  =  0xbd0af923  ;  *(fft_input_ref  +  47)   = *(float *) &a;  *(fft_input_ref  +  46)   =  *(float *) &b;
//    a  =  0x3d523e5c  ;  b  =  0x3dac6ac2  ;  *(fft_input_ref  +  49)   = *(float *) &a;  *(fft_input_ref  +  48)   =  *(float *) &b;
//    a  =  0x3cc63cfb  ;  b  =  0xbd445565  ;  *(fft_input_ref  +  51)   = *(float *) &a;  *(fft_input_ref  +  50)   =  *(float *) &b;
//    a  =  0xbc848e04  ;  b  =  0x3d877920  ;  *(fft_input_ref  +  53)   = *(float *) &a;  *(fft_input_ref  +  52)   =  *(float *) &b;
//    a  =  0x3d86e547  ;  b  =  0x3d692c49  ;  *(fft_input_ref  +  55)   = *(float *) &a;  *(fft_input_ref  +  54)   =  *(float *) &b;
//    a  =  0x3d919e73  ;  b  =  0x3d99b995  ;  *(fft_input_ref  +  57)   = *(float *) &a;  *(fft_input_ref  +  56)   =  *(float *) &b;
//    a  =  0x3da589ad  ;  b  =  0x3ccd5b68  ;  *(fft_input_ref  +  59)   = *(float *) &a;  *(fft_input_ref  +  58)   =  *(float *) &b;
//    a  =  0x3d215db3  ;  b  =  0xbd71d1d4  ;  *(fft_input_ref  +  61)   = *(float *) &a;  *(fft_input_ref  +  60)   =  *(float *) &b;
//    a  =  0x3ca1c68f  ;  b  =  0xbda2107b  ;  *(fft_input_ref  +  63)   = *(float *) &a;  *(fft_input_ref  +  62)   =  *(float *) &b;
//    a  =  0xbdf1233e  ;  b  =  0x3da57d9e  ;  *(fft_input_ref  +  65)   = *(float *) &a;  *(fft_input_ref  +  64)   =  *(float *) &b;
//    a  =  0x3d366517  ;  b  =  0x3cfbb949  ;  *(fft_input_ref  +  67)   = *(float *) &a;  *(fft_input_ref  +  66)   =  *(float *) &b;
//    a  =  0x3d9faa04  ;  b  =  0x3ccffc98  ;  *(fft_input_ref  +  69)   = *(float *) &a;  *(fft_input_ref  +  68)   =  *(float *) &b;
//    a  =  0xbd98c75c  ;  b  =  0x3bfb6dca  ;  *(fft_input_ref  +  71)   = *(float *) &a;  *(fft_input_ref  +  70)   =  *(float *) &b;
//    a  =  0x3b1b413a  ;  b  =  0xbd44d445  ;  *(fft_input_ref  +  73)   = *(float *) &a;  *(fft_input_ref  +  72)   =  *(float *) &b;
//    a  =  0x3d0fbca1  ;  b  =  0x3dc82f0e  ;  *(fft_input_ref  +  75)   = *(float *) &a;  *(fft_input_ref  +  74)   =  *(float *) &b;
//    a  =  0x3d8b48d4  ;  b  =  0xbdfd82fd  ;  *(fft_input_ref  +  77)   = *(float *) &a;  *(fft_input_ref  +  76)   =  *(float *) &b;
//    a  =  0x3d21c151  ;  b  =  0x3d5c3372  ;  *(fft_input_ref  +  79)   = *(float *) &a;  *(fft_input_ref  +  78)   =  *(float *) &b;
//    a  =  0xbde19008  ;  b  =  0xbda084a5  ;  *(fft_input_ref  +  81)   = *(float *) &a;  *(fft_input_ref  +  80)   =  *(float *) &b;
//    a  =  0xbba6223e  ;  b  =  0xbd4bc59c  ;  *(fft_input_ref  +  83)   = *(float *) &a;  *(fft_input_ref  +  82)   =  *(float *) &b;
//    a  =  0xbbce8102  ;  b  =  0xbd02ecaf  ;  *(fft_input_ref  +  85)   = *(float *) &a;  *(fft_input_ref  +  84)   =  *(float *) &b;
//    a  =  0xbd557796  ;  b  =  0x3d5101b0  ;  *(fft_input_ref  +  87)   = *(float *) &a;  *(fft_input_ref  +  86)   =  *(float *) &b;
//    a  =  0xbd9b8130  ;  b  =  0x3d01d5c3  ;  *(fft_input_ref  +  89)   = *(float *) &a;  *(fft_input_ref  +  88)   =  *(float *) &b;
//    a  =  0xbd88b439  ;  b  =  0x3d3c6da4  ;  *(fft_input_ref  +  91)   = *(float *) &a;  *(fft_input_ref  +  90)   =  *(float *) &b;
//    a  =  0x3b9ab29e  ;  b  =  0x3d1e4e6a  ;  *(fft_input_ref  +  93)   = *(float *) &a;  *(fft_input_ref  +  92)   =  *(float *) &b;
//    a  =  0x3da49a13  ;  b  =  0x3ce1e710  ;  *(fft_input_ref  +  95)   = *(float *) &a;  *(fft_input_ref  +  94)   =  *(float *) &b;
//    a  =  0xbdff876a  ;  b  =  0x3d2f87ad  ;  *(fft_input_ref  +  97)   = *(float *) &a;  *(fft_input_ref  +  96)   =  *(float *) &b;
//    a  =  0xbccdf04e  ;  b  =  0xbd1f1cfc  ;  *(fft_input_ref  +  99)   = *(float *) &a;  *(fft_input_ref  +  98)   =  *(float *) &b;
//    a  =  0x3d144242  ;  b  =  0xbd0100e7  ;  *(fft_input_ref  +  101)  = *(float *) &a;  *(fft_input_ref  +  100)  =  *(float *) &b;
//    a  =  0x3cbf1412  ;  b  =  0x3cd4de7f  ;  *(fft_input_ref  +  103)  = *(float *) &a;  *(fft_input_ref  +  102)  =  *(float *) &b;
//    a  =  0x3d1b24ea  ;  b  =  0xbd4d887f  ;  *(fft_input_ref  +  105)  = *(float *) &a;  *(fft_input_ref  +  104)  =  *(float *) &b;
//    a  =  0x3d391b3f  ;  b  =  0xbd864452  ;  *(fft_input_ref  +  107)  = *(float *) &a;  *(fft_input_ref  +  106)  =  *(float *) &b;
//    a  =  0x3be94ee4  ;  b  =  0xbd36f9fd  ;  *(fft_input_ref  +  109)  = *(float *) &a;  *(fft_input_ref  +  108)  =  *(float *) &b;
//    a  =  0x3d7d230c  ;  b  =  0x3d0b7e4e  ;  *(fft_input_ref  +  111)  = *(float *) &a;  *(fft_input_ref  +  110)  =  *(float *) &b;
//    a  =  0xbde57e24  ;  b  =  0x3da0e842  ;  *(fft_input_ref  +  113)  = *(float *) &a;  *(fft_input_ref  +  112)  =  *(float *) &b;
//    a  =  0xbe1ad081  ;  b  =  0xbd5013a9  ;  *(fft_input_ref  +  115)  = *(float *) &a;  *(fft_input_ref  +  114)  =  *(float *) &b;
//    a  =  0x3be514c2  ;  b  =  0x3c53f9e8  ;  *(fft_input_ref  +  117)  = *(float *) &a;  *(fft_input_ref  +  116)  =  *(float *) &b;
//    a  =  0xbe148451  ;  b  =  0x3cc0d0ab  ;  *(fft_input_ref  +  119)  = *(float *) &a;  *(fft_input_ref  +  118)  =  *(float *) &b;
//    a  =  0x3bcc0bdd  ;  b  =  0x3d4aff6d  ;  *(fft_input_ref  +  121)  = *(float *) &a;  *(fft_input_ref  +  120)  =  *(float *) &b;
//    a  =  0xbd3825e1  ;  b  =  0x3bf74470  ;  *(fft_input_ref  +  123)  = *(float *) &a;  *(fft_input_ref  +  122)  =  *(float *) &b;
//    a  =  0xbce67f91  ;  b  =  0x3d679fed  ;  *(fft_input_ref  +  125)  = *(float *) &a;  *(fft_input_ref  +  124)  =  *(float *) &b;
//    a  =  0xbde75ab8  ;  b  =  0xbd080410  ;  *(fft_input_ref  +  127)  = *(float *) &a;  *(fft_input_ref  +  126)  =  *(float *) &b;
//    a  =  0xbe43e1ca  ;  b  =  0xbd96a0dc  ;  *(fft_input_ref  +  129)  = *(float *) &a;  *(fft_input_ref  +  128)  =  *(float *) &b;
//    a  =  0x3da03eea  ;  b  =  0xbd62c882  ;  *(fft_input_ref  +  131)  = *(float *) &a;  *(fft_input_ref  +  130)  =  *(float *) &b;
//    a  =  0x3c075d57  ;  b  =  0xbd9b0037  ;  *(fft_input_ref  +  133)  = *(float *) &a;  *(fft_input_ref  +  132)  =  *(float *) &b;
//    a  =  0xbbf76e61  ;  b  =  0xbcae31d7  ;  *(fft_input_ref  +  135)  = *(float *) &a;  *(fft_input_ref  +  134)  =  *(float *) &b;
//    a  =  0x3e0d0e99  ;  b  =  0x3c3bcb1d  ;  *(fft_input_ref  +  137)  = *(float *) &a;  *(fft_input_ref  +  136)  =  *(float *) &b;
//    a  =  0x3bba27af  ;  b  =  0xbd2a1944  ;  *(fft_input_ref  +  139)  = *(float *) &a;  *(fft_input_ref  +  138)  =  *(float *) &b;
//    a  =  0xbd40d0ab  ;  b  =  0x3cbb5e0f  ;  *(fft_input_ref  +  141)  = *(float *) &a;  *(fft_input_ref  +  140)  =  *(float *) &b;
//    a  =  0x3cc128bf  ;  b  =  0xbe23b70f  ;  *(fft_input_ref  +  143)  = *(float *) &a;  *(fft_input_ref  +  142)  =  *(float *) &b;
//    a  =  0x3e59fa98  ;  b  =  0xbd544135  ;  *(fft_input_ref  +  145)  = *(float *) &a;  *(fft_input_ref  +  144)  =  *(float *) &b;
//    a  =  0x3d76f6d7  ;  b  =  0x3dcf0e0b  ;  *(fft_input_ref  +  147)  = *(float *) &a;  *(fft_input_ref  +  146)  =  *(float *) &b;
//    a  =  0xbb7e4799  ;  b  =  0x3cf4afd5  ;  *(fft_input_ref  +  149)  = *(float *) &a;  *(fft_input_ref  +  148)  =  *(float *) &b;
//    a  =  0xbd60e94f  ;  b  =  0x3d916c1e  ;  *(fft_input_ref  +  151)  = *(float *) &a;  *(fft_input_ref  +  150)  =  *(float *) &b;
//    a  =  0xbdb1df76  ;  b  =  0x3cb7aa26  ;  *(fft_input_ref  +  153)  = *(float *) &a;  *(fft_input_ref  +  152)  =  *(float *) &b;
//    a  =  0x3da6783a  ;  b  =  0x3cf631b6  ;  *(fft_input_ref  +  155)  = *(float *) &a;  *(fft_input_ref  +  154)  =  *(float *) &b;
//    a  =  0x3db0a3d7  ;  b  =  0xbd38dd61  ;  *(fft_input_ref  +  157)  = *(float *) &a;  *(fft_input_ref  +  156)  =  *(float *) &b;
//    a  =  0x3b4308ff  ;  b  =  0xbc97b310  ;  *(fft_input_ref  +  159)  = *(float *) &a;  *(fft_input_ref  +  158)  =  *(float *) &b;
//    a  =  0xbdf7c3d7  ;  b  =  0xbe46644e  ;  *(fft_input_ref  +  161)  = *(float *) &a;  *(fft_input_ref  +  160)  =  *(float *) &b;
//    a  =  0xbb1edbf9  ;  b  =  0x3da51c19  ;  *(fft_input_ref  +  163)  = *(float *) &a;  *(fft_input_ref  +  162)  =  *(float *) &b;
//    a  =  0x3d2cbb80  ;  b  =  0x3d276ea4  ;  *(fft_input_ref  +  165)  = *(float *) &a;  *(fft_input_ref  +  164)  =  *(float *) &b;
//    a  =  0xbd8fa1e4  ;  b  =  0x3d50d068  ;  *(fft_input_ref  +  167)  = *(float *) &a;  *(fft_input_ref  +  166)  =  *(float *) &b;
//    a  =  0x3d2f2c73  ;  b  =  0x3ddb79d9  ;  *(fft_input_ref  +  169)  = *(float *) &a;  *(fft_input_ref  +  168)  =  *(float *) &b;
//    a  =  0xbbe58a33  ;  b  =  0xbcc49dbf  ;  *(fft_input_ref  +  171)  = *(float *) &a;  *(fft_input_ref  +  170)  =  *(float *) &b;
//    a  =  0xbb7e36d2  ;  b  =  0xbd0db9c7  ;  *(fft_input_ref  +  173)  = *(float *) &a;  *(fft_input_ref  +  172)  =  *(float *) &b;
//    a  =  0xbc3ba55d  ;  b  =  0xbd738a3b  ;  *(fft_input_ref  +  175)  = *(float *) &a;  *(fft_input_ref  +  174)  =  *(float *) &b;
//    a  =  0x3d7d48cb  ;  b  =  0x3be17a03  ;  *(fft_input_ref  +  177)  = *(float *) &a;  *(fft_input_ref  +  176)  =  *(float *) &b;
//    a  =  0xbba89763  ;  b  =  0x3d969db6  ;  *(fft_input_ref  +  179)  = *(float *) &a;  *(fft_input_ref  +  178)  =  *(float *) &b;
//    a  =  0x3b06594b  ;  b  =  0x3d8ad602  ;  *(fft_input_ref  +  181)  = *(float *) &a;  *(fft_input_ref  +  180)  =  *(float *) &b;
//    a  =  0x3d39cd81  ;  b  =  0x3d5b5c7d  ;  *(fft_input_ref  +  183)  = *(float *) &a;  *(fft_input_ref  +  182)  =  *(float *) &b;
//    a  =  0x3d3061c8  ;  b  =  0xbddfc116  ;  *(fft_input_ref  +  185)  = *(float *) &a;  *(fft_input_ref  +  184)  =  *(float *) &b;
//    a  =  0xbd79d2bf  ;  b  =  0xbc82c6ef  ;  *(fft_input_ref  +  187)  = *(float *) &a;  *(fft_input_ref  +  186)  =  *(float *) &b;
//    a  =  0x3d801712  ;  b  =  0xbc2caff7  ;  *(fft_input_ref  +  189)  = *(float *) &a;  *(fft_input_ref  +  188)  =  *(float *) &b;
//    a  =  0xbda1b973  ;  b  =  0x391b3073  ;  *(fft_input_ref  +  191)  = *(float *) &a;  *(fft_input_ref  +  190)  =  *(float *) &b;
//    a  =  0x3dc3e20d  ;  b  =  0xbe7122fb  ;  *(fft_input_ref  +  193)  = *(float *) &a;  *(fft_input_ref  +  192)  =  *(float *) &b;
//    a  =  0x3cfd2631  ;  b  =  0x3c78a090  ;  *(fft_input_ref  +  195)  = *(float *) &a;  *(fft_input_ref  +  194)  =  *(float *) &b;
//    a  =  0xbcce8102  ;  b  =  0x3cfee2ca  ;  *(fft_input_ref  +  197)  = *(float *) &a;  *(fft_input_ref  +  196)  =  *(float *) &b;
//    a  =  0xbb92f6e8  ;  b  =  0x3da57fb7  ;  *(fft_input_ref  +  199)  = *(float *) &a;  *(fft_input_ref  +  198)  =  *(float *) &b;
//    a  =  0x3caf661f  ;  b  =  0xbc6147ae  ;  *(fft_input_ref  +  201)  = *(float *) &a;  *(fft_input_ref  +  200)  =  *(float *) &b;
//    a  =  0xbcea5508  ;  b  =  0x3d0c0ce9  ;  *(fft_input_ref  +  203)  = *(float *) &a;  *(fft_input_ref  +  202)  =  *(float *) &b;
//    a  =  0x3d668901  ;  b  =  0xbd65d6bf  ;  *(fft_input_ref  +  205)  = *(float *) &a;  *(fft_input_ref  +  204)  =  *(float *) &b;
//    a  =  0xbe02a6f4  ;  b  =  0xbce1975f  ;  *(fft_input_ref  +  207)  = *(float *) &a;  *(fft_input_ref  +  206)  =  *(float *) &b;
//    a  =  0x3cbb44e5  ;  b  =  0x3e1ac4b5  ;  *(fft_input_ref  +  209)  = *(float *) &a;  *(fft_input_ref  +  208)  =  *(float *) &b;
//    a  =  0x3d98e087  ;  b  =  0xbd81f7d7  ;  *(fft_input_ref  +  211)  = *(float *) &a;  *(fft_input_ref  +  210)  =  *(float *) &b;
//    a  =  0x3d19ee89  ;  b  =  0x3c0e4fb9  ;  *(fft_input_ref  +  213)  = *(float *) &a;  *(fft_input_ref  +  212)  =  *(float *) &b;
//    a  =  0x3cb4e54f  ;  b  =  0x3daf71a8  ;  *(fft_input_ref  +  215)  = *(float *) &a;  *(fft_input_ref  +  214)  =  *(float *) &b;
//    a  =  0xbc196b76  ;  b  =  0xbdd65dc0  ;  *(fft_input_ref  +  217)  = *(float *) &a;  *(fft_input_ref  +  216)  =  *(float *) &b;
//    a  =  0x3c9a75cd  ;  b  =  0x3da9d170  ;  *(fft_input_ref  +  219)  = *(float *) &a;  *(fft_input_ref  +  218)  =  *(float *) &b;
//    a  =  0xbd66527a  ;  b  =  0x3d189481  ;  *(fft_input_ref  +  221)  = *(float *) &a;  *(fft_input_ref  +  220)  =  *(float *) &b;
//    a  =  0x3dc8711d  ;  b  =  0x3d84ed70  ;  *(fft_input_ref  +  223)  = *(float *) &a;  *(fft_input_ref  +  222)  =  *(float *) &b;
//    a  =  0x3dff876a  ;  b  =  0x3aafa2f0  ;  *(fft_input_ref  +  225)  = *(float *) &a;  *(fft_input_ref  +  224)  =  *(float *) &b;
//    a  =  0x3cc8a58b  ;  b  =  0x3d1f2885  ;  *(fft_input_ref  +  227)  = *(float *) &a;  *(fft_input_ref  +  226)  =  *(float *) &b;
//    a  =  0x3c9c7150  ;  b  =  0x3c03b603  ;  *(fft_input_ref  +  229)  = *(float *) &a;  *(fft_input_ref  +  228)  =  *(float *) &b;
//    a  =  0x3c3ed30f  ;  b  =  0x3d909f1f  ;  *(fft_input_ref  +  231)  = *(float *) &a;  *(fft_input_ref  +  230)  =  *(float *) &b;
//    a  =  0x3c720ea6  ;  b  =  0x3d1f676f  ;  *(fft_input_ref  +  233)  = *(float *) &a;  *(fft_input_ref  +  232)  =  *(float *) &b;
//    a  =  0xbcd3e2d6  ;  b  =  0x3cb2d0a2  ;  *(fft_input_ref  +  235)  = *(float *) &a;  *(fft_input_ref  +  234)  =  *(float *) &b;
//    a  =  0xbcbac92a  ;  b  =  0x3c7e974a  ;  *(fft_input_ref  +  237)  = *(float *) &a;  *(fft_input_ref  +  236)  =  *(float *) &b;
//    a  =  0x3d85b4ab  ;  b  =  0x3b830a0b  ;  *(fft_input_ref  +  239)  = *(float *) &a;  *(fft_input_ref  +  238)  =  *(float *) &b;
//    a  =  0x3e06a0dc  ;  b  =  0xbda665e0  ;  *(fft_input_ref  +  241)  = *(float *) &a;  *(fft_input_ref  +  240)  =  *(float *) &b;
//    a  =  0xbd1b5b70  ;  b  =  0x3c3a0a52  ;  *(fft_input_ref  +  243)  = *(float *) &a;  *(fft_input_ref  +  242)  =  *(float *) &b;
//    a  =  0x3d694035  ;  b  =  0xbd38151a  ;  *(fft_input_ref  +  245)  = *(float *) &a;  *(fft_input_ref  +  244)  =  *(float *) &b;
//    a  =  0xbd88e68e  ;  b  =  0x3d23fcca  ;  *(fft_input_ref  +  247)  = *(float *) &a;  *(fft_input_ref  +  246)  =  *(float *) &b;
//    a  =  0x3d896feb  ;  b  =  0x3cec2acc  ;  *(fft_input_ref  +  249)  = *(float *) &a;  *(fft_input_ref  +  248)  =  *(float *) &b;
//    a  =  0xbde872b0  ;  b  =  0x3d9470eb  ;  *(fft_input_ref  +  251)  = *(float *) &a;  *(fft_input_ref  +  250)  =  *(float *) &b;
//    a  =  0x3d1db552  ;  b  =  0xbcd8537e  ;  *(fft_input_ref  +  253)  = *(float *) &a;  *(fft_input_ref  +  252)  =  *(float *) &b;
//    a  =  0xbd7940ff  ;  b  =  0x3daf6cf0  ;  *(fft_input_ref  +  255)  = *(float *) &a;  *(fft_input_ref  +  254)  =  *(float *) &b;
}

void gen_ref_result (TYPE *fft_output_ref) { 
    float fft_ref[256];
    unsigned int a, b;

    for (int i = 0; i < 2 * DIM; i++) {
      fft_output_ref[i] = (TYPE) 1;
    }

//    a  =  0x3F35053A  ;  b  =  0x3F3504E6  ;  fft_output_ref[1]    =  *(float *)&a;  fft_output_ref[0]    =  *(float *)&b;
//    a  =  0x3F35051E  ;  b  =  0x3F3504FE  ;  fft_output_ref[3]    =  *(float *)&a;  fft_output_ref[2]    =  *(float *)&b;
//    a  =  0xBF3504D9  ;  b  =  0x3F3504AF  ;  fft_output_ref[5]    =  *(float *)&a;  fft_output_ref[4]    =  *(float *)&b;
//    a  =  0xBF3504E7  ;  b  =  0xBF3504FD  ;  fft_output_ref[7]    =  *(float *)&a;  fft_output_ref[6]    =  *(float *)&b;
//    a  =  0xBF3504E0  ;  b  =  0xBF3504B6  ;  fft_output_ref[9]    =  *(float *)&a;  fft_output_ref[8]    =  *(float *)&b;
//    a  =  0xBF350491  ;  b  =  0xBF350506  ;  fft_output_ref[11]   =  *(float *)&a;  fft_output_ref[10]   =  *(float *)&b;
//    a  =  0xBF35053D  ;  b  =  0x3F35050E  ;  fft_output_ref[13]   =  *(float *)&a;  fft_output_ref[12]   =  *(float *)&b;
//    a  =  0x3F800036  ;  b  =  0xBF7FFFEA  ;  fft_output_ref[15]   =  *(float *)&a;  fft_output_ref[14]   =  *(float *)&b;
//    a  =  0x3F350534  ;  b  =  0x3F350591  ;  fft_output_ref[17]   =  *(float *)&a;  fft_output_ref[16]   =  *(float *)&b;
//    a  =  0xBF3504F8  ;  b  =  0x3F3504C4  ;  fft_output_ref[19]   =  *(float *)&a;  fft_output_ref[18]   =  *(float *)&b;
//    a  =  0x3F3504F3  ;  b  =  0x3F35049A  ;  fft_output_ref[21]   =  *(float *)&a;  fft_output_ref[20]   =  *(float *)&b;
//    a  =  0xBF3504C9  ;  b  =  0x3F35051F  ;  fft_output_ref[23]   =  *(float *)&a;  fft_output_ref[22]   =  *(float *)&b;
//    a  =  0xBF3504F5  ;  b  =  0xBF35050B  ;  fft_output_ref[25]   =  *(float *)&a;  fft_output_ref[24]   =  *(float *)&b;
//    a  =  0x3F350503  ;  b  =  0xBF3504AA  ;  fft_output_ref[27]   =  *(float *)&a;  fft_output_ref[26]   =  *(float *)&b;
//    a  =  0x3F350552  ;  b  =  0x3F3504BD  ;  fft_output_ref[29]   =  *(float *)&a;  fft_output_ref[28]   =  *(float *)&b;
//    a  =  0x3F7FFFE8  ;  b  =  0x3F7FFFF4  ;  fft_output_ref[31]   =  *(float *)&a;  fft_output_ref[30]   =  *(float *)&b;
//    a  =  0xBF3504F4  ;  b  =  0xBF3504E8  ;  fft_output_ref[33]   =  *(float *)&a;  fft_output_ref[32]   =  *(float *)&b;
//    a  =  0x3F3504CF  ;  b  =  0x3F3504D0  ;  fft_output_ref[35]   =  *(float *)&a;  fft_output_ref[34]   =  *(float *)&b;
//    a  =  0x3F3504E9  ;  b  =  0xBF3504C7  ;  fft_output_ref[37]   =  *(float *)&a;  fft_output_ref[36]   =  *(float *)&b;
//    a  =  0xBF350594  ;  b  =  0xBF3504C6  ;  fft_output_ref[39]   =  *(float *)&a;  fft_output_ref[38]   =  *(float *)&b;
//    a  =  0xBF350515  ;  b  =  0x3F350553  ;  fft_output_ref[41]   =  *(float *)&a;  fft_output_ref[40]   =  *(float *)&b;
//    a  =  0xBF35052A  ;  b  =  0xBF3504D2  ;  fft_output_ref[43]   =  *(float *)&a;  fft_output_ref[42]   =  *(float *)&b;
//    a  =  0xBF350513  ;  b  =  0x3F3504ED  ;  fft_output_ref[45]   =  *(float *)&a;  fft_output_ref[44]   =  *(float *)&b;
//    a  =  0x3F800013  ;  b  =  0x3F7FFFDC  ;  fft_output_ref[47]   =  *(float *)&a;  fft_output_ref[46]   =  *(float *)&b;
//    a  =  0xBF35055D  ;  b  =  0xBF35053D  ;  fft_output_ref[49]   =  *(float *)&a;  fft_output_ref[48]   =  *(float *)&b;
//    a  =  0xBF3504E6  ;  b  =  0xBF3504F5  ;  fft_output_ref[51]   =  *(float *)&a;  fft_output_ref[50]   =  *(float *)&b;
//    a  =  0xBF3504EB  ;  b  =  0xBF35055B  ;  fft_output_ref[53]   =  *(float *)&a;  fft_output_ref[52]   =  *(float *)&b;
//    a  =  0x3F3504B3  ;  b  =  0x3F350568  ;  fft_output_ref[55]   =  *(float *)&a;  fft_output_ref[54]   =  *(float *)&b;
//    a  =  0x3F3504E7  ;  b  =  0x3F3504FD  ;  fft_output_ref[57]   =  *(float *)&a;  fft_output_ref[56]   =  *(float *)&b;
//    a  =  0xBF3504D5  ;  b  =  0xBF3504FC  ;  fft_output_ref[59]   =  *(float *)&a;  fft_output_ref[58]   =  *(float *)&b;
//    a  =  0x3F3504B4  ;  b  =  0x3F3504D7  ;  fft_output_ref[61]   =  *(float *)&a;  fft_output_ref[60]   =  *(float *)&b;
//    a  =  0x3F800016  ;  b  =  0x3F7FFFB5  ;  fft_output_ref[63]   =  *(float *)&a;  fft_output_ref[62]   =  *(float *)&b;
//    a  =  0xBF350473  ;  b  =  0x3F3504E6  ;  fft_output_ref[65]   =  *(float *)&a;  fft_output_ref[64]   =  *(float *)&b;
//    a  =  0xBF3504CB  ;  b  =  0xBF3504C7  ;  fft_output_ref[67]   =  *(float *)&a;  fft_output_ref[66]   =  *(float *)&b;
//    a  =  0xBF3504FF  ;  b  =  0xBF3504F7  ;  fft_output_ref[69]   =  *(float *)&a;  fft_output_ref[68]   =  *(float *)&b;
//    a  =  0xBF35055D  ;  b  =  0xBF350506  ;  fft_output_ref[71]   =  *(float *)&a;  fft_output_ref[70]   =  *(float *)&b;
//    a  =  0xBF3504E9  ;  b  =  0xBF3504B9  ;  fft_output_ref[73]   =  *(float *)&a;  fft_output_ref[72]   =  *(float *)&b;
//    a  =  0xBF3504C6  ;  b  =  0xBF35057C  ;  fft_output_ref[75]   =  *(float *)&a;  fft_output_ref[74]   =  *(float *)&b;
//    a  =  0x3F3504F6  ;  b  =  0x3F3504E0  ;  fft_output_ref[77]   =  *(float *)&a;  fft_output_ref[76]   =  *(float *)&b;
//    a  =  0x3F800033  ;  b  =  0xBF800014  ;  fft_output_ref[79]   =  *(float *)&a;  fft_output_ref[78]   =  *(float *)&b;
//    a  =  0xBF3504AD  ;  b  =  0xBF350507  ;  fft_output_ref[81]   =  *(float *)&a;  fft_output_ref[80]   =  *(float *)&b;
//    a  =  0x3F35050D  ;  b  =  0xBF3504EA  ;  fft_output_ref[83]   =  *(float *)&a;  fft_output_ref[82]   =  *(float *)&b;
//    a  =  0x3F350527  ;  b  =  0x3F3504EF  ;  fft_output_ref[85]   =  *(float *)&a;  fft_output_ref[84]   =  *(float *)&b;
//    a  =  0x3F3504D1  ;  b  =  0x3F3504E1  ;  fft_output_ref[87]   =  *(float *)&a;  fft_output_ref[86]   =  *(float *)&b;
//    a  =  0x3F35053D  ;  b  =  0x3F3504B9  ;  fft_output_ref[89]   =  *(float *)&a;  fft_output_ref[88]   =  *(float *)&b;
//    a  =  0x3F35052B  ;  b  =  0xBF3504E1  ;  fft_output_ref[91]   =  *(float *)&a;  fft_output_ref[90]   =  *(float *)&b;
//    a  =  0xBF35054D  ;  b  =  0xBF350564  ;  fft_output_ref[93]   =  *(float *)&a;  fft_output_ref[92]   =  *(float *)&b;
//    a  =  0x3F7FFFB9  ;  b  =  0x3F7FFFE5  ;  fft_output_ref[95]   =  *(float *)&a;  fft_output_ref[94]   =  *(float *)&b;
//    a  =  0xBF350512  ;  b  =  0x3F3504D7  ;  fft_output_ref[97]   =  *(float *)&a;  fft_output_ref[96]   =  *(float *)&b;
//    a  =  0x3F3504DF  ;  b  =  0x3F350513  ;  fft_output_ref[99]   =  *(float *)&a;  fft_output_ref[98]   =  *(float *)&b;
//    a  =  0xBF3504B8  ;  b  =  0xBF35053E  ;  fft_output_ref[101]  =  *(float *)&a;  fft_output_ref[100]  =  *(float *)&b;
//    a  =  0xBF3504AB  ;  b  =  0xBF3504E5  ;  fft_output_ref[103]  =  *(float *)&a;  fft_output_ref[102]  =  *(float *)&b;
//    a  =  0xBF3504E4  ;  b  =  0x3F3504F3  ;  fft_output_ref[105]  =  *(float *)&a;  fft_output_ref[104]  =  *(float *)&b;
//    a  =  0x3F3504A9  ;  b  =  0x3F3504FB  ;  fft_output_ref[107]  =  *(float *)&a;  fft_output_ref[106]  =  *(float *)&b;
//    a  =  0xBF3504B6  ;  b  =  0x3F3504F2  ;  fft_output_ref[109]  =  *(float *)&a;  fft_output_ref[108]  =  *(float *)&b;
//    a  =  0x3F7FFFD4  ;  b  =  0x3F800010  ;  fft_output_ref[111]  =  *(float *)&a;  fft_output_ref[110]  =  *(float *)&b;
//    a  =  0xBF3504F6  ;  b  =  0xBF350511  ;  fft_output_ref[113]  =  *(float *)&a;  fft_output_ref[112]  =  *(float *)&b;
//    a  =  0xBF3504B0  ;  b  =  0xBF350508  ;  fft_output_ref[115]  =  *(float *)&a;  fft_output_ref[114]  =  *(float *)&b;
//    a  =  0x3F35049D  ;  b  =  0x3F3504ED  ;  fft_output_ref[117]  =  *(float *)&a;  fft_output_ref[116]  =  *(float *)&b;
//    a  =  0x3F35050E  ;  b  =  0x3F3504CD  ;  fft_output_ref[119]  =  *(float *)&a;  fft_output_ref[118]  =  *(float *)&b;
//    a  =  0x3F350484  ;  b  =  0x3F3504BF  ;  fft_output_ref[121]  =  *(float *)&a;  fft_output_ref[120]  =  *(float *)&b;
//    a  =  0x3F350505  ;  b  =  0xBF3504C8  ;  fft_output_ref[123]  =  *(float *)&a;  fft_output_ref[122]  =  *(float *)&b;
//    a  =  0x3F350555  ;  b  =  0x3F3504E7  ;  fft_output_ref[125]  =  *(float *)&a;  fft_output_ref[124]  =  *(float *)&b;
//    a  =  0x3F7FFFE9  ;  b  =  0x3F800027  ;  fft_output_ref[127]  =  *(float *)&a;  fft_output_ref[126]  =  *(float *)&b;
//    a  =  0x3F350518  ;  b  =  0xBF350508  ;  fft_output_ref[129]  =  *(float *)&a;  fft_output_ref[128]  =  *(float *)&b;
//    a  =  0xBF350527  ;  b  =  0xBF35050F  ;  fft_output_ref[131]  =  *(float *)&a;  fft_output_ref[130]  =  *(float *)&b;
//    a  =  0xBF350500  ;  b  =  0x3F3504A1  ;  fft_output_ref[133]  =  *(float *)&a;  fft_output_ref[132]  =  *(float *)&b;
//    a  =  0xBF3504C5  ;  b  =  0xBF35048B  ;  fft_output_ref[135]  =  *(float *)&a;  fft_output_ref[134]  =  *(float *)&b;
//    a  =  0x3F35052A  ;  b  =  0xBF3504C8  ;  fft_output_ref[137]  =  *(float *)&a;  fft_output_ref[136]  =  *(float *)&b;
//    a  =  0xBF3504C5  ;  b  =  0xBF35053A  ;  fft_output_ref[139]  =  *(float *)&a;  fft_output_ref[138]  =  *(float *)&b;
//    a  =  0xBF35054C  ;  b  =  0xBF350505  ;  fft_output_ref[141]  =  *(float *)&a;  fft_output_ref[140]  =  *(float *)&b;
//    a  =  0x3F800004  ;  b  =  0xBF7FFFAA  ;  fft_output_ref[143]  =  *(float *)&a;  fft_output_ref[142]  =  *(float *)&b;
//    a  =  0xBF3504F1  ;  b  =  0xBF3504A7  ;  fft_output_ref[145]  =  *(float *)&a;  fft_output_ref[144]  =  *(float *)&b;
//    a  =  0xBF350551  ;  b  =  0x3F35050F  ;  fft_output_ref[147]  =  *(float *)&a;  fft_output_ref[146]  =  *(float *)&b;
//    a  =  0xBF350543  ;  b  =  0xBF3504C1  ;  fft_output_ref[149]  =  *(float *)&a;  fft_output_ref[148]  =  *(float *)&b;
//    a  =  0x3F350472  ;  b  =  0x3F35052C  ;  fft_output_ref[151]  =  *(float *)&a;  fft_output_ref[150]  =  *(float *)&b;
//    a  =  0xBF35052F  ;  b  =  0xBF3504C5  ;  fft_output_ref[153]  =  *(float *)&a;  fft_output_ref[152]  =  *(float *)&b;
//    a  =  0xBF350507  ;  b  =  0xBF35052C  ;  fft_output_ref[155]  =  *(float *)&a;  fft_output_ref[154]  =  *(float *)&b;
//    a  =  0xBF35050B  ;  b  =  0xBF35051D  ;  fft_output_ref[157]  =  *(float *)&a;  fft_output_ref[156]  =  *(float *)&b;
//    a  =  0x3F800009  ;  b  =  0x3F7FFFEF  ;  fft_output_ref[159]  =  *(float *)&a;  fft_output_ref[158]  =  *(float *)&b;
//    a  =  0xBF3504F9  ;  b  =  0x3F3504A2  ;  fft_output_ref[161]  =  *(float *)&a;  fft_output_ref[160]  =  *(float *)&b;
//    a  =  0xBF35048B  ;  b  =  0xBF3504C3  ;  fft_output_ref[163]  =  *(float *)&a;  fft_output_ref[162]  =  *(float *)&b;
//    a  =  0x3F3504AD  ;  b  =  0x3F3504C7  ;  fft_output_ref[165]  =  *(float *)&a;  fft_output_ref[164]  =  *(float *)&b;
//    a  =  0xBF3504B0  ;  b  =  0xBF35046E  ;  fft_output_ref[167]  =  *(float *)&a;  fft_output_ref[166]  =  *(float *)&b;
//    a  =  0xBF35052D  ;  b  =  0xBF35054D  ;  fft_output_ref[169]  =  *(float *)&a;  fft_output_ref[168]  =  *(float *)&b;
//    a  =  0xBF3504DF  ;  b  =  0xBF350512  ;  fft_output_ref[171]  =  *(float *)&a;  fft_output_ref[170]  =  *(float *)&b;
//    a  =  0xBF350512  ;  b  =  0x3F3504EB  ;  fft_output_ref[173]  =  *(float *)&a;  fft_output_ref[172]  =  *(float *)&b;
//    a  =  0x3F80000B  ;  b  =  0x3F800030  ;  fft_output_ref[175]  =  *(float *)&a;  fft_output_ref[174]  =  *(float *)&b;
//    a  =  0xBF3504C9  ;  b  =  0xBF35051A  ;  fft_output_ref[177]  =  *(float *)&a;  fft_output_ref[176]  =  *(float *)&b;
//    a  =  0xBF350549  ;  b  =  0x3F35054E  ;  fft_output_ref[179]  =  *(float *)&a;  fft_output_ref[178]  =  *(float *)&b;
//    a  =  0xBF3504ED  ;  b  =  0xBF3504FB  ;  fft_output_ref[181]  =  *(float *)&a;  fft_output_ref[180]  =  *(float *)&b;
//    a  =  0xBF35052C  ;  b  =  0x3F3504F3  ;  fft_output_ref[183]  =  *(float *)&a;  fft_output_ref[182]  =  *(float *)&b;
//    a  =  0xBF35050D  ;  b  =  0xBF350573  ;  fft_output_ref[185]  =  *(float *)&a;  fft_output_ref[184]  =  *(float *)&b;
//    a  =  0xBF35051E  ;  b  =  0x3F350548  ;  fft_output_ref[187]  =  *(float *)&a;  fft_output_ref[186]  =  *(float *)&b;
//    a  =  0xBF350481  ;  b  =  0xBF3504D5  ;  fft_output_ref[189]  =  *(float *)&a;  fft_output_ref[188]  =  *(float *)&b;
//    a  =  0x3F7FFFC3  ;  b  =  0x3F80000F  ;  fft_output_ref[191]  =  *(float *)&a;  fft_output_ref[190]  =  *(float *)&b;
//    a  =  0xBF350517  ;  b  =  0xBF350508  ;  fft_output_ref[193]  =  *(float *)&a;  fft_output_ref[192]  =  *(float *)&b;
//    a  =  0xBF35049D  ;  b  =  0xBF3504C5  ;  fft_output_ref[195]  =  *(float *)&a;  fft_output_ref[194]  =  *(float *)&b;
//    a  =  0x3F350507  ;  b  =  0x3F350582  ;  fft_output_ref[197]  =  *(float *)&a;  fft_output_ref[196]  =  *(float *)&b;
//    a  =  0xBF3504C0  ;  b  =  0xBF3504E8  ;  fft_output_ref[199]  =  *(float *)&a;  fft_output_ref[198]  =  *(float *)&b;
//    a  =  0x3F35054C  ;  b  =  0xBF350504  ;  fft_output_ref[201]  =  *(float *)&a;  fft_output_ref[200]  =  *(float *)&b;
//    a  =  0xBF3504ED  ;  b  =  0xBF350501  ;  fft_output_ref[203]  =  *(float *)&a;  fft_output_ref[202]  =  *(float *)&b;
//    a  =  0xBF3504E1  ;  b  =  0x3F3504C6  ;  fft_output_ref[205]  =  *(float *)&a;  fft_output_ref[204]  =  *(float *)&b;
//    a  =  0x3F80000D  ;  b  =  0xBF800016  ;  fft_output_ref[207]  =  *(float *)&a;  fft_output_ref[206]  =  *(float *)&b;
//    a  =  0xBF350522  ;  b  =  0xBF3504E6  ;  fft_output_ref[209]  =  *(float *)&a;  fft_output_ref[208]  =  *(float *)&b;
//    a  =  0x3F3504CD  ;  b  =  0x3F3504A8  ;  fft_output_ref[211]  =  *(float *)&a;  fft_output_ref[210]  =  *(float *)&b;
//    a  =  0xBF35050F  ;  b  =  0xBF35052F  ;  fft_output_ref[213]  =  *(float *)&a;  fft_output_ref[212]  =  *(float *)&b;
//    a  =  0x3F3504C5  ;  b  =  0x3F3504E0  ;  fft_output_ref[215]  =  *(float *)&a;  fft_output_ref[214]  =  *(float *)&b;
//    a  =  0xBF3504DE  ;  b  =  0xBF3504FC  ;  fft_output_ref[217]  =  *(float *)&a;  fft_output_ref[216]  =  *(float *)&b;
//    a  =  0x3F3504C2  ;  b  =  0x3F3504E7  ;  fft_output_ref[219]  =  *(float *)&a;  fft_output_ref[218]  =  *(float *)&b;
//    a  =  0xBF3504C2  ;  b  =  0xBF3504E2  ;  fft_output_ref[221]  =  *(float *)&a;  fft_output_ref[220]  =  *(float *)&b;
//    a  =  0x3F7FFFB6  ;  b  =  0x3F800016  ;  fft_output_ref[223]  =  *(float *)&a;  fft_output_ref[222]  =  *(float *)&b;
//    a  =  0x3F3504AD  ;  b  =  0xBF3504D5  ;  fft_output_ref[225]  =  *(float *)&a;  fft_output_ref[224]  =  *(float *)&b;
//    a  =  0xBF350521  ;  b  =  0xBF350557  ;  fft_output_ref[227]  =  *(float *)&a;  fft_output_ref[226]  =  *(float *)&b;
//    a  =  0xBF3504BD  ;  b  =  0x3F35051F  ;  fft_output_ref[229]  =  *(float *)&a;  fft_output_ref[228]  =  *(float *)&b;
//    a  =  0xBF35050F  ;  b  =  0xBF3504FA  ;  fft_output_ref[231]  =  *(float *)&a;  fft_output_ref[230]  =  *(float *)&b;
//    a  =  0xBF35051F  ;  b  =  0xBF3504D1  ;  fft_output_ref[233]  =  *(float *)&a;  fft_output_ref[232]  =  *(float *)&b;
//    a  =  0xBF350538  ;  b  =  0xBF350509  ;  fft_output_ref[235]  =  *(float *)&a;  fft_output_ref[234]  =  *(float *)&b;
//    a  =  0xBF3504A0  ;  b  =  0xBF350509  ;  fft_output_ref[237]  =  *(float *)&a;  fft_output_ref[236]  =  *(float *)&b;
//    a  =  0x3F7FFF92  ;  b  =  0x3F7FFF88  ;  fft_output_ref[239]  =  *(float *)&a;  fft_output_ref[238]  =  *(float *)&b;
//    a  =  0xBF350502  ;  b  =  0xBF3504BA  ;  fft_output_ref[241]  =  *(float *)&a;  fft_output_ref[240]  =  *(float *)&b;
//    a  =  0x3F3504DB  ;  b  =  0x3F35049C  ;  fft_output_ref[243]  =  *(float *)&a;  fft_output_ref[242]  =  *(float *)&b;
//    a  =  0xBF3504F6  ;  b  =  0xBF3504F7  ;  fft_output_ref[245]  =  *(float *)&a;  fft_output_ref[244]  =  *(float *)&b;
//    a  =  0x3F3504A0  ;  b  =  0x3F350547  ;  fft_output_ref[247]  =  *(float *)&a;  fft_output_ref[246]  =  *(float *)&b;
//    a  =  0xBF3504A9  ;  b  =  0xBF3504C8  ;  fft_output_ref[249]  =  *(float *)&a;  fft_output_ref[248]  =  *(float *)&b;
//    a  =  0xBF35052F  ;  b  =  0x3F3504F8  ;  fft_output_ref[251]  =  *(float *)&a;  fft_output_ref[250]  =  *(float *)&b;
//    a  =  0xBF35050A  ;  b  =  0xBF350538  ;  fft_output_ref[253]  =  *(float *)&a;  fft_output_ref[252]  =  *(float *)&b;
//    a  =  0x3F800021  ;  b  =  0x3F80001D  ;  fft_output_ref[255]  =  *(float *)&a;  fft_output_ref[254]  =  *(float *)&b;
}

//###################################################################################
// Function - Check Result of FFT
//###################################################################################
void check_result(TYPE *sw, TYPE *hw) {


    int error_count = 0;
    float diff;
    float c, d;
    
    for (int i = 0; i < DIM * 2; i++) {
        c = sw[i];
        d = hw[i];
        diff = fabs(c - d) / c * 100;

        if (diff > 0.01) {
            printf("[ERROR] Ref = %f, HW = %f\n", c, d);
            printf("[ERROR] Error in result at index [%d]. Error: %f ...\n", (i + 1), diff);
            error_count++;
        } else {
            //printf("[ INFO] Actual = %.3f, Ref = %.3f\n", d, c);
        }
    }

    if (error_count == 0) {
        printf("\n[ INFO] FFT PASSED!!\n\n");
    } else {
        printf("\n[ERROR] FFT FAILED!!\n\n");
    }

}

//###################################################################################
// Function to Print FFT Outputs
//###################################################################################
void print_matrix (TYPE *base) {

    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            printf("[%d][%d] = %f\n", i, j, base[i * DIM + j]);
        }
    }
}
