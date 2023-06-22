#include "inverse.h"

#define PROGPATH DASH_DATA "Dash-RadioCorpus/temporal_mitigation/"
#define ZIN PROGPATH "z.txt"
#define ZIMAGIN PROGPATH "zimag.txt"
#define SIN PROGPATH "s.txt"
#define SIMAGIN PROGPATH "simag.txt"

class perf_counter
{
    public:
        uint64_t tot, cnt, calls;
        perf_counter() : tot(0), cnt(0), calls(0) {};
        inline void reset() { tot = cnt = calls = 0; }
        inline void start() { cnt = 0; calls++; };
        inline void stop() { tot +=  (0 - cnt); };
        inline uint64_t avg_cpu_cycles() { return ((tot+(calls>>1)) / calls); }
};

int main(int argc , char* argv[])
{
    ////Initializing the main variables that will be required in the overall computation of the Z response
    perf_counter hw_ctr,hw_ctr1,hw_ctr2,hw_ctr3,hw_ctr4,hw_ctr5,hw_ctr6,hw_ctr7,hw_ctr8,hw_ctr9,hw_ctr10,hw_ctr11;
    /// Initializing the Z signal which will have 4*64 dimension
    float *Z,*Zi;
    Z = (float *)malloc(N*M*sizeof(float));
    Zi = (float *)malloc(N*M*sizeof(float));

    /// Now defining the jammer signal which will have the same dimensions as the message signal , The jammer is denoted by S
    float *S,*Si;
    S = (float *)malloc(N*M*sizeof(float));
    Si = (float *)malloc(N*M*sizeof(float));

    //\now defining the argument files which will contain the corresponding values of Z and S
    FILE *Zreal ,*Zimag , *Sreal,*Simag;
    Zreal=fopen(ZIN,"r");
    Zimag=fopen(ZIMAGIN,"r");
    Sreal=fopen(SIN,"r");
    Simag=fopen(SIMAGIN,"r");


    //\now copying the contents of the files into the arrays that have been assigned for the signal and the jammer

    for (int i=0 ; i<N ; i++)
    {
        for (int j=0;j<M ; j++)
        {
            fscanf(Zreal,"%f",&Z[i*M +j]);
            fscanf(Zimag,"%f",&Zi[i*M +j]);
            fscanf(Sreal,"%f",&S[i*M +j]);
            fscanf(Simag,"%f",&Si[i*M +j]);
        }
    }

    cout<<"Done reading the files from the input provided \n";

    //// Computing the hermitian of S
    float *Shermitian,*Shermitianimag;
    Shermitian = (float *)malloc(M*N*sizeof(float));
    Shermitianimag = (float *)malloc(M*N*sizeof(float));




    cout<<"Calling a function to compute the hermitian \n";
    hw_ctr.start();
    hermitian(S,Si,Shermitian,Shermitianimag);
    hw_ctr.stop();
    uint64_t hw_cycles = hw_ctr.avg_cpu_cycles();
    std::cout << "Average number of CPU cycles running hermitian in hardware: "
        << hw_cycles << std::endl;

    //	float Stemp[64][4],Simagtemp[64][4];
    //
    //		for(int i=0;i<64;i++)
    //			for(int j=0;j<4;j++)
    //			{
    //				Stemp[i][j]=Shermitian[i*4+j];
    //				Simagtemp[i][j]=Shermitianimag[i*4+j];
    //			}

    cout<<"Printing out the hermitian real part \n";
    display(Shermitian,64,4);

    cout<<"Printing out the hermitian imag part \n";
    display(Shermitianimag,64,4);






    //// Now computing the result from the first multiplication (Z*S^H)--> Result 1
    float *result1,*result1imag;
    result1 = (float *)malloc(N*N*sizeof(float));
    result1imag = (float *)malloc(N*N*sizeof(float));

    cout<<"Now computing the matrix multiplication of Z*S^H ---> C3 \n ";
    hw_ctr1.start();
    mmult(Z,Zi,Shermitian,Shermitianimag,result1,result1imag);
    hw_ctr1.stop();
    hw_cycles = hw_ctr1.avg_cpu_cycles();
    std::cout << "Average number of CPU cycles running mmult in hardware: "
        << hw_cycles << std::endl;

    //	float result1temp[4][4],result1tempimag[4][4];
    //
    //
    //	for(int i=0;i<4;i++)
    //				for(int j=0;j<4;j++)
    //				{
    //					result1temp[i][j]=result1[i*4+j];
    //					result1tempimag[i][j]=result1imag[i*4+j];
    //				}
    //
    //	cout<<"Real Part \n ";
    //	display(result1,4,4);

    cout<<"Imag Part \n";
    display(result1imag,4,4);


    //// Now computing the second matrix multiplication (S*S^H) ---> Result2
    float *result2,*result2imag;
    result2 = (float *)malloc(N*N*sizeof(float));
    result2imag = (float *)malloc(N*N*sizeof(float));


    //	cout<<"Now computing the result of S*S^H ---> C1 \n";
    hw_ctr2.start();
    mmult(S,Si,Shermitian,Shermitianimag,result2,result2imag);
    hw_ctr2.stop();
    hw_cycles = hw_ctr2.avg_cpu_cycles();
    std::cout << "Average number of CPU cycles running mmult in hardware: "
        << hw_cycles << std::endl;


    //	float result2temp[4][4],result2tempimag[4][4];
    //
    //	for(int i=0;i<4;i++)
    //				for(int j=0;j<4;j++)
    //				{
    //					result2temp[i][j]=result2[i*4+j];
    //					result2tempimag[i][j]=result2imag[i*4+j];
    //				}


    cout<<" ************************** Line 119 check  S*S^H *************** \n";


    cout<<"Real Part \n";
    display(result2,4,4);

    cout<<"Imag part \n";
    display(result2imag,4,4);
    //




    cout<<"\n the result is fine uptil here \n";
    //// Now computing the inverse of the above result which is (S*S^H)^-1 ---> D and Di

    float *inv1,*inv2;
    //	*intmedt1,*intmedt2; // To store inverse of A[][]
    inv1=(float *)malloc(N*N*sizeof(float));
    inv2=(float *)malloc(N*N*sizeof(float));
    //	intmedt1=(float *)malloc(N*N*sizeof(float));
    //	intmedt2=(float *)malloc(N*N*sizeof(float));

    cout<<"Now computing the inverse of the above matrix multiplication ---> C2=(C1)^-1 \n";

    //	inverse(result2,inv1);
    hw_ctr3.start();
    alternateinverse(result2,inv1);
    hw_ctr3.stop();
    hw_cycles = hw_ctr3.avg_cpu_cycles();
    std::cout << "Average number of CPU cycles running inverse in hardware: "
        << hw_cycles << std::endl;



    display(inv1,4,4);
    cout<<"\n";

    hw_ctr4.start();
    alternateinverse(result2imag,inv2);
    hw_ctr4.stop();
    hw_cycles = hw_ctr4.avg_cpu_cycles();
    std::cout << "Average number of CPU cycles running inverse in hardware: "
        << hw_cycles << std::endl;

    //	display(inv2,4,4);


    cout<<"*****Check for line 162 ********\n ";


    cout<<"Printing the inverse of (S*S^H) C2 --> \n ";

    float *resultreal,*resultimag;
    float *resultrealinv,*resultimaginv;
    resultreal=(float *)malloc(N*N*sizeof(float));
    resultimag=(float *)malloc(N*N*sizeof(float));
    resultrealinv=(float *)malloc(N*N*sizeof(float));
    resultimaginv=(float *)malloc(N*N*sizeof(float));

    hw_ctr8.start();
    realpart(result2,result2imag,inv1,inv2,resultreal);
    alternateinverse(resultreal,resultrealinv);
    hw_ctr8.stop();

    hw_cycles = hw_ctr8.avg_cpu_cycles();
    std::cout << "Average number of CPU cycles running real part of inverse in hardware: "
        << hw_cycles << std::endl;


    display(resultrealinv,4,4);
    cout<<"\n";



    cout<<"\n Fine uptill here too \n";

    // Currently only having trouble with imaginary part of inverse

    hw_ctr9.start();
    imagpart(result2,result2imag,inv1,inv2,resultimag);
    alternateinverse(resultimag,resultimaginv);
    hw_ctr9.stop();
    hw_cycles = hw_ctr9.avg_cpu_cycles();
    std::cout << "Average number of CPU cycles running imag part of inverse in hardware: "
        << hw_cycles << std::endl;


    cout<<"\n";
    display(resultimaginv,4,4);

    /** No need for this now that intmedt1 and intmedt2 is dynamically allocated
    //	float *D,*Di;
    //	D= (float *)malloc(N * N * sizeof(float));
    //	Di= (float *)malloc(N * N * sizeof(float));
    //
    //	cout<<"Now copying the above data in a dynamic array D and Di \n";
    //
    //	for (int i=0 ; i<N ; i++)
    //	        {
    //	        	for (int j=0;j<N ; j++)
    //	        	{ D[i*N + j]=intmedt1[i][j];
    //	        	  Di[i*N + j]=intmedt2[i][j];
    //
    //	        	}
    //
    //	        }
    */

    /// Now computing the result of (Z*S^H)*(S.S^H)^-1  ---> result3 which is a 4*4 and 4*4 multiplication

    float *result3,*result3imag;
    result3 = (float *)malloc(N*N*sizeof(float));
    result3imag = (float *)malloc(N*N*sizeof(float));


    cout<<"Now computing the result (Z*S^H)*(S*S^H)^-1 \n ";
    hw_ctr5.start();
    mmult4(result1,result1imag,resultrealinv,resultimaginv,result3,result3imag);
    hw_ctr5.stop();
    hw_cycles = hw_ctr5.avg_cpu_cycles();
    std::cout << "Average number of CPU cycles running hermitian in hardware: "
        << hw_cycles << std::endl;

    //	float result3temp[4][4],result3imagtemp[4][4];
    //
    //	for (int i=0 ; i<N ; i++)
    //		        {
    //		        	for (int j=0;j<N ; j++)
    //		        	{ result3temp[i][j]=result3[i*N + j];
    //		        	  result3imagtemp[i][j]=result3imag[i*N + j];
    //
    //		        	}
    //
    //		        }

    cout<<"\n****** Result from line 191 **************\n";

    cout<<"Result from C3*C2 \n";
    cout<<"Real Part \n";
    display(result3,4,4);

    cout<<"Imag Part \n";
    display(result3imag,4,4);


    /// Now computing the final matrix multiplication which is result3*S ---> result4 this is 4*4 and 4*64 multiplication

    float *result4,*result4imag;
    result4 = (float *)malloc(N*M*sizeof(float));
    result4imag = (float *)malloc(N*M*sizeof(float));



    cout<<"Final multiplication is being computed \n";
    hw_ctr6.start();
    mmult64(result3,result3imag,S,Si,result4,result4imag);
    hw_ctr6.stop();
    hw_cycles = hw_ctr6.avg_cpu_cycles();
    std::cout << "Average number of CPU cycles running hermitian in hardware: "
        << hw_cycles << std::endl;


    //	float result4temp[4][64],result4tempimag[4][64];
    //
    //	for (int i=0 ; i<4 ; i++)
    //		        {
    //		        	for (int j=0;j<64 ; j++)
    //		        	{ result4temp[i][j]=result4[i*64+j];
    //		        	  result4tempimag[i][j]=result4imag[i*64+j];
    //
    //		        	}
    //
    //		        }
    //


    cout<<"*******Final Multiplication result Line 221 *********** \n";
    cout<<"Real Part \n";
    display(result4,4,64);

    cout<<"Imag Part \n";
    display(result4imag,4,64);
    /// Now we have to compute the final operation which is matrix subtraction : (Z - result4) ---> Zr  4*64 - 4*64
    float *zres , *zresimag;
    zres = (float *)malloc(N*M*sizeof(float));
    zresimag = (float *)malloc(N*M*sizeof(float));



    cout<<"Now computing the matrix subtraction from the above result to compute the Z response \n";
    hw_ctr7.start();
    msub(Z,Zi,result4,result4imag,zres,zresimag);
    hw_ctr7.stop();

    hw_cycles = hw_ctr7.avg_cpu_cycles();
    std::cout << "Average number of CPU cycles running hermitian in hardware: "
        << hw_cycles << std::endl;



    cout<<"*****Final result being printed ******\n";
    //// Printing the result out
    cout<<"Real part: \n";
    display(zres,4,64);
    cout<<"Imag part: \n";
    display(zresimag,4,64);

    //	cout<<"Printing the real part of the result first \n ";
    //	for(int i=0;i<N;i++)
    //	{for(int j=0;j<M;j++)
    //		{ cout<<zres[i*M + j]<<" ";
    //		}
    //		cout<<"\n";
    //
    //	}
    //
    //	cout<<"Printing the imaginary part of the result now \n ";
    //		for(int i=0;i<N;i++)
    //		{for(int j=0;j<M;j++)
    //			{ cout<<zresimag[i*M + j]<<" ";
    //			}
    //			cout<<"\n";
    //
    //		}
    // *** From previous inverse code
    /*
       float *A, *B,*D,*Di;
       A= (float *)malloc(N * N * sizeof(float));
       B= (float *)malloc(N * N * sizeof(float));
       D= (float *)malloc(N * N * sizeof(float));
       Di= (float *)malloc(N * N * sizeof(float));
    //    C_sw= (float *)malloc(N * N * sizeof(float));
    //    C_swi= (float *)malloc(N * N * sizeof(float));

    if (!A || !B || !D || !Di) {
    if (A) free(A);
    //if (Ai) free(Ai);
    if (B) free(B);
    //if (Bi) free(Bi);
    if (D) free(D);
    if (Di) free(Di);
    //              if (C_sw) free(C_sw);
    //              if (C_swi) free(C_swi);
    return 2;
    }
    FILE *data1 , *data2;

    data1=fopen(argv[1],"r");
    data2=fopen(argv[2],"r");

    for (int i=0 ; i<N ; i++)
    {
    for (int j=0;j<N ; j++)
    {
    fscanf(data1,"%f",&A[i*N +j]);
    fscanf(data2,"%f",&B[i*N +j]);
    }
    }

    //    display(A);
    //    display(B);

    float inv1[N][N],inv2[N][N],intmedt1[N][N],intmedt2[N][N]; // To store inverse of A[][]


    cout << "\nThe Inverse is :\n";
    inverse(A, inv1);
    display(inv1);
    inverse(B, inv2);
    display(inv2);
    realpart(A,B,inv1,inv2,intmedt1);
    display(intmedt1);
    imagpart(A,B,inv1,inv2,intmedt2);
    display(intmedt2);


    for (int i=0 ; i<N ; i++)
    {
    for (int j=0;j<N ; j++)
    { D[i*N + j]=intmedt1[i][j];
    Di[i*N + j]=intmedt2[i][j];

    }

    }

    free(A);
    free(B);
    free(D);
    free(Di);
    //    free(C_sw);
    //    free(C_swi);



*/

    ///****
    /// Now freeing up the variables used
    free(Z);
    free(Zi);
    free(S);
    free(Si);
    free(Shermitian);
    free(Shermitianimag);
    free(result1);
    free(result1imag);
    free(result2);
    free(result2imag);
    free(inv1);
    free(inv2);
    free(resultreal);
    free(resultrealinv);
    free(resultimag);
    free(resultimaginv);
    free(result3);
    free(result3imag);
    free(result4);
    free(result4imag);
    free(zres);
    free(zresimag);





    return 0;

}
