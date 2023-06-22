#include "inverse.h"

#define PROGPATH DASH_DATA "Dash-RadioCorpus/temporal_mitigation/"
#define ZIN PROGPATH "z.txt"
#define ZIMAGIN PROGPATH "zimag.txt"
#define SIN PROGPATH "s.txt"
#define SIMAGIN PROGPATH "simag.txt"
#define FILEREAD 0


#include "DashExtras.h"

class perf_counter {
   public:
	uint64_t tot, cnt, calls;
	perf_counter() : tot(0), cnt(0), calls(0){};
	inline void reset() { tot = cnt = calls = 0; }
	inline void start() {
		cnt = 0;
		calls++;
	};
	inline void stop() { tot += (0 - cnt); };
	inline uint64_t avg_cpu_cycles() { return ((tot + (calls >> 1)) / calls); };
};

void temporalmitigation(float CommsReal[N*M],float CommsImag[N*M],float RadarReal[N*M],float RadarImag[N*M],float OutputReal[N*M],float OutputImag[N*M]) {
	// Initializing the main variables that will be required in the overall computation of the Z response

	// Initializing the Z signal which will have 4*64 dimension
	float *Z, *Zi;
	Z = (float *)malloc(N * M * sizeof(float));
	Zi = (float *)malloc(N * M * sizeof(float));

	// Now defining the jammer signal which will have the same dimensions as the message signal , The jammer is denoted
	// by S
	float *S, *Si;
	S = (float *)malloc(N * M * sizeof(float));
	Si = (float *)malloc(N * M * sizeof(float));


// The following section has an option where you can read the input from a text file or you could just do a function call


	#ifdef FILEREAD
	// now defining the argument files which will contain the corresponding values of Z and S
	FILE *Zreal, *Zimag, *Sreal, *Simag;
	Zreal = fopen(ZIN, "r");
	Zimag = fopen(ZIMAGIN, "r");
	Sreal = fopen(SIN, "r");
	Simag = fopen(SIMAGIN, "r");

	// now copying the contents of the files into the arrays that have been assigned for the signal and the jammer

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			fscanf(Zreal, "%f", &Z[i * M + j]);
			fscanf(Zimag, "%f", &Zi[i * M + j]);
			fscanf(Sreal, "%f", &S[i * M + j]);
			fscanf(Simag, "%f", &Si[i * M + j]);
		}
	}

	//cout << "Done reading the files from the input provided \n";

	#endif

// the following section reads the input as passed from the arguments 
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			Z[i * M + j]=CommsReal[i*M+j];
			Zi[i * M + j]=CommsImag[i*M+j];
			S[i * M + j]=RadarReal[i*M+j];
			Si[i * M + j]=RadarImag[i*M+j];
		}
	}




	// The following lines contain the counter cycles variable :
	uint64_t hw_cycles;
	//// Computing the hermitian of S
	float *Shermitian, *Shermitianimag;
	Shermitian = (float *)malloc(M * N * sizeof(float));
	Shermitianimag = (float *)malloc(M * N * sizeof(float));

	perf_counter ctr1, ctr2, ctr3, ctr4, ctr5, ctr6, ctr7, ctr8;

	//printf("Calling a function to compute the hermitian \n");
	ctr1.start();
	KERN_ENTER(make_label("transpose[Ar-%d][Ac-%d][complex][float32]", N, M));
	hermitian(S, Si, Shermitian, Shermitianimag);
	KERN_EXIT(make_label("transpose[Ar-%d][Ac-%d][complex][float32]", N, M));
	ctr1.stop();
	hw_cycles = ctr1.avg_cpu_cycles();
	//printf("Time taken for the hermitian operator is %l \n", hw_cycles);
	//printf("Printing out the hermitian real part \n");
	display(Shermitian, 64, 4);

	//printf("Printing out the hermitian imag part \n");
	display(Shermitianimag, 64, 4);

	//// Now computing the result from the first multiplication (Z*S^H)--> Result 1
	float *result1, *result1imag;
	result1 = (float *)malloc(N * N * sizeof(float));
	result1imag = (float *)malloc(N * N * sizeof(float));

	//printf("Now computing the matrix multiplication of Z*S^H ---> C3 \n ");
	ctr2.start();
	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][float32][complex]", N, M, N));
	mmult(Z, Zi, Shermitian, Shermitianimag, result1, result1imag);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][float32][complex]", N, M, N));
	ctr2.stop();
	hw_cycles = ctr2.avg_cpu_cycles();
	//printf("Time taken for the complex multiplication is %l \n", hw_cycles);

	//printf("Imag Part \n");
	display(result1imag, 4, 4);

	// Now computing the second matrix multiplication (S*S^H) ---> Result2
	float *result2, *result2imag;
	result2 = (float *)malloc(N * N * sizeof(float));
	result2imag = (float *)malloc(N * N * sizeof(float));

	//printf("Now computing the result of S*S^H ---> C1 \n");
	ctr3.start();
	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][float32][complex]", N, M, N));
	mmult(S, Si, Shermitian, Shermitianimag, result2, result2imag);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][float32][complex]", N, M, N));
	ctr3.stop();
	hw_cycles = ctr3.avg_cpu_cycles();

	//cout << " ************************** Line 119 check  S*S^H *************** \n";

	//printf("Real Part \n");
	display(result2, 4, 4);

	//printf("Imag part \n");
	display(result2imag, 4, 4);

	//"\n the result is fine uptil here \n";
	// Now computing the inverse of the above result which is (S*S^H)^-1 ---> D and Di

	float *inv1, *inv2, *intmedt1, *intmedt2, *intmedb1, *intmedb2, *intmedb3, *intmedb4;
	float *buffer1, *buffer2, *buffer3, *buffer4;
	float *bufferinv1, *bufferinv2, *bufferinv3, *bufferinv4, *bufferinv5;
	// To store inverse of A[][]
	inv1 = (float *)malloc(N * N * sizeof(float));
	inv2 = (float *)malloc(N * N * sizeof(float));
	intmedt1 = (float *)malloc(N * N * sizeof(float));
	intmedt2 = (float *)malloc(N * N * sizeof(float));
	intmedb1 = (float *)malloc(N * N * sizeof(float));
	intmedb2 = (float *)malloc(N * N * sizeof(float));
	intmedb3 = (float *)malloc(N * N * sizeof(float));
	intmedb4 = (float *)malloc(N * N * sizeof(float));
	buffer1 = (float *)malloc(N * N * sizeof(float));
	buffer2 = (float *)malloc(N * N * sizeof(float));
	buffer3 = (float *)malloc(N * N * sizeof(float));
	buffer4 = (float *)malloc(N * N * sizeof(float));
	// The following arrays are used for the inverse computation

	bufferinv1 = (float *)malloc(N * N * sizeof(float));
	bufferinv2 = (float *)malloc(N * N * sizeof(float));
	bufferinv3 = (float *)malloc(N * N * sizeof(float));
	bufferinv4 = (float *)malloc(N * N * sizeof(float));
	bufferinv5 = (float *)malloc(N * N * sizeof(float));
	//cout << "Now computing the inverse of the above matrix multiplication ---> C2=(C1)^-1 \n";

	// inverse(result2,inv1);
	// Computing the inverse of a == Real part
	ctr4.start();
	KERN_ENTER(make_label("matrixInverse[Ar-%d][Ac-%d][float32][scalar]", N, N));
	alternateinverse(result2, inv1);
	KERN_EXIT(make_label("matrixInverse[Ar-%d][Ac-%d][float32][scalar]", N, N));
	ctr4.stop();
	hw_cycles = ctr4.avg_cpu_cycles();
	//cout << "Time taken for the inverse is" << hw_cycles << "\n";
	display(inv1, 4, 4);
	//cout << "\n";
	// compute res1 =inv(inv1)*result2imag == > bufferinv1
	ctr5.start();
	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));
	mmultiply(inv1, result2imag, bufferinv1);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));
	ctr5.stop();
	hw_cycles = ctr5.avg_cpu_cycles();
	//cout << "Time taken for the a real 4*4 matrix multiplication" << hw_cycles << "\n";
	// compute res2 = inv(result2imag*res1+result2)

	// result2imag*bufferinv1 == > bufferinv2
	ctr5.start();
	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));
	mmultiply(result2imag, bufferinv1, bufferinv2);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));
	ctr5.stop();
	hw_cycles = ctr5.avg_cpu_cycles();
	//cout << "Time taken for the real 4*4 matrix multiplication is" << hw_cycles << "\n";
	// bufferinv2+result2 ==> bufferinv3
	ctr6.start();
    KERN_ENTER(make_label("ZIP[add][%d,%d][complex][float32]"));
	mmadd(bufferinv2, result2, bufferinv3);
    KERN_EXIT(make_label("ZIP[add][%d,%d][complex][float32]",N,N));
	ctr6.stop();
	hw_cycles = ctr6.avg_cpu_cycles();
	//cout << "Time taken for the Matrix addition 4*4 is" << hw_cycles << "\n";
	// Now computing inv(bufferinv3) ==> bufferinv4
	// The following 'bufferinv4' is real part of the inverse
	//cout << "\n The imaginary part :::: \n";
	ctr6.start();
	KERN_ENTER(make_label("matrixInverse[Ar-%d][Ac-%d][float32][scalar]", N, N));
	alternateinverse(bufferinv3, bufferinv4);
	KERN_EXIT(make_label("matrixInverse[Ar-%d][Ac-%d][float32][scalar]", N, N));
	ctr6.stop();
	hw_cycles = ctr6.avg_cpu_cycles();
	//cout << "The time taken for the inverse is " << hw_cycles << "\n";
	display(bufferinv4, 4, 4);

	// Now computing the imaginary part of the inverse
	ctr6.start();
	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));
	mmultiply(bufferinv1, bufferinv4, bufferinv5);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));
	ctr6.stop();
	hw_cycles = ctr6.avg_cpu_cycles();
	//cout << "The time taken for a 4*4 matrix multiplication is" << hw_cycles << "\n";
	for (int k = 0; k < 4; k++) {
		for (int m = 0; m < 4; m++) {
			bufferinv5[k * 4 + m] *= -1;
		}
	}
	// Final result is -bufferinv5 == > imag part of inverse
	//

	//cout << "*****Check for line 162 ********\n ";

	//cout << "Printing the inverse of (S*S^H) C2 --> \n ";

	// Real part of the result involving the inverse
	//cout << "\n Fine uptill here too \n";
	// Currently only having trouble with imaginary part of inverse
	//	imagpart(result2,result2imag,inv1,inv2,intmedt2);
	//cout << "\n";
	//	display(intmedt2,4,4);
	// Now computing the result of (Z*S^H)*(S.S^H)^-1  ---> result3 which is a 4*4 and 4*4 multiplication

	float *result3, *result3imag;
	result3 = (float *)malloc(N * N * sizeof(float));
	result3imag = (float *)malloc(N * N * sizeof(float));

	//cout << "Now computing the result (Z*S^H)*(S*S^H)^-1 \n ";
	ctr6.start();
	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));
	mmult4(result1, result1imag, bufferinv4, bufferinv5, result3, result3imag);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, N));
	ctr6.stop();
	hw_cycles = ctr6.avg_cpu_cycles();
	//cout << "The time taken for complex 4*4 matrix multiplication is" << hw_cycles << "\n";
	//cout << "\n****** Result from line 191 **************\n";

	//cout << "Result from C3*C2 \n";
	//cout << "Real Part \n";
	display(result3, 4, 4);

	//cout << "Imag Part \n";
	display(result3imag, 4, 4);

	// Now computing the final matrix multiplication which is result3*S ---> result4 this is 4*4 and 4*64
	// multiplication

	float *result4, *result4imag;
	result4 = (float *)malloc(N * M * sizeof(float));
	result4imag = (float *)malloc(N * M * sizeof(float));

	//cout << "Final multiplication is being computed \n";
	ctr7.start();
	KERN_ENTER(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, M));
	mmult64(result3, result3imag, S, Si, result4, result4imag);
	KERN_EXIT(make_label("GEMM[Ar-%d][Ac-%d][Bc-%d][complex][float32]", N, N, M));
	ctr7.stop();
	hw_cycles = ctr7.avg_cpu_cycles();
	//cout << "Time taken for the 4*4 x 4*64 matrix multiplication is" << hw_cycles << "\n";

	//cout << "*******Final Multiplication result Line 221 *********** \n";
	//cout << "Real Part \n";
	display(result4, 4, 64);

	//cout << "Imag Part \n";
	display(result4imag, 4, 64);
	// Now we have to compute the final operation which is matrix subtraction : (Z - result4) ---> Zr  4*64 - 4*64
	float *zres, *zresimag;
	zres = (float *)malloc(N * M * sizeof(float));
	zresimag = (float *)malloc(N * M * sizeof(float));

	//cout << "Now computing the matrix subtraction from the above result to compute the Z response \n";
	ctr8.start();
	KERN_ENTER(make_label("ZIP[subtract][%d,%d][complex][float32]", N, M));
	msub(Z, Zi, result4, result4imag, zres, zresimag);
	KERN_EXIT(make_label("ZIP[subtract][%d,%d][complex][float32]", N, M));
	ctr8.stop();
	hw_cycles = ctr8.avg_cpu_cycles();
	//cout << "The time taken for the final matrix subtraction 4*64 is " << hw_cycles << "\n";
	//cout << "*****Final result being printed ******\n";
	// Printing the result out
	
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			OutputReal[i * M + j]=zres[i*M+j];
			OutputImag[i * M + j]=zresimag[i*M+j];
		}
	}





	//cout << "Real part: \n";
	display(zres, 4, 64);
	//cout << "Imag part: \n";
	display(zresimag, 4, 64);

	// Now freeing up the variables used
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
	free(bufferinv1);
	free(bufferinv2);
	free(bufferinv3);
	free(bufferinv4);
	free(bufferinv5);
	free(intmedt1);
	free(intmedt2);
	free(result3);
	free(result3imag);
	free(result4);
	free(result4imag);
	free(zres);
	free(zresimag);

	//return 0;
}
