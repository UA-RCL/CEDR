//
//  main.c
//  Rayleigh_distribution
//
//  Created by Shunyao Wu on 9/13/15.
//  Copyright (c) 2015 Shunyao Wu. All rights reserved.
//

#include "rayleigh_ch.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#define MAX_N 300000 /*这个值为N可以定义的最大长度*/
#define N 100000     /*产生随机序列的点数，注意不要大于MAX_N*/
#define leng 6
#define step 10
#define den step *leng + 1

void randr(double *x, int num, double sigma) {
	double x1[MAX_N];
	int i;
	time_t timep;
	struct tm *stime;
	unsigned seed;
	stime = gmtime(&timep);
	seed = stime->tm_sec * stime->tm_min * stime->tm_hour;
	srand(seed);
	for (i = 0; i < num; i++) {
		x1[i] = rand();
		x[i] = x1[i] / RAND_MAX;
		x[i] = sigma * sqrt(-2 * log(x[i]));
	}
}
