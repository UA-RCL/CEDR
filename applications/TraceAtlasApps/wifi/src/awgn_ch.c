#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include "awgn_ch.h"

double Gaussion(double miu,double sigma)
{
    double U1, U2, W, mult;
//    static double X1, X2;
//    static int call = 0;
    double X1, X2;
    int call = 0;
    if (call == 1)
    {
        call = !call;
        return (miu + sigma * (double) X2);
    }

    do
    {
        U1 = -1 + ((double) rand () / RAND_MAX) * 2;
        U2 = -1 + ((double) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);

    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (miu + sigma * (double) X1);
}
                                                                                                                                            
