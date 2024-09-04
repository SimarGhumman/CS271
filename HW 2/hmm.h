#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// pi[N], A[N][N] and B[N][M]
//#define N 2
#define N 26
//#define M 27
#define M 53

// character set
// Note: Size of character set must be M
#define ALPHABET {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53}

//#define ALPHABET {"abcdefghijklmnopqrstuvwxyz"}

// maximum characters per line
#define MAX_CHARS 500

// other
#define EPSILON 0.00001
#define DABS(x) ((x) < (0.0) ? -(x) : (x))

// debugging and/or printing
//#define PRINT_OBS
//#define PRINT_GET_T
//#define CHECK_GAMMAS
#define PRINT_REESTIMATES

struct stepStruct
{
    int obs;
    double c;
    double alpha[N];
    double beta[N];
    double gamma[N];
    double diGamma[N][N];
};

void alphaPass(struct stepStruct *step,
               double pi[], 
               double A[][N],
               double B[][M],
               int T);

void betaPass(struct stepStruct *step,
              double pi[], 
              double A[][N],
              double B[][M],
              int T);

void computeGammas(struct stepStruct *step,
                   double pi[], 
                   double A[][N],
                   double B[][M],
                   int T);
                   
void reestimatePi(struct stepStruct *step, 
                  double piBar[]);
                  
void reestimateA(struct stepStruct *step, 
                 double Abar[][N], 
                 int T);

void reestimateB(struct stepStruct *step, 
                 double Bbar[][M], 
                 int T);

void initMatrices(double pi[], 
                  double A[][N], 
                  double B[][M],
                  int seed);

int GetT();

int GetObservations(struct stepStruct *step,
                    int T);
                                  
void printPi(double pi[]);

void printA(double A[][N]);

void printBT(double B[][M]);
