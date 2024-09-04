#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// pi[N], A[N][N] and B[N][M]
#define N 2
//#define N 26
#define M 27
//#define M 26

// character set
// Note: Size of character set must be M
#define ALPHABET {"abcdefghijklmnopqrstuvwxyz "}
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
    float c;
    float alpha[N];
    float beta[N];
    float gamma[N];
    float diGamma[N][N];
};

void alphaPass(struct stepStruct *step,
               float pi[], 
               float A[][N],
               float B[][M],
               int T);

void betaPass(struct stepStruct *step,
              float pi[], 
              float A[][N],
              float B[][M],
              int T);

void computeGammas(struct stepStruct *step,
                   float pi[], 
                   float A[][N],
                   float B[][M],
                   int T);
                   
void reestimatePi(struct stepStruct *step, 
                  float piBar[]);
                  
void reestimateA(struct stepStruct *step, 
                 float Abar[][N], 
                 int T);

void reestimateB(struct stepStruct *step, 
                 float Bbar[][M], 
                 int T);

void reestimateWV(struct stepStruct *step, float A[N][N], float B[N][M], float W[N][N], float V[N][M], float logProb, int T);

void initMatrices(float pi[], 
                  float A[][N], 
                  float B[][M],
                  int seed);

int GetT(char fname[],
         int startPos,
         int startChar,
         int maxChars);

int GetObservations(char fname[], 
                    struct stepStruct *step,
                    int T,
                    int startPos,
                    int startChar,
                    int maxChars);
                                  
void printPi(float pi[]);

void printA(float A[][N]);

void printBT(float B[][M]);