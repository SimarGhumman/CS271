//
// Hidden Markov Model program for written English
//
// Consistent with "Revealing Introduction" version dated January 12, 2017
//
// The program reads in data (English text) ignoring
// all punctuation, leaving only 26 letters and
// word spaces. Consequently, the A matrix is N x N and 
// the B matrix is N x M, where M = 27. The program
// begins with random (approximately uniform) A, B and pi,
// then attempts to climb to the optimal A and B.
// This program is related to the work described in 
// Cave and Neuwirth's paper "Hidden Markov Models
// for English"
//
// Note: The data file must not have any formatting!
//
// To compile: gcc -o hmm hmm.c -O3
//

#include "hmm.h"

int Z408[24][17] = {
    { 1, 2, 3, 4, 5, 4, 6, 7, 2, 8, 9,10,11,12,13,11, 7},// 1-17
    {14,15,16,17,18,19,20,21, 1,22, 3,23,24,25,26,19,17},// 18-34
    {27,28,19,29, 6,30, 8,31,26,32,33,34,35,19,36,37,38},// 35-51
    {39,40, 4, 1, 2, 7, 3, 9,10,41, 6, 2,42,10,43,26,44},// 52-68
    { 8,29,45,27, 5,28,46,47,48,12,20,22,15,14,17,31,19},// 69-85
    {23,16,26,18,36, 1,24,30,38,21,26,13,49,37,50,39,40},// 86-102
    {10,34,33,30,19,44,43, 9, 1,26,18, 7,32,21,39, 2, 7},// 103-119
    {45,46, 4, 3, 2, 7,23,13,26,44,22,27, 6,29,10,10, 8},// 120-136
    {51, 5,24,26,12,30,38,14,26,25,49,37,45,27,47, 1,52},// 137-153
    { 7, 3,36,10,16,28,11,21,48,34,40,17,44, 6,22, 8,20},// 154-170
    { 5,51,12, 9,15,14,30,37,16,33,45,38,43,29,10,21,22},// 171-187
    {30, 1,36,10,53,32,19,47,48,46,17, 4,23,13,28,35,41},// 188-204
    { 3,37,27,49,10, 6,33, 2,45,38,34,15,44,24,22,11,18},// 205-221
    {47,30,25,28, 8,37, 1,49,45,27,43,34,41,38, 5,40, 3},// 222-238
    {50, 6,12, 8,41, 1,52, 7,15,14,48,16,15,32,33, 9, 3},// 239-255
    {29,11,39,47,43,42, 6,17,21,31,36,50,18, 2, 2,25,27},// 256-272
    {34, 8,38,39,51,44, 4, 1, 2, 2, 5,42,41, 3,52, 7,15},// 273-289
    {12,17,13,26,14,26,53,20,52,49,51,16,23, 1,41, 1, 7},// 290-306
    { 2, 9,32,37,10, 6,51,16,53,46,19,26,53,29,39,26,14},// 307-323
    {15, 5,17,18,19,24,44,53,32,19,41, 1, 2,52,45,33,53},// 324-340
    {22,25,20, 7,13, 1,50,13,41,36,46,48,31,45,25,11,26},// 341-357
    {53,17,46,52,52,21,17,37, 3, 9,10,13,35,20, 2,18,51},// 358-374
    { 5,23,28,32,33,26,53,49,28,30,16,47, 7, 3,35,14,21},// 375-391
    {15,44,13,47, 1,14,30,21,26,44,22,27,38,11,19,30, 8} // 392-408
};

typedef struct {
    int key;   // column
    int value; // row
} KeyValuePair;

KeyValuePair dict[M];

int findMaxRowInColumn(double matrix[N][M], int col) {
    int maxRow = 0;
    for (int i = 1; i < N; i++) {
        if (matrix[i][col] > matrix[maxRow][col]) {
            maxRow = i;
        }
    }
    return maxRow;
}

double calculateAccuracy(const char* expected, const char* outputText) {
    int correctCount = 0;
    for (int i = 0; i < strlen(expected); i++) {
        if (expected[i] == outputText[i]) {
            correctCount++;
        }
    }
    return ((double)correctCount / (double)strlen(expected)) * 100;
}

int main(int argc, const char *argv[])
{
    int startPos,
        startChar,
        maxChars,
        maxIters,
        i,
        j,
        T,
        iter;
        
    int seed;

    double logProb,
           newLogProb;

    double pi[N],
           piBar[N],
           A[N][N],
           Abar[N][N],
           B[N][M],
           Bbar[N][M];
           
    char fname[80];
    
    struct stepStruct *step;

    if(argc != 7)
    {
oops:   fprintf(stderr, "\nUsage: %s filename startPos startChar maxChars maxIters seed\n\n", argv[0]);
        fprintf(stderr, "where filename == input file\n");
        fprintf(stderr, "      startPos == starting position for each line (numbered from 0)\n");
        fprintf(stderr, "      startChar == starting character in file (numbered from 0)\n");
        fprintf(stderr, "      maxChars == max characters to read (<= 0 to read all)\n");
        fprintf(stderr, "      maxIters == max iterations of re-estimation algorithm\n");
        fprintf(stderr, "      seed == seed value for pseudo-random number generator (PRNG)\n\n");
        fprintf(stderr, "For example:\n\n      %s datafile 0 0 0 100 1241\n\n", argv[0]);
        fprintf(stderr, "will read all of `datafile' and perform a maximum of 100 iterations.\n\n");
        fprintf(stderr, "For the English text example, try:\n\n      %s BrownCorpus 15 1000 50000 200 22761\n\n", argv[0]);
        fprintf(stderr, "will read from `BrownCorpus' and seed the PRNG with 22761,\n");
        fprintf(stderr, "will not read characters 0 thru 14 of each new line in `BrownCorpus',\n");
        fprintf(stderr, "will not save the first 1000 characters read and\n");
        fprintf(stderr, "will save a maximum of 50k observations.\n\n");
        exit(0);
    }

//    sprintf(fname, argv[1]);
    strcpy(fname, argv[1]);
    startPos = atoi(argv[2]);
    startChar = atoi(argv[3]);
    maxChars = atoi(argv[4]);
    maxIters = atoi(argv[5]);
    seed = atoi(argv[6]);

    ////////////////////////
    // read the data file //
    ////////////////////////
    
    // determine number of observations
    printf("GetT... ");
    fflush(stdout);
    T = GetT();
    
    printf("T = %d\n", T);

    // allocate memory
    printf("allocating %lu bytes of memory... ", (T + 1) * sizeof(struct stepStruct));
    fflush(stdout);
    if((step = calloc(T + 1, sizeof(struct stepStruct))) == NULL)
    {
        fprintf(stderr, "\nUnable to allocate alpha\n\n");
        exit(0);
    }
    printf("done\n");

    // read in the observations from file
    printf("GetObservations... ");
    fflush(stdout);
    T = GetObservations(step, T);
    printf("T = %d\n", T);

    /////////////////////////
    // hidden markov model //
    /////////////////////////

    char bestDecodedText[4096]; // Make sure this is sufficiently large
    double bestAccuracy = 0;


    for (int restarts = 0; restarts < 100000; restarts++) {
        unsigned int seed = (unsigned int) time(NULL) + restarts;    
        srand(seed);
    
        
        // initialize pi[], A[][] and B[][]
        initMatrices(pi, A, B, seed);
        
        // print pi[], A[][] and B[][] transpose
        // printf("\nN = %d, M = %d, T = %d\n", N, M, T);
        // printf("initial pi =\n");
        // printPi(pi);
        // printf("initial A =\n");
        // printA(A);
        // printf("initial B^T =\n");
        // printBT(B);

        // initialization
        iter = 0;
        logProb = -1.0;
        newLogProb = 0.0;

        for(i = 0; i < N; ++i)
        {
            for(j = 0; j < N; ++j)
            {
                Abar[i][j] = A[i][j];
            }
        }
        printf("\nbegin restarts = %d\n", restarts);

        // main loop
        while((iter < maxIters) && (newLogProb > logProb))
        {
            // printf("\nbegin iteration = %d\n", iter);

            logProb = newLogProb;

            // alpha (or forward) pass
            // printf("alpha pass... ");
            // fflush(stdout);
            alphaPass(step, pi, A, B, T);
            // printf("done\n");
            
            // // beta (or backwards) pass
            // printf("beta pass... ");
            // fflush(stdout);
            betaPass(step, pi, A, B, T);
            // printf("done\n");
            
            // // compute gamma's and diGamma's
            // printf("compute gamma's and diGamma's... ");
            // fflush(stdout);
            computeGammas(step, pi, A, B, T);
            // printf("done\n");
            
            // // find piBar, reestimate of pi
            // printf("reestimate pi... ");
            // fflush(stdout);
            reestimatePi(step, piBar);
            // printf("done\n");
            
            // // // find Abar, reestimate of A
            // // printf("reestimate A... ");
            // // fflush(stdout);
            // // reestimateA(step, Abar, T);
            // // printf("done\n");
            
            // // find Bbar, reestimate of B
            // printf("reestimate B... ");
            // fflush(stdout);
            reestimateB(step, Bbar, T);
            // printf("done\n");
            
    #ifdef PRINT_REESTIMATES
            // printf("piBar =\n");
            // printPi(piBar);
            // printf("Abar =\n");
            // printA(Abar);
            // printf("Bbar^T = \n");
            // printBT(Bbar);
    #endif // PRINT_REESTIMATES

            // assign pi, A and B corresponding "bar" values
            for(i = 0; i < N; ++i)
            {
                pi[i] = piBar[i];
            
                for(j = 0; j < N; ++j)
                {
                    A[i][j] = Abar[i][j];
                }

                for(j = 0; j < M; ++j)
                {
                    B[i][j] = Bbar[i][j];
                }
                
            }// next i

            // compute log [P(observations | lambda)], where lambda = (A,B,pi)
            newLogProb = 0.0;
            for(i = 0; i < T; ++i)
            {
                newLogProb += log(step[i].c);
            }
            newLogProb = -newLogProb;

            // a little trick so that no initial logProb is required
            if(iter == 0)
            {
                logProb = newLogProb - 1.0;
            }

            // printf("completed iteration = %d, log [P(observation | lambda)] = %f\n", 
            //         iter, newLogProb);

            ++iter;

        }// end while
        
        // printf("\nT = %d, N = %d, M = %d, iterations = %d\n\n", T, N, M, iter);
        // printf("final pi =\n");
        // printPi(pi);
        // printf("\nfinal A =\n");
        // printA(A);
        // printf("\nfinal B^T =\n");
        // printBT(B);
        // printf("\nlog [P(observations | lambda)] = %f\n\n", newLogProb);

        int dictSize = 0;

        // Find the max row for each column and add to the dictionary.
        for (int j = 0; j < M; j++) {
            int row = findMaxRowInColumn(B, j);
            dict[dictSize].key = j + 1;
            dict[dictSize].value = row;
            dictSize++;
        }

        // Now, let's use the dictionary to decode the Z408 matrix.
        char outputText[24 * 17 + 1] = {0};
        int outputIndex = 0;

        for (int i = 0; i < 24; i++) {
            for (int j = 0; j < 17; j++) {
                int targetKey = Z408[i][j];

                // Search the dictionary for the key.
                for (int k = 0; k < dictSize; k++) {
                    if (dict[k].key == targetKey) {
                        // Convert the row index to a character.
                        char mappedChar;
                        if (dict[k].value >= 0 && dict[k].value < 26) { // 'a' to 'z'
                            mappedChar = 'a' + dict[k].value;
                        } else {
                            mappedChar = '?';
                        }
                        outputText[outputIndex++] = mappedChar;
                        break;
                    }
                }
            }
        }

        // printf("Decoded Text: %s\n", outputText);
        const char* expected = "ilikekillingpeoplebecauseitissomuchfunitismorefunthankillingwildgameintheforrestbecausemanisthemostdangeroueanamalofalltokillsomethinggivesmethemostthrillingexperenceitisevenbetterthangettingyourrocksoffwithagirlthebestpartofitisthaewhenidieiwillbereborninparadiceandalltheihavekilledwillbecomemyslavesiwillnotgiveyoumynamebecauseyouwilltrytosloidownoratopmycollectiogofslavesformyafterlifeebeorietemethhpiti";

        double currentAccuracy = calculateAccuracy(expected, outputText);
        if (currentAccuracy > bestAccuracy) {
            bestAccuracy = currentAccuracy;
            strcpy(bestDecodedText, outputText); // Save this as the best decoded text so far
        }
    }

    printf("Best Decoded Text: %s\n", bestDecodedText);
    printf("Best Accuracy: %.2lf%%\n", bestAccuracy);

    return 0;
}

//
// alpha pass (or forward pass) including scaling
//
void alphaPass(struct stepStruct *step,
               double pi[], 
               double A[][N],
               double B[][M],
               int T)
{
    int i,
        j,
        t;
        
    double ftemp;
    
    // compute alpha[0]'s
    ftemp = 0.0;
    for(i = 0; i < N; ++i)
    {
        step[0].alpha[i] = pi[i] * B[i][step[0].obs];
        ftemp += step[0].alpha[i];
    }
    step[0].c = 1.0 / ftemp;

    // scale alpha[0]'s
    for(i = 0; i < N; ++i)
    {
        step[0].alpha[i] /= ftemp;
    }

    // alpha pass
    for(t = 1; t < T; ++t)
    {
        ftemp = 0.0;
        for(i = 0; i < N; ++i)
        {
            step[t].alpha[i] = 0.0;
            for(j = 0; j < N; ++j)
            {
                step[t].alpha[i] += step[t - 1].alpha[j] * A[j][i];
            }
            step[t].alpha[i] *= B[i][step[t].obs];
            ftemp += step[t].alpha[i];
        }
        step[t].c = 1.0 / ftemp;
        
        // scale alpha's
        for(i = 0; i < N; ++i)
        {
            step[t].alpha[i] /= ftemp;
        }
    
    }// next t
    
}// end alphaPass


//
// beta pass (or backwards pass) including scaling
//
void betaPass(struct stepStruct *step,
              double pi[], 
              double A[][N],
              double B[][M],
              int T)
{
    int i,
        j,
        t;

    // compute scaled beta[T - 1]'s
    for(i = 0; i < N; ++i)
    {
        step[T - 1].beta[i] = 1.0 * step[T - 1].c;
    }

    // beta pass
    for(t = T - 2; t >= 0; --t)
    {
        for(i = 0; i < N; ++i)
        {
            step[t].beta[i] = 0.0;
            for(j = 0; j < N; ++j)
            {
                step[t].beta[i] += A[i][j] * B[j][step[t + 1].obs] * step[t + 1].beta[j];
            }
            
            // scale beta's (same scale factor as alpha's)
            step[t].beta[i] *= step[t].c;
        }

    }// next t
        
}// end betaPass


//
// compute gamma's and diGamma's including optional error checking
//
void computeGammas(struct stepStruct *step,
                   double pi[], 
                   double A[][N],
                   double B[][M],
                   int T)
{
    int i,
        j,
        t;
        
    double denom;

#ifdef CHECK_GAMMAS
    double ftemp,
           ftemp2;
#endif // CHECK_GAMMAS

    // compute gamma's and diGamma's
    for(t = 0; t < T - 1; ++t)// t = 0,1,2,...,T-2
    {
        
#ifdef CHECK_GAMMAS
        ftemp2 = 0.0;
#endif // CHECK_GAMMAS

        for(i = 0; i < N; ++i)
        {
            step[t].gamma[i] = 0.0;
            for(j = 0; j < N; ++j)
            {
                step[t].diGamma[i][j] = (step[t].alpha[i] * A[i][j] * B[j][step[t + 1].obs] * step[t + 1].beta[j]);
                step[t].gamma[i] += step[t].diGamma[i][j];
            }

#ifdef CHECK_GAMMAS
            // verify that gamma[i] == alpha[i]*beta[i] / sum(alpha[j]*beta[j])
            ftemp2 += step[t].gamma[i];
            ftemp = 0.0;
            for(j = 0; j < N; ++j)
            {
                ftemp += step[t].alpha[j] * step[t].beta[j];
            }
            ftemp = (step[t].alpha[i] * step[t].beta[i]) / ftemp;
            if(DABS(ftemp - step[t].gamma[i]) > EPSILON)
            {
                printf("gamma[%d] = %f (%f) ", i, step[t].gamma[i], ftemp);
                printf("********** Error !!!\n");
            }
#endif // CHECK_GAMMAS

        }// next i
            
#ifdef CHECK_GAMMAS
        if(DABS(1.0 - ftemp2) > EPSILON)
        {
            printf("sum of gamma's = %f (should sum to 1.0)\n", ftemp2);
        }
#endif // CHECK_GAMMAS
            
    }// next t
    
    // special case for t = T-1
    for(j = 0; j < N; ++j)
    {
        step[T-1].gamma[j] = step[T-1].alpha[j];
    }

}// end computeGammas


//
// reestimate pi, the initial distribution
//
void reestimatePi(struct stepStruct *step, 
                  double piBar[])
{
    int i;
    
    // reestimate pi[]        
    for(i = 0; i < N; ++i)
    {
        piBar[i] = step[0].gamma[i];
    }
        
}// end reestimatePi


//
// reestimate the A matrix
//
void reestimateA(struct stepStruct *step, 
                 double Abar[][N], 
                 int T)
{
    int i,
        j,
        t;
    
    double numer,
           denom;
           
    // reestimate A[][]
    for(i = 0; i < N; ++i)
    {
        denom = 0.0;
        // t = 0,1,2,...,T-2
        for(t = 0; t < T - 1; ++t)
        {
            denom += step[t].gamma[i];
            
        }// next t

        for(j = 0; j < N; ++j)
        {
            numer = 0.0;

            // t = 0,1,2,...,T-2
            for(t = 0; t < T - 1; ++t)
            {
                numer += step[t].diGamma[i][j];

            }// next t

            Abar[i][j] = numer / denom;
        
        }// next j
        
    }// next i
        
} // end reestimateA    


//
// reestimate the B matrix
//
void reestimateB(struct stepStruct *step, 
                 double Bbar[][M], 
                 int T)
{
    int i,
        j,
        t;
    
    double numer,
           denom;
           
    // reestimate B[][]
    for(i = 0; i < N; ++i)
    {
        denom = 0.0;
        // t = 0,1,2,...,T-1
        for(t = 0; t < T; ++t)
        {
            denom += step[t].gamma[i];
            
        }// next t

        for(j = 0; j < M; ++j)
        {
            numer = 0.0;

            // t = 0,1,2,...,T-1
            for(t = 0; t < T; ++t)
            {
                if(step[t].obs == j)
                {
                    numer += step[t].gamma[i];
                }
                
            }// next t

            Bbar[i][j] = numer / denom;
        
        }// next j
        
    }// next i
        
}// end reestimateB


//
// initialize pi[], A[][] and B[][]
//
void initMatrices(double pi[], 
                  double A[][N], 
                  double B[][M],
                  int seed)
{
    int i,
        j;
        
    double prob,
           ftemp,
           ftemp2;
    
    // initialize pseudo-random number generator
    srandom(seed);

    // initialize pi
    prob = 1.0 / (double)N;
    ftemp = prob / 10.0;
    ftemp2 = 0.0;
    for(i = 0; i < N; ++i)
    {
        if((random() & 0x1) == 0)
        {
            pi[i] = prob + (double)(random() & 0x7) / 8.0 * ftemp;
        }
        else
        {
            pi[i] = prob - (double)(random() & 0x7) / 8.0 * ftemp;
        }
        ftemp2 += pi[i];
        
    }// next i
    
    for(i = 0; i < N; ++i)
    {
        pi[i] /= ftemp2;
    }

    double tempA[N][N] = {
        {0.00424, 0.00000, 0.01090, 0.00046, 0.00280, 0.01590, 0.02620, 0.08627, 0.00555, 0.25108, 0.00247, 0.00000, 0.00256, 0.00453, 0.12329, 0.26975, 0.00002, 0.00026, 0.00247, 0.00013, 0.00004, 0.06350,  0.04831, 0.00003, 0.06478, 0.01444},
        {0.00135, 0.02745, 0.41743, 0.00000, 0.00003, 0.00924, 0.00204, 0.01479, 0.00000, 0.04454, 0.00021, 0.00292, 0.01522, 0.00008, 0.02863, 0.04859, 0.00091, 0.00420, 0.00223, 0.00091, 0.00308, 0.04255,  0.00700, 0.26010, 0.06435, 0.00212},
        {0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.09636, 0.12981, 0.00004, 0.00237, 0.03511, 0.09620, 0.00000, 0.04265, 0.00313, 0.03435, 0.03174, 0.10603, 0.00000, 0.14682, 0.00028, 0.00000, 0.02346,  0.04042, 0.00000, 0.20624, 0.00497},
        {0.00000, 0.00000, 0.01244, 0.00000, 0.14167, 0.00793, 0.01212, 0.03086, 0.01332, 0.12434, 0.00510, 0.06500, 0.02703, 0.00136, 0.01656, 0.07225, 0.01493, 0.00006, 0.00582, 0.08256, 0.12431, 0.02231,  0.05164, 0.10772, 0.05761, 0.00307},
        {0.00000, 0.00000, 0.00275, 0.00000, 0.00000, 0.00275, 0.22994, 0.02958, 0.06038, 0.03704, 0.03636, 0.00000, 0.00598, 0.00000, 0.05762, 0.13421, 0.02915, 0.00000, 0.00268, 0.00000, 0.00027, 0.26124,  0.00171, 0.00019, 0.10814, 0.00001},
        {0.02120, 0.00004, 0.01484, 0.01627, 0.00005, 0.00225, 0.12382, 0.03858, 0.00330, 0.11328, 0.01280, 0.00073, 0.02162, 0.00109, 0.01072, 0.06785, 0.00000, 0.10590, 0.00793, 0.00243, 0.02260, 0.12572,  0.08937, 0.12271, 0.07399, 0.00090},
        {0.31389, 0.02316, 0.02963, 0.33282, 0.00283, 0.00000, 0.00026, 0.06374, 0.00011, 0.00307, 0.00019, 0.00017, 0.00268, 0.00040, 0.00178, 0.00968, 0.00000, 0.12140, 0.00003, 0.02298, 0.00050, 0.00641,  0.00006, 0.05067, 0.01355, 0.00000},
        {0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.28399, 0.35845, 0.00556, 0.00000, 0.00000, 0.00090, 0.00000, 0.00258, 0.00002, 0.00001, 0.00000, 0.01545, 0.00000, 0.06721, 0.00000, 0.00000, 0.00000,  0.01986, 0.00000, 0.20069, 0.04527},
        {0.00000, 0.00000, 0.00015, 0.00000, 0.00001, 0.25152, 0.03548, 0.00000, 0.00700, 0.01556, 0.00187, 0.00000, 0.08079, 0.00441, 0.00794, 0.01202, 0.00002, 0.00001, 0.03901, 0.00018, 0.00002, 0.00281,  0.11112, 0.00020, 0.13298, 0.29689},
        {0.00000, 0.00000, 0.00800, 0.00000, 0.00000, 0.00074, 0.05934, 0.00000, 0.63829, 0.00089, 0.00001, 0.00000, 0.15892, 0.00006, 0.00068, 0.00120, 0.01268, 0.00000, 0.04383, 0.00000, 0.00000, 0.00000,  0.05608, 0.00001, 0.01925, 0.00001},
        {0.00000, 0.00000, 0.00924, 0.00329, 0.00131, 0.05144, 0.05040, 0.01550, 0.03788, 0.17822, 0.00000, 0.00000, 0.06409, 0.00215, 0.08463, 0.20608, 0.00000, 0.02951, 0.00096, 0.00002, 0.01385, 0.09683,  0.00279, 0.01555, 0.13625, 0.00000},
        {0.00000, 0.00000, 0.01023, 0.04985, 0.00004, 0.03011, 0.06443, 0.13622, 0.32661, 0.10648, 0.00002, 0.00001, 0.00124, 0.00090, 0.03615, 0.11920, 0.05768, 0.00007, 0.03948, 0.00000, 0.00011, 0.00054,  0.01055, 0.00207, 0.00629, 0.00174},
        {0.26649, 0.01064, 0.01257, 0.26832, 0.00159, 0.00000, 0.00119, 0.09857, 0.00245, 0.05063, 0.00000, 0.00028, 0.00000, 0.20071, 0.02666, 0.04845, 0.00000, 0.01064, 0.00000, 0.00078, 0.00000, 0.00000,  0.00000, 0.00000, 0.00003, 0.00000},
        {0.04255, 0.25021, 0.00590, 0.23918, 0.01079, 0.00000, 0.00000, 0.01218, 0.00000, 0.04990, 0.00000, 0.05568, 0.00000, 0.00000, 0.00628, 0.00005, 0.00000, 0.23172, 0.00000, 0.08723, 0.00001, 0.00830,  0.00000, 0.00003, 0.00000, 0.00000},
        {0.00016, 0.00000, 0.02490, 0.00000, 0.00000, 0.34212, 0.08857, 0.00000, 0.00378, 0.00000, 0.12147, 0.00000, 0.06587, 0.09245, 0.00947, 0.00191, 0.00002, 0.00082, 0.05502, 0.00000, 0.00000, 0.00000,  0.07910, 0.00000, 0.06798, 0.04635},
        {0.00001, 0.00000, 0.02120, 0.00000, 0.00000, 0.02093, 0.14904, 0.00000, 0.00502, 0.00000, 0.00000, 0.00000, 0.34244, 0.11262, 0.00056, 0.00056, 0.00064, 0.00028, 0.04049, 0.00000, 0.00000, 0.00000,  0.13313, 0.00000, 0.17304, 0.00003},
        {0.00000, 0.00000, 0.01108, 0.00002, 0.00397, 0.01011, 0.17345, 0.15365, 0.00010, 0.06033, 0.00000, 0.00000, 0.44594, 0.00000, 0.00334, 0.05394, 0.00000, 0.00000, 0.00004, 0.01179, 0.00000, 0.00046,  0.07153, 0.00000, 0.00003, 0.00022},
        {0.00001, 0.00000, 0.01417, 0.02289, 0.08793, 0.13296, 0.02469, 0.09088, 0.00499, 0.04316, 0.07805, 0.00010, 0.00189, 0.00030, 0.01775, 0.01040, 0.02131, 0.00133, 0.00379, 0.00833, 0.00433, 0.10154,  0.06241, 0.00069, 0.07528, 0.19084},
        {0.08438, 0.08226, 0.01378, 0.03197, 0.00042, 0.00000, 0.00022, 0.11227, 0.00220, 0.05003, 0.00642, 0.00089, 0.01565, 0.02283, 0.05330, 0.05572, 0.00000, 0.09810, 0.00008, 0.20549, 0.00000, 0.00137,  0.00001, 0.00006, 0.16256, 0.00000},
        {0.00000, 0.00000, 0.01490, 0.00000, 0.05400, 0.13293, 0.03114, 0.00108, 0.01530, 0.14608, 0.04551, 0.00000, 0.00138, 0.03998, 0.00747, 0.01963, 0.33012, 0.00000, 0.01077, 0.00086, 0.00005, 0.00040,  0.03689, 0.00014, 0.05136, 0.06001},
        {0.00000, 0.00000, 0.00092, 0.00000, 0.00037, 0.00839, 0.16992, 0.00328, 0.00005, 0.28383, 0.00002, 0.00030, 0.12015, 0.05617, 0.00313, 0.04546, 0.00020, 0.00067, 0.01958, 0.00029, 0.00078, 0.03464,  0.03506, 0.02611, 0.19047, 0.00021},
        {0.00000, 0.00000, 0.02499, 0.00000, 0.02171, 0.00070, 0.05499, 0.00442, 0.01952, 0.25357, 0.00011, 0.00000, 0.03677, 0.02107, 0.06289, 0.18493, 0.00001, 0.00009, 0.06422, 0.08686, 0.00001, 0.02449,  0.01503, 0.00002, 0.12345, 0.00013},
        {0.03192, 0.23101, 0.02671, 0.05161, 0.00323, 0.00000, 0.00000, 0.04917, 0.00005, 0.01894, 0.08167, 0.00227, 0.00030, 0.00373, 0.01252, 0.01187, 0.00000, 0.09547, 0.00000, 0.27048, 0.00001, 0.09781,  0.00000, 0.00046, 0.01076, 0.00000},
        {0.00000, 0.00000, 0.00208, 0.00263, 0.00006, 0.01559, 0.10593, 0.00504, 0.04463, 0.17568, 0.00417, 0.00000, 0.03277, 0.03256, 0.13007, 0.12122, 0.00042, 0.00025, 0.02047, 0.00000, 0.00085, 0.01335,  0.08827, 0.00923, 0.16602, 0.02871},
        {0.01813, 0.05751, 0.01011, 0.43858, 0.02202, 0.00000, 0.00002, 0.09912, 0.00000, 0.00538, 0.00000, 0.06585, 0.00000, 0.00000, 0.00083, 0.00065, 0.00000, 0.01209, 0.00000, 0.16838, 0.00004, 0.07356,  0.00002, 0.02770, 0.00001, 0.00000},
        {0.00141, 0.00000, 0.03137, 0.00004, 0.00002, 0.00001, 0.00124, 0.06556, 0.01680, 0.14688, 0.00688, 0.00053, 0.01023, 0.01293, 0.08481, 0.43003, 0.00000, 0.05217, 0.00258, 0.00014, 0.00736, 0.06429,  0.02117, 0.01703, 0.02651, 0.00000},
    };

    for(i = 0; i < N; ++i) {
        for(j = 0; j < N; ++j) {
            A[i][j] = tempA[i][j];
        }
    }
    
    // initialize B[][]
    prob = 1.0 / (double)M;
    ftemp = prob / 10.0;
    for(i = 0; i < N; ++i)
    {
        ftemp2 = 0.0;
        for(j = 0; j < M; ++j)
        {
            if((random() & 0x1) == 0)
            {
                B[i][j] = prob + (double)(random() & 0x7) / 8.0 * ftemp;
            }
            else
            {
                B[i][j] = prob - (double)(random() & 0x7) / 8.0 * ftemp;
            }
            ftemp2 += B[i][j];
            
        }// next j
        
        for(j = 0; j < M; ++j)
        {
            B[i][j] /= ftemp2;
        }
        
    }// next i
    
}// end initMatrices


//
// read (but don't save) observations get T
//


//
// print pi[]
//
void printPi(double pi[])
{
    int i;
        
    double ftemp;

    ftemp = 0.0;
    for(i = 0; i < N; ++i)
    {
        printf("%8.5f ", pi[i]);
        ftemp += pi[i];
    }
    printf(",  sum = %f\n", ftemp);

}// end printPi


//
// print A[][]
//
void printA(double A[][N])
{
    int i,
        j;
        
    double ftemp;

    for(i = 0; i < N; ++i)
    {
        ftemp = 0.0;
        for(j = 0; j < N; ++j)
        {
            printf("%8.5f ", A[i][j]);
            ftemp += A[i][j];
        }
        printf(",  sum = %f\n", ftemp);
        
    }// next i

}// end printA

int GetT()
{
    int totalNum = 0;

    for (int i = 0; i < 24; ++i)
    {
        for (int j = 0; j < 17; ++j)
        {
            // Assuming the Z408 array directly contains the observation indices
            totalNum++;
        }
    }
    
    return totalNum;
}

int GetObservations(struct stepStruct *step, int T)
{
    int num = 0;
    
    for (int i = 0; i < 24; ++i)
    {
        for (int j = 0; j < 17; ++j)
        {
            // Assuming the Z408 array directly contains the observation indices
            step[num].obs = Z408[i][j] - 1; // Assuming observations indices start from 0
            num++;
            if(num > T)
            {
                printf("\nError --- T exceeded in GetObservations()\n\n");
                exit(0);
            }
        }
    }
    
    return num;
}

//
// print BT[][]
//
void printBT(double B[][M])
{
    int i, j;
    double ftemp;
    int alphabet[M] = ALPHABET;
    
    for(i = 0; i < M; ++i)
    {
        printf("%d ", alphabet[i]);
        for(j = 0; j < N; ++j)
        {
            printf("%8.5f ", B[j][i]);
        }
        printf("\n");
    }
    for(i = 0; i < N; ++i)
    {
        ftemp = 0.0;
        for(j = 0; j < M; ++j)
        {
            ftemp += B[i][j];
        }
        printf("sum[%d] = %f ", i, ftemp);
    }
    printf("\n");
}

