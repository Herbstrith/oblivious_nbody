/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <GL/glew.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <paramgl.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <assert.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.h"
#include <helper_cuda_gl.h>

#include <helper_functions.h>

#include "bodysystemcuda.h"
#include "bodysystemcpu.h"
#include "cuda_runtime.h"
#include <sys/time.h>

// time measurement
clock_t start, end;


// view params
int ox = 0, oy = 0;
int buttonState          = 0;
float camera_trans[]     = {0, -2, -150};
float camera_rot[]       = {0, 0, 0};
float camera_trans_lag[] = {0, -2, -150};
float camera_rot_lag[]   = {0, 0, 0};
const float inertia      = 0.1f;


bool benchmark = false;
bool compareToCPU = false;
bool QATest = false;
int blockSize = 256;
bool useHostMem = false;
bool fp64 = false;
bool useCpu = false;
int  numDevsRequested = 1;
bool displayEnabled = false;
bool bPause = false;
bool bFullscreen = false;
bool bDispInteractions = false;
bool bSupportDouble = false;
int flopsPerInteraction = 20;

char deviceName[100];

enum { M_VIEW = 0, M_MOVE };

int numBodies = 16384;

std::string tipsyFile = "";

int numIterations = 0; // run until exit


const int timeFactor = 1;       //1000 for ms or 1000000 for s;


void computePerfStats(double &interactionsPerSecond, double &gflops,
                      float milliseconds, int iterations)
{
    // double precision uses intrinsic operation followed by refinement,
    // resulting in higher operation count per interaction.
    // (Note Astrophysicists use 38 flops per interaction no matter what,
    // based on "historical precedent", but they are using FLOP/s as a
    // measure of "science throughput". We are using it as a measure of
    // hardware throughput.  They should really use interactions/s...
    // const int flopsPerInteraction = fp64 ? 30 : 20;
    interactionsPerSecond = (float)numBodies * (float)numBodies;
    interactionsPerSecond *= 1e-9 * iterations * 1000000 / milliseconds;
    gflops = interactionsPerSecond * (float)flopsPerInteraction;
}


////////////////////////////////////////
// Demo Parameters
////////////////////////////////////////
struct NBodyParams
{
    float m_timestep;
    float m_clusterScale;
    float m_velocityScale;
    float m_softening;
    float m_damping;
    float m_pointSize;
    float m_x, m_y, m_z;

    void print()
    {
        printf("{ %f, %f, %f, %f, %f, %f, %f, %f, %f },\n",
               m_timestep, m_clusterScale, m_velocityScale,
               m_softening, m_damping, m_pointSize, m_x, m_y, m_z);
    }
};

NBodyParams demoParams[] =
{
    { 0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    { 0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    { 0.0006f, 0.16f, 1000000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    { 0.0006f, 0.16f, 1000000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    { 0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
    { 0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
    { 0.016000f, 6.040000f, 0.000000f, 1.000000f, 1.000000f, 0.760000f, 0, 0, -50},
};

int numDemos = sizeof(demoParams) / sizeof(NBodyParams);
bool cycleDemo = true;
int activeDemo = 0;
float demoTime = 10000000.0f; // ms
StopWatchInterface *demoTimer = NULL, *timer = NULL;

// run multiple iterations to compute an average sort time

NBodyParams activeParams = demoParams[activeDemo];

// The UI.
//ParamListGL *paramlist;  // parameter list
bool bShowSliders = true;

// fps
static int fpsCount = 0;
static int fpsLimit = 5;
cudaEvent_t startEvent, stopEvent;
cudaEvent_t startEventIteration, stopEventIteration;
cudaEvent_t hostMemSyncEvent;

template <typename T>
class NBodyDemo
{
    public:
        static void Create()
        {
            m_singleton = new NBodyDemo;
        }
        static void Destroy()
        {
            delete m_singleton;
        }

        static void init(int numBodies, int numDevices, int blockSize,
                         bool usePBO, bool useHostMem, bool useCpu)
        {
            m_singleton->_init(numBodies, numDevices, blockSize, usePBO, useHostMem, useCpu);
        }

        static void reset(int numBodies, NBodyConfig config)
        {
            m_singleton->_reset(numBodies, config);
        }

        static void runBenchmark(int iterations)
        {
            m_singleton->_runBenchmark(iterations);
        }



        static void getArrays(T *pos, T *vel)
        {
            struct timeval tr_start,tr_end;
            gettimeofday(&tr_start, NULL);

            T *_pos = m_singleton->m_nbody->getArray(BODYSYSTEM_POSITION);
            T *_vel = m_singleton->m_nbody->getArray(BODYSYSTEM_VELOCITY);
            memcpy(pos, _pos, m_singleton->m_nbody->getNumBodies() * 4 * sizeof(T));
            memcpy(vel, _vel, m_singleton->m_nbody->getNumBodies() * 4 * sizeof(T));
            //gettimeofday(&tr_end, NULL);
            //printf( "Time to getarrays memcpy and cudamemcpy : %f\n", ((double)tr_end.tv_sec+(double)tr_end.tv_usec/1000000) - ((double)tr_start.tv_sec+(double)tr_start.tv_usec/1000000) );

        }

        static void setArrays(const T *pos, const T *vel)
        {
            struct timeval tr_start,tr_end;
            gettimeofday(&tr_start, NULL);

            if (pos != m_singleton->m_hPos)
            {
                memcpy(m_singleton->m_hPos, pos, numBodies * 4 * sizeof(T));
            }

            if (vel != m_singleton->m_hVel)
            {
                memcpy(m_singleton->m_hVel, vel, numBodies * 4 * sizeof(T));
            }

            //gettimeofday(&tr_end, NULL);
            //printf( "Time to setarrays memcpy : %f\n", ((double)tr_end.tv_sec+(double)tr_end.tv_usec/1000000) - ((double)tr_start.tv_sec+(double)tr_start.tv_usec/1000000) );
            //struct timeval cudatr_start,cudatr_end;
            //gettimeofday(&cudatr_start, NULL);

            m_singleton->m_nbody->setArray(BODYSYSTEM_POSITION, m_singleton->m_hPos);

            m_singleton->m_nbody->setArray(BODYSYSTEM_VELOCITY, m_singleton->m_hVel);

            //gettimeofday(&cudatr_end, NULL);
            //printf( "Time to setarrays cudamemcpy : %f\n", ((double)cudatr_end.tv_sec+(double)cudatr_end.tv_usec/1000000) - ((double)cudatr_start.tv_sec+(double)cudatr_start.tv_usec/1000000) );

        }

    private:
        static NBodyDemo *m_singleton;

        BodySystem<T>     *m_nbody;
        BodySystemCUDA<T> *m_nbodyCuda;
        BodySystemCPU<T>  *m_nbodyCpu;



        T *m_hPos;
        T *m_hVel;
        float *m_hColor;

    private:
        NBodyDemo()
            : m_nbody(0),
              m_nbodyCuda(0),
              m_nbodyCpu(0),
              m_hPos(0),
              m_hVel(0),
              m_hColor(0)
        {

        }

        ~NBodyDemo()
        {
            if (m_nbodyCpu)
            {
                delete m_nbodyCpu;
            }

            if (m_nbodyCuda)
            {
                delete m_nbodyCuda;
            }

            if (m_hPos)
            {
                delete [] m_hPos;
            }

            if (m_hVel)
            {
                delete [] m_hVel;
            }

            if (m_hColor)
            {
                delete [] m_hColor;
            }

            sdkDeleteTimer(&demoTimer);


        }

        void _init(int numBodies, int numDevices, int blockSize,
                   bool bUsePBO, bool useHostMem, bool useCpu)
        {
            if (useCpu)
            {
                m_nbodyCpu = new BodySystemCPU<T>(numBodies);
                m_nbody = m_nbodyCpu;
                m_nbodyCuda = 0;
            }
            else
            {
                m_nbodyCuda = new BodySystemCUDA<T>(numBodies, numDevices, blockSize, bUsePBO, useHostMem);
                m_nbody = m_nbodyCuda;
                m_nbodyCpu = 0;
            }

            // allocate host memory
            m_hPos = new T[numBodies*4];
            m_hVel = new T[numBodies*4];
            m_hColor = new float[numBodies*4];

            m_nbody->setSoftening(activeParams.m_softening);
            m_nbody->setDamping(activeParams.m_damping);

            if (useCpu)
            {
                sdkCreateTimer(&timer);
                sdkStartTimer(&timer);
            }
            else
            {
                checkCudaErrors(cudaEventCreate(&startEvent));
                checkCudaErrors(cudaEventCreate(&stopEvent));
                checkCudaErrors(cudaEventCreate(&startEventIteration));
                checkCudaErrors(cudaEventCreate(&stopEventIteration));
                checkCudaErrors(cudaEventCreate(&hostMemSyncEvent));
            }



            sdkCreateTimer(&demoTimer);
            sdkStartTimer(&demoTimer);
        }

        void _reset(int numBodies, NBodyConfig config)
        {
            if (tipsyFile == "")
            {
                randomizeBodies(config, m_hPos, m_hVel, m_hColor,
                                activeParams.m_clusterScale,
                                activeParams.m_velocityScale,
                                numBodies, true);
                setArrays(m_hPos, m_hVel);
            }
            else
            {
                m_nbody->loadTipsyFile(tipsyFile);
                ::numBodies = m_nbody->getNumBodies();
            }
        }

        void _runBenchmark(int iterations)
        {
            // once without timing to prime the device
            if (!useCpu)
            {
                m_nbody->update(activeParams.m_timestep);
            }

            if (useCpu)
            {
                sdkCreateTimer(&timer);
                sdkStartTimer(&timer);
            }
            else
            {
                checkCudaErrors(cudaEventRecord(startEvent, 0));
            }

            struct timeval tr_start,tr_end;
            gettimeofday(&tr_start, NULL);


            struct timeval tr_start_iter,tr_end_iter;
            float counter = 0;
            for (int i = 0; i < iterations; ++i)
            {
                float milliseconds_iteration = 0;
                checkCudaErrors(cudaEventRecord(startEventIteration, 0));

                m_nbody->update(activeParams.m_timestep);

                checkCudaErrors(cudaEventRecord(stopEventIteration, 0));
                            checkCudaErrors(cudaEventSynchronize(stopEventIteration));
                checkCudaErrors(cudaEventElapsedTime(&milliseconds_iteration, startEventIteration, stopEventIteration));
                            printf("%.3f iteration\n",milliseconds_iteration);
            }
            //loop unroling
            /*m_nbody->update(activeParams.m_timestep);
            m_nbody->update(activeParams.m_timestep);
            m_nbody->update(activeParams.m_timestep);
            m_nbody->update(activeParams.m_timestep);
            m_nbody->update(activeParams.m_timestep);
            m_nbody->update(activeParams.m_timestep);
            m_nbody->update(activeParams.m_timestep);
            m_nbody->update(activeParams.m_timestep);
            m_nbody->update(activeParams.m_timestep);
            m_nbody->update(activeParams.m_timestep);
*/

            float milliseconds = 0;

            checkCudaErrors(cudaEventRecord(stopEvent, 0));
            checkCudaErrors(cudaEventSynchronize(stopEvent));
            checkCudaErrors(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));


            printf("%.3f gputime\n",milliseconds);
            //       numBodies, iterations, milliseconds);
            //printf("%d bodies, total time for %d iterations: %.3f ms\n",
            //       numBodies, iterations, milliseconds);
           // printf("= %.3f billion interactions per second\n", interactionsPerSecond);
            //printf("= %.3f %s-precision GFLOP/s at %d flops per interaction\n", gflops,
              //     (sizeof(T) > 4) ? "double" : "single", flopsPerInteraction);
        }
};

void finalize()
{
    struct timeval tr_start,tr_end;
    gettimeofday(&tr_start, NULL);

    if (!useCpu)
    {
        checkCudaErrors(cudaEventDestroy(startEvent));
        checkCudaErrors(cudaEventDestroy(stopEvent));
        checkCudaErrors(cudaEventDestroy(hostMemSyncEvent));
    }
    /*gettimeofday(&tr_end, NULL);
    printf( "finalize runtime %f \n", ((double)tr_end.tv_sec+(double)tr_end.tv_usec/1000000) - ((double)tr_start.tv_sec+(double)tr_start.tv_usec/1000000) );
    */
    NBodyDemo<float>::Destroy();
    /*gettimeofday(&tr_end, NULL);
    printf( "finalize runtime %f \n", ((double)tr_end.tv_sec+(double)tr_end.tv_usec/1000000) - ((double)tr_start.tv_sec+(double)tr_start.tv_usec/1000000) );
    */
    if (bSupportDouble) NBodyDemo<double>::Destroy();
    /*gettimeofday(&tr_end, NULL);
    printf( "finalize runtime %f \n", ((double)tr_end.tv_sec+(double)tr_end.tv_usec/1000000) - ((double)tr_start.tv_sec+(double)tr_start.tv_usec/1000000) );
    */
}

template <> NBodyDemo<double> *NBodyDemo<double>::m_singleton = 0;
template <> NBodyDemo<float> *NBodyDemo<float>::m_singleton = 0;

template <typename T_new, typename T_old>
void switchDemoPrecision()
{
    cudaDeviceSynchronize();

    fp64 = !fp64;
    flopsPerInteraction = fp64 ? 30 : 20;

    T_old *oldPos = new T_old[numBodies * 4];
    T_old *oldVel = new T_old[numBodies * 4];

    NBodyDemo<T_old>::getArrays(oldPos, oldVel);

    // convert float to double
    T_new *newPos = new T_new[numBodies * 4];
    T_new *newVel = new T_new[numBodies * 4];

    for (int i = 0; i < numBodies * 4; i++)
    {
        newPos[i] = (T_new)oldPos[i];
        newVel[i] = (T_new)oldVel[i];
    }

    NBodyDemo<T_new>::setArrays(newPos, newVel);

    cudaDeviceSynchronize();

    delete [] oldPos;
    delete [] oldVel;
    delete [] newPos;
    delete [] newVel;
}

void
showHelp()
{
    printf("\t-numbodies=<N>    (number of bodies (>= 1) to run in simulation) \n");
    printf("\t-device=<d>       (where d=0,1,2.... for the CUDA device to use)\n");
    printf("\t-numdevices=<i>   (where i=(number of CUDA devices > 0) to use for simulation)\n");
    printf("\t-numiterations=<N>   (run n-body simulation with n iterations)\n");
    printf("\t-tipsy=<file.bin> (load a tipsy model file for simulation)\n\n");
}


static double gettime(void){
 struct timeval tr;
 gettimeofday(&tr,NULL);
 return (double)tr.tv_sec+(double)tr.tv_usec/1000000;
}

//////////////////////////////////////////////////////////////////////////////
// Program main
//////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    struct timeval tr_start,tr_end, tr_app_start,tr_app_end;
    gettimeofday(&tr_app_start, NULL);
    gettimeofday(&tr_start, NULL);

   double t1,t2;
    benchmark = true;

    flopsPerInteraction =  20;

    int numDevsAvailable = 0;
    bool customGPU = false;
    cudaGetDeviceCount(&numDevsAvailable);

    if (numDevsAvailable < numDevsRequested)
    {
        printf("Error: only %d Devices available, %d requested.  Exiting.\n", numDevsAvailable, numDevsRequested);
        exit(EXIT_SUCCESS);
    }


    blockSize = 0;


    if (blockSize == 0)   // blockSize not set on command line
        blockSize = 256;

    // default number of bodies is #SMs * 4 * CTA size

    if (checkCmdLineFlag(argc, (const char **) argv, "numbodies"))
    {
        numBodies = getCmdLineArgumentInt(argc, (const char **)argv, "numbodies");

        if (numBodies % blockSize)
        {
            int newNumBodies = ((numBodies / blockSize) + 1) * blockSize;
            //printf("Warning: \"number of bodies\" specified %d is not a multiple of %d.\n", numBodies, blockSize);
            //printf("Rounding up to the nearest multiple: %d.\n", newNumBodies);
            numBodies = newNumBodies;
        }
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "numiterations"))
    {
        numIterations = getCmdLineArgumentInt(argc, (const char **)argv, "numiterations");
    }
   t1 = gettime();
    //printf( "---Timestamp init variable and args : %f\n", ((double)tr_end.tv_sec+(double)tr_end.tv_usec/1000000) - ((double)tr_start.tv_sec+(double)tr_start.tv_usec/1000000) );

    // Create the demo -- either double (fp64) or float (fp32, default) implementation
    NBodyDemo<float>::Create();

    //printf( "---Timestamp create class : %f\n", ((double)tr_end.tv_sec+(double)tr_end.tv_usec/1000000) - ((double)tr_start.tv_sec+(double)tr_start.tv_usec/1000000) );

    NBodyDemo<float>::init(numBodies, numDevsRequested, blockSize, !(benchmark || compareToCPU || useHostMem), useHostMem, useCpu);

    //printf( "---Timestamp initialize class(malloc) : %f\n", ((double)tr_end.tv_sec+(double)tr_end.tv_usec/1000000) - ((double)tr_start.tv_sec+(double)tr_start.tv_usec/1000000) );

    NBodyDemo<float>::reset(numBodies, NBODY_CONFIG_SHELL);

    printf( "%f memory \n",(gettime() - t1)*1000);

    if (numIterations <= 0)
    {
        numIterations = 10;
    }
    //printf( "---Timestamp before processing: %f\n", ((double)tr_end.tv_sec+(double)tr_end.tv_usec/1000000) - ((double)tr_start.tv_sec+(double)tr_start.tv_usec/1000000)* );
    NBodyDemo<float>::runBenchmark(numIterations);

    //printf( "---Timestamp after processing: %f\n", ((double)tr_end.tv_sec+(double)tr_end.tv_usec/1000000) - ((double)tr_start.tv_sec+(double)tr_start.tv_usec/1000000) );


    finalize();

   // printf( "%f runtime \n", (((double)tr_end.tv_sec+(double)tr_end.tv_usec/1000000) - ((double)tr_start.tv_sec+(double)tr_start.tv_usec/1000000))*timeFactor );
      printf("%f runtime \n", (gettime() - t1)*1000);
    //printf( "---total time application : %f\n", ((double)tr_app_end.tv_sec+(double)tr_app_end.tv_usec/1000000) - ((double)tr_app_start.tv_sec+(double)tr_app_start.tv_usec/1000000) );

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(0);
}
