#include <CL/opencl.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int n = 100000;
unsigned int sqrtn = sqrt(n) + 1;
unsigned int* rootPrimes;
unsigned char* sieve;
char* kernelCode;

cl_platform_id platformID[1];
cl_device_id deviceID;
cl_context context;
cl_command_queue commandQueue;
cl_program kernelProgram;
cl_kernel kernel;
cl_mem sieveBuffer, rootPrimesBuffer;

void OpenCLsetup() {
    // read file
    // rb because of utf errors
    FILE* f = fopen("primesKernel", "rb");
    fseek(f, 0, SEEK_END);
    int length = ftell(f);
    kernelCode = (char*)malloc(length + 1);
    fseek(f, 0, SEEK_SET);
    fread(kernelCode, 1, length, f);
    kernelCode[length] = '\0';
    fclose(f);

    // setup context, command queue and program
    clGetPlatformIDs(1, platformID, NULL);
    clGetDeviceIDs(platformID[0], CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL);
    context = clCreateContext(0, 1, &deviceID, NULL, NULL, NULL);
    commandQueue = clCreateCommandQueue(context, deviceID, 0, NULL);

    kernelProgram = clCreateProgramWithSource(context, 1, (const char**)&kernelCode, NULL, NULL);
    clBuildProgram(kernelProgram, 0, NULL, NULL, NULL, NULL);
    cl_build_status status;
    clGetProgramBuildInfo(kernelProgram, deviceID, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL);
    kernel = clCreateKernel(kernelProgram, "sieveOfEratosthenes", NULL);

    // buffer allocation
    rootPrimesBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int) * sqrtn, NULL, NULL);
    sieveBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, n, NULL, NULL);
}


void sieveGPU(unsigned char* sieve) {
    // create sieve
    for (int i = 0; i < n; i++) {
        sieve[i] = 1;
    }

    // load sieve and root primes array to the gpu
    clEnqueueWriteBuffer(commandQueue, sieveBuffer, CL_TRUE, 0, sizeof(unsigned char) * n, sieve, 0, NULL, NULL);
    clEnqueueWriteBuffer(commandQueue, rootPrimesBuffer, CL_TRUE, 0, sizeof(unsigned int) * sqrtn, rootPrimes, 0, NULL, NULL);

    // setup local and global work sizes
    size_t globalSize, localSize;
    clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &localSize, NULL);
    localSize /= 32;
    globalSize = localSize * 32;

    // load arguments and command queue
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rootPrimesBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &sieveBuffer);
    clSetKernelArg(kernel, 2, sizeof(long long int), &n);
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    // excecute queue
    clFinish(commandQueue);

    // get sieve
    clEnqueueReadBuffer(commandQueue, sieveBuffer, CL_TRUE, 0, sizeof(unsigned char) * n, sieve, 0, NULL, NULL);
}


int main() {
    OpenCLsetup();
    int primes = 0;

    // calculate primes to sqrt(n)
    unsigned char* rootPrimesSieve = (unsigned char*)malloc(sizeof(unsigned char) * (sqrtn + 1));
    for (int i = 0; i <= sqrtn; i++) {
        rootPrimesSieve[i] = 1;
    }
    rootPrimesSieve[0] = 0;
    rootPrimesSieve[1] = 0;
    for (int i = 2; i * i <= sqrtn; i++) {
        if (rootPrimesSieve[i]) {
            for (int j = i * i; j <= sqrtn; j += i)
                rootPrimesSieve[j] = 0;
        }
    }

    // put primes to a list
    rootPrimes = (unsigned int*)malloc(sizeof(unsigned int) * (sqrtn));
    for (int i = 0; i <= sqrtn; i++) {
        if (rootPrimesSieve[i]) {
            rootPrimes[primes] = i;
            primes++;
        }
    }

    // calculate in gpu
    sieve = (unsigned char*)malloc(sizeof(unsigned char) * n);
    double start, stop;
    start = omp_get_wtime();
    sieveGPU(sieve);
    stop = omp_get_wtime();

    // count primes
    for (int i = 0; i < n; i++) {
        if (sieve[i]) {
            primes++;
        }
    }

    printf("%d -> %f\n", primes - 1, stop - start);

    // free alocated memory
    free(kernelCode);
    free(rootPrimes);
    free(rootPrimesSieve);
    free(sieve);
    
    return 0;
}
