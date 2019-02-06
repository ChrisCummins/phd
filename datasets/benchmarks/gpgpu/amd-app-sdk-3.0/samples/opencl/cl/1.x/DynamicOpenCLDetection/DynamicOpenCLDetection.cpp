#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

typedef int (*FuncType)(int, char**);
FuncType Function;

const int NUM_ELEMENTS = 256;

/**
*******************************************************************************
* @fn fillRandom
* @brief Fill vector with random numbers.
*
* @param vector stores the random sequence
* @param width and height the size of arrayPtr
* @param the range of the element
* @param seed
* @return int SDK_SUCCESS on success and nonzero on failure.
*******************************************************************************
*/
int fillRandom(
         std::vector<float> &vec, 
         const int width,
         const int height,
         const float rangeMin,
         const float rangeMax,
         unsigned int seed = 123);

/**
*******************************************************************************
* @fn printVector
* @brief Print the vector elements.
*
* @param header Vector name.
* @param vec The vector.
*******************************************************************************
*/
void printVector(
    char* header, 
    const std::vector<float> vec);

/**
*******************************************************************************
* @fn CPUAdd
* @brief Perform addition of random numbers on CPU
*         when OpenCL runtime is not found.
*
* @return int SDK_SUCCESS on success and nonzero on failure.
*******************************************************************************
*/
int CPUAdd()
{
  std::vector<float> inputA(NUM_ELEMENTS);  // Input array "A".
  if (fillRandom(inputA, NUM_ELEMENTS, 1, 0, 255) != SDK_SUCCESS)
    return SDK_FAILURE;
  printVector((char *)"inputA:", inputA);

  std::vector<float> inputB(NUM_ELEMENTS);  // Input array "B".
  if (fillRandom(inputB, NUM_ELEMENTS, 1, 0, 255, 100) != SDK_SUCCESS)
    return SDK_FAILURE;
  printVector((char *)"inputB:", inputB);

  std::vector<float> output(NUM_ELEMENTS);  // "Output" array.

  for (int i = 0; i < NUM_ELEMENTS; i++)
      output[i] = inputA[i] + inputB[i];
  printVector((char *)"output:", output);

  return SDK_SUCCESS;
}

/* Running GPU or CPU versions of a code by dynamically detecting the
    OpenCL runtime */
int main(int argc, char **argv) {

/* Check for OpenCL runtime - OpenCL.dll in Windows
                          and libOpenCL.so in linux */

char funcName[20] = "VectorAdd";  // function to be found in .dll/.so

#ifdef _WIN32
  if (LoadLibrary("OpenCL.dll"))
  {
    std::cout << "OpenCL.dll found. Running using OpenCL runtime..."
              << std::endl;

    /* Check if the VectorAddition.dll file that contains the gpu
    implementation is present */
    HMODULE handle;
    handle = LoadLibrary("VectorAddition.dll");
    if (!handle)
    {
      std::cout << "VectorAddition.dll not found." << std::endl;	
      return SDK_FAILURE;
    }

    Function = (FuncType)GetProcAddress(handle, funcName);
    if (GetLastError())
    {
        std::cout << "Function \"" << funcName << "\" not found."
                  << std::endl;
        return SDK_FAILURE;
    }

    (*Function)(0, NULL);

    FreeLibrary(handle);
  }
  else
  {
    std::cout << "OpenCL.dll not found. Running using CPU path..."
              << std::endl;
    if (CPUAdd() != SDK_SUCCESS)
      return SDK_FAILURE;
  }
#else
  if (dlopen("libOpenCL.so", RTLD_LAZY))
  {
    std::cout << "libOpenCL.so found. Running using OpenCL runtime..."
              << std::endl;

    /* Check if the VectorAddition.so file that contains the gpu
    implementation is present */
    void *handle;
    char *error;
    handle = dlopen("./libVectorAddition.so", RTLD_LAZY);
    if (!handle)
    {
      std::cout << "VectorAddition.so not found." << std::endl;
      return SDK_FAILURE;
    }

    Function = (FuncType)dlsym(handle, funcName);
    if ((error = dlerror()) != NULL)  {
      fputs(error, stderr);
      return SDK_FAILURE;
    }

    (*Function)(0, NULL);

    dlclose(handle);
  }
  else
  {
    std::cout << "libOpenCL.so not found. Running using CPU path..."
              << std::endl;
    if (CPUAdd() != SDK_SUCCESS)
      return SDK_FAILURE;
  }
#endif

  std::cout << std::endl;
  return SDK_SUCCESS;
}

/**
*******************************************************************************
* Implementation of printVector                                               *
******************************************************************************/
void printVector(
    char* header, 
    const std::vector<float> vec)
{
    std::cout << std::endl << header << std::endl;

    // print all the elements of the data 
    for(std::vector<float>::size_type ix = 0; ix != vec.size(); ++ix)
    {
        std::cout << vec[ix] << " ";
    }
    std::cout << std::endl;
}

/**
*******************************************************************************
* Implementation of fillRandom                                                *
******************************************************************************/
int fillRandom(
         std::vector<float> &vec, 
         const int width,
         const int height,
         const float rangeMin,
         const float rangeMax,
         unsigned int seed)
{
    if(vec.empty())
    {
        std::cout << "Cannot fill vector." << std::endl;
        return SDK_FAILURE;
    }

    // set seed
    if(!seed)
        seed = (unsigned int)time(NULL);

    srand(seed);

    // set the range
    double range = double(rangeMax - rangeMin) + 1.0; 

    /* random initialisation of input */
    for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++)
        {
            int index = i*width + j;
            vec[index] = rangeMin + float(range*rand()/(RAND_MAX + 1.0)); 
        }

    return SDK_SUCCESS;
}
