/* Copyright (c) 2013 Simon Leblanc

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define BOLD ""
#define NORMAL ""

#define LENGTH(array) (sizeof(array) / sizeof(array[0]))

/*****************************************************************************\
▏ Constants                                                                   ▕
\*****************************************************************************/

const size_t indent_size = 2;
const size_t column_size = 40;

/*****************************************************************************\
▏ Types                                                                       ▕
\*****************************************************************************/

typedef void (*Printer)(size_t indent, const char* key, const char* value);

typedef void (*Formatter)(size_t indent, const char* key, void* value,
                          size_t size, Printer print);

typedef enum { BASIC, ADVANCED } Level;

typedef enum { PRETTY, RAW } Style;

typedef struct ParameterList {
  const char* name;
  struct ParameterList* next;
} ParameterList;

typedef struct {
  cl_platform_info id;
  const char* raw_name;
  const char* pretty_name;
  Formatter rawPrint;
  Formatter prettyPrint;
  Level level;
} PlatformParameter;

typedef struct {
  cl_device_info id;
  const char* raw_name;
  const char* pretty_name;
  Formatter rawPrint;
  Formatter prettyPrint;
  Level level;
} DeviceParameter;

/*****************************************************************************\
▏ Printers and formatters prototypes                                          ▕
\*****************************************************************************/

void printValue(size_t indent, const char* key, const char* value);
void printWithKey(size_t indent, const char* key, const char* value);

void printExecutionCapabilities(size_t indent, const char* key, void* value,
                                size_t size, Printer print);
void printGlobalMemCacheType(size_t indent, const char* key, void* value,
                             size_t size, Printer print);
void printLocalMemType(size_t indent, const char* key, void* value, size_t size,
                       Printer print);
void printQueueProperties(size_t indent, const char* key, void* value,
                          size_t size, Printer print);
void printFPConfig(size_t indent, const char* key, void* value, size_t size,
                   Printer print);
void printExtensions(size_t indent, const char* key, void* value, size_t size,
                     Printer print);
void printDimensions(size_t indent, const char* key, void* value, size_t size,
                     Printer print);
void printMemSize(size_t indent, const char* key, void* value, size_t size,
                  Printer print);
void printSize(size_t indent, const char* key, void* value, size_t size,
               Printer print);
void printUlong(size_t indent, const char* key, void* value, size_t size,
                Printer print);
void printUint(size_t indent, const char* key, void* value, size_t size,
               Printer print);
void printBool(size_t indent, const char* key, void* value, size_t size,
               Printer print);
void printDeviceType(size_t indent, const char* key, void* value, size_t size,
                     Printer print);
void printStringValue(size_t indent, const char* key, void* value, size_t size,
                      Printer print);
void printString(size_t indent, const char* key, void* value, size_t size,
                 Printer print);
#if __OPENCL_VERSION__ >= 120
void printKernels(size_t indent, const char* key, void* value, size_t size,
                  Printer print);
void printParentDevice(size_t indent, const char* key, void* value, size_t size,
                       Printer print);
void printPartitionProperties(size_t indent, const char* key, void* value,
                              size_t size, Printer print);
void printPartitionAffinityDomain(size_t indent, const char* key, void* value,
                                  size_t size, Printer print);
void printPartitionType(size_t indent, const char* key, void* value,
                        size_t size, Printer print);
#endif

/*****************************************************************************\
▏ List of parameters                                                          ▕
\*****************************************************************************/

const PlatformParameter platform_parameters[] = {
    {CL_PLATFORM_NAME, "CL_PLATFORM_NAME", "Name", printString, printString,
     BASIC},
    {CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR", "Vendor", printString,
     printString, ADVANCED},
    {CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION", "Version", printString,
     printString, BASIC},
    {CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE", "Profile", printString,
     printString, ADVANCED},
    {CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS", "Extensions",
     printString, printExtensions, ADVANCED}};

const DeviceParameter device_parameters[] = {
    // General
    {CL_DEVICE_NAME, "CL_DEVICE_NAME", "Name", printString, printString, BASIC},
    {CL_DEVICE_TYPE, "CL_DEVICE_TYPE", "Type", printDeviceType, printDeviceType,
     BASIC},
    {CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR", "Vendor", printString, printString,
     ADVANCED},
    {CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID", "Vendor ID", printUint,
     printUint, ADVANCED},
    {CL_DEVICE_PROFILE, "CL_DEVICE_PROFILE", "Profile", printString,
     printString, ADVANCED},
    {CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE", "Available", printBool,
     printBool, ADVANCED},
    {CL_DEVICE_VERSION, "CL_DEVICE_VERSION", "Version", printString,
     printString, BASIC},
    {CL_DRIVER_VERSION, "CL_DRIVER_VERSION", "Driver version", printString,
     printString, ADVANCED},
#if __OPENCL_VERSION__ >= 120
    {CL_DEVICE_PARENT_DEVICE, "CL_DEVICE_PARENT_DEVICE", "Parent device",
     printParentDevice, printParentDevice, ADVANCED},
    {CL_DEVICE_REFERENCE_COUNT, "CL_DEVICE_REFERENCE_COUNT", "Reference count",
     printUint, printUint, ADVANCED},
#endif

    // Compiler
    {CL_DEVICE_COMPILER_AVAILABLE, "CL_DEVICE_COMPILER_AVAILABLE",
     "Compiler available", printBool, printBool, ADVANCED},
#if __OPENCL_VERSION__ >= 120
    {CL_DEVICE_LINKER_AVAILABLE, "CL_DEVICE_LINKER_AVAILABLE",
     "Linker available", printBool, printBool, ADVANCED},
#endif
#if __OPENCL_VERSION__ >= 110
    {CL_DEVICE_OPENCL_C_VERSION, "CL_DEVICE_OPENCL_C_VERSION",
     "OpenCL C version", printString, printString, ADVANCED},
#endif

    // Misc
    {CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS", "Address space size",
     printUint, printUint, ADVANCED},
    {CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE", "Little endian",
     printBool, printBool, ADVANCED},
    {CL_DEVICE_ERROR_CORRECTION_SUPPORT, "CL_DEVICE_ERROR_CORRECTION_SUPPORT",
     "Error correction support", printBool, printBool, ADVANCED},
#if __OPENCL_VERSION__ >= 110
    {CL_DEVICE_HOST_UNIFIED_MEMORY, "CL_DEVICE_HOST_UNIFIED_MEMORY",
     "Unified memory", printBool, printBool, ADVANCED},
#endif
    {CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN",
     "Address alignment (bits)", printUint, printUint, ADVANCED},
#if __OPENCL_VERSION__ < 120
    {CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE",
     "Smallest alignment (bytes)", printUint, printUint, ADVANCED},
#endif
    {CL_DEVICE_PROFILING_TIMER_RESOLUTION,
     "CL_DEVICE_PROFILING_TIMER_RESOLUTION", "Resolution of timer (ns)",
     printSize, printSize, ADVANCED},
    {CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY",
     "Max clock frequency (MHz)", printUint, printUint, ADVANCED},
    {CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS",
     "Max compute units", printUint, printUint, ADVANCED},
    {CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS",
     "Max constant args", printUint, printUint, ADVANCED},
    {CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE",
     "Max constant buffer size", printUlong, printMemSize, ADVANCED},
    {CL_DEVICE_MAX_MEM_ALLOC_SIZE, "CL_DEVICE_MAX_MEM_ALLOC_SIZE",
     "Max mem alloc size", printUlong, printMemSize, ADVANCED},
    {CL_DEVICE_MAX_PARAMETER_SIZE, "CL_DEVICE_MAX_PARAMETER_SIZE",
     "Max parameter size", printSize, printSize, ADVANCED},
    {CL_DEVICE_QUEUE_PROPERTIES, "CL_DEVICE_QUEUE_PROPERTIES",
     "Command-queue supported props", printQueueProperties,
     printQueueProperties, ADVANCED},
    {CL_DEVICE_EXECUTION_CAPABILITIES, "CL_DEVICE_EXECUTION_CAPABILITIES",
     "Execution capabilities", printExecutionCapabilities,
     printExecutionCapabilities, ADVANCED},
#if __OPENCL_VERSION__ >= 120
    {CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,
     "CL_DEVICE_PREFERRED_INTEROP_USER_SYNC", "User synchronization preferred",
     printBool, printBool, ADVANCED},
    {CL_DEVICE_PRINTF_BUFFER_SIZE, "CL_DEVICE_PRINTF_BUFFER_SIZE",
     "Printf buffer size", printSize, printMemSize, ADVANCED},
#endif

    // Memory
    {CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE",
     "Global memory size", printUlong, printMemSize, BASIC},
    {CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE",
     "Global memory cache size", printUlong, printMemSize, ADVANCED},
    {CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE",
     "Global memory line cache size", printUint, printUint, ADVANCED},
    {CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE", "Local memory size",
     printUlong, printMemSize, BASIC},
    {CL_DEVICE_LOCAL_MEM_TYPE, "CL_DEVICE_LOCAL_MEM_TYPE", "Local memory type",
     printLocalMemType, printLocalMemType, ADVANCED},
    {CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE",
     "Global memory cache type", printGlobalMemCacheType,
     printGlobalMemCacheType, ADVANCED},

    // Work group
    {CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE",
     "Max work group size", printSize, printSize, BASIC},
    {CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS",
     "Max work item dimensions", printUint, printUint, ADVANCED},
    {CL_DEVICE_MAX_WORK_ITEM_SIZES, "CL_DEVICE_MAX_WORK_ITEM_SIZES",
     "Max work item sizes", printDimensions, printDimensions, BASIC},

    // Images
    {CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT", "Image support",
     printBool, printBool, ADVANCED},
    {CL_DEVICE_IMAGE2D_MAX_HEIGHT, "CL_DEVICE_IMAGE2D_MAX_HEIGHT",
     "Max 2D image height", printSize, printSize, ADVANCED},
    {CL_DEVICE_IMAGE2D_MAX_WIDTH, "CL_DEVICE_IMAGE2D_MAX_WIDTH",
     "Max 2D image width", printSize, printSize, ADVANCED},
    {CL_DEVICE_IMAGE3D_MAX_DEPTH, "CL_DEVICE_IMAGE3D_MAX_DEPTH",
     "Max 3D image depth", printSize, printSize, ADVANCED},
    {CL_DEVICE_IMAGE3D_MAX_HEIGHT, "CL_DEVICE_IMAGE3D_MAX_HEIGHT",
     "Max 3D image height", printSize, printSize, ADVANCED},
    {CL_DEVICE_IMAGE3D_MAX_WIDTH, "CL_DEVICE_IMAGE3D_MAX_WIDTH",
     "Max 3D image width", printSize, printSize, ADVANCED},
    {CL_DEVICE_MAX_READ_IMAGE_ARGS, "CL_DEVICE_MAX_READ_IMAGE_ARGS",
     "Max read image args", printUint, printUint, ADVANCED},
    {CL_DEVICE_MAX_WRITE_IMAGE_ARGS, "CL_DEVICE_MAX_WRITE_IMAGE_ARGS",
     "Max write image args", printUint, printUint, ADVANCED},
    {CL_DEVICE_MAX_SAMPLERS, "CL_DEVICE_MAX_SAMPLERS", "Max samplers",
     printUint, printUint, ADVANCED},
#if __OPENCL_VERSION__ >= 120
    {CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, "CL_DEVICE_IMAGE_MAX_BUFFER_SIZE",
     "Max pixels for 1D image from buffer", printSize, printSize, ADVANCED},
    {CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, "CL_DEVICE_IMAGE_MAX_ARRAY_SIZE",
     "Max images in 1D/2D image array", printSize, printSize, ADVANCED},
#endif

// Vectors
#if __OPENCL_VERSION__ >= 110
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR",
     "Native vector width char", printUint, printUint, ADVANCED},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT",
     "Native vector width short", printUint, printUint, ADVANCED},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT",
     "Native vector width int", printUint, printUint, ADVANCED},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG",
     "Native vector width long", printUint, printUint, ADVANCED},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF",
     "Native vector width half", printUint, printUint, ADVANCED},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT",
     "Native vector width float", printUint, printUint, ADVANCED},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
     "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE", "Native vector width double",
     printUint, printUint, ADVANCED},
#endif
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
     "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR", "Preferred vector width char",
     printUint, printUint, ADVANCED},
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
     "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT", "Preferred vector width short",
     printUint, printUint, ADVANCED},
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
     "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT", "Preferred vector width int",
     printUint, printUint, ADVANCED},
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
     "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG", "Preferred vector width long",
     printUint, printUint, ADVANCED},
#if __OPENCL_VERSION__ >= 110
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
     "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF", "Preferred vector width half",
     printUint, printUint, ADVANCED},
#endif
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
     "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT", "Preferred vector width float",
     printUint, printUint, ADVANCED},
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
     "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE", "Preferred vector width double",
     printUint, printUint, ADVANCED},

// Floating-points
#ifdef CL_DEVICE_HALF_FP_CONFIG
    {CL_DEVICE_HALF_FP_CONFIG, "CL_DEVICE_HALF_FP_CONFIG",
     "Half precision float capability", printFPConfig, printFPConfig, ADVANCED},
#endif
    {CL_DEVICE_SINGLE_FP_CONFIG, "CL_DEVICE_SINGLE_FP_CONFIG",
     "Single precision float capability", printFPConfig, printFPConfig,
     ADVANCED},
#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
    {CL_DEVICE_DOUBLE_FP_CONFIG, "CL_DEVICE_DOUBLE_FP_CONFIG",
     "Double precision float capability", printFPConfig, printFPConfig,
     ADVANCED},
#endif

// Partitioning
#if __OPENCL_VERSION__ >= 120
    {CL_DEVICE_PARTITION_MAX_SUB_DEVICES, "CL_DEVICE_PARTITION_MAX_SUB_DEVICES",
     "Max number of sub-devices", printUint, printUint, ADVANCED},
    {CL_DEVICE_PARTITION_PROPERTIES, "CL_DEVICE_PARTITION_PROPERTIES",
     "Partition types supported", printPartitionProperties,
     printPartitionProperties, ADVANCED},
    {CL_DEVICE_PARTITION_AFFINITY_DOMAIN, "CL_DEVICE_PARTITION_AFFINITY_DOMAIN",
     "Affinity domains supported", printPartitionAffinityDomain,
     printPartitionAffinityDomain, ADVANCED},
    {CL_DEVICE_PARTITION_TYPE, "CL_DEVICE_PARTITION_TYPE", "Partition type",
     printPartitionType, printPartitionType, ADVANCED},
#endif

// Built-in kernels
#if __OPENCL_VERSION__ >= 120
    {CL_DEVICE_BUILT_IN_KERNELS, "CL_DEVICE_BUILT_IN_KERNELS",
     "Built-in kernels", printString, printKernels, ADVANCED},
#endif

    // Extensions
    {CL_DEVICE_EXTENSIONS, "CL_DEVICE_EXTENSIONS", "Extensions", printString,
     printExtensions, ADVANCED},
};

/*****************************************************************************\
▏ Main helpers                                                                ▕
\*****************************************************************************/

void printHelp(const char* argv0) {
  printf("Usage: %s [-ahlr] [platform[:device]] [CL_PARAMETER ...]\n", argv0);
  printf("\n");
  printf("Options:\n");
  printf("  -a --all    Display all parameters.\n");
  printf("  -h --help   Display this help notice.\n");
  printf("  -l --list   List platforms and devices.\n");
  printf(
      "  -r --raw    Raw output (by default the values are pretty-printed).\n");
  printf("\n");
  printf("Further help:\n");
  printf("  man %s\n", argv0);
}

void parseOptions(int argc, char* argv[], Level* level, Style* style,
                  long* platform, long* device, int* list,
                  ParameterList** queries) {
  for (int i = 1; i < argc; ++i) {
    const char c = argv[i][0];
    if (c == '-') {
      for (const char* l = &argv[i][1]; *l != '\0'; ++l) {
        switch (*l) {
          case 'a':
            *level = ADVANCED;
            break;
          case 'h':
            printHelp(argv[0]);
            exit(0);
          case 'l':
            *list = 1;
            return;
          case 'r':
            *style = RAW;
            break;
          case '-':
            if (strcmp(++l, "all") == 0) {
              *level = ADVANCED;
              l += 2;
            } else if (strcmp(l, "help") == 0) {
              printHelp(argv[0]);
              exit(0);
            } else if (strcmp(l, "list") == 0) {
              *list = 1;
              return;
            } else if (strcmp(l, "raw") == 0) {
              *style = RAW;
              l += 2;
            } else {
              fprintf(stderr,
                      "%s: '--%s' is not a valid option. See '%s -h'.\n",
                      argv[0], l, argv[0]);
              exit(EXIT_FAILURE);
            }
            break;
          default:
            fprintf(stderr, "%s: '-%c' is not a valid option. See '%s -h'.\n",
                    argv[0], *l, argv[0]);
            exit(EXIT_FAILURE);
        }
      }
    } else if (isdigit(c)) {
      char* p;
      *platform = strtol(argv[i], &p, 10);
      if (*p == ':' && isdigit(*++p)) {
        *device = strtol(p, &p, 10);
      } else if (*p != '\0') {
        fprintf(stderr, "%s: '%s' is not a valid option. See '%s -h'.\n",
                argv[0], argv[i], argv[0]);
        exit(EXIT_FAILURE);
      }
    } else if (c == 'C') {
      int found = 0;
      for (size_t j = 0; j < LENGTH(platform_parameters); ++j) {
        if (strcmp(argv[i], platform_parameters[j].raw_name) == 0) {
          ParameterList* v = malloc(sizeof(ParameterList));
          v->name = argv[i];
          v->next = *queries;
          *queries = v;
          ++found;
        }
      }
      for (size_t j = 0; j < LENGTH(device_parameters); ++j) {
        if (strcmp(argv[i], device_parameters[j].raw_name) == 0) {
          ParameterList* v = malloc(sizeof(ParameterList));
          v->name = argv[i];
          v->next = *queries;
          *queries = v;
          ++found;
        }
      }
      if (!found) {
        fprintf(stderr, "%s: '%s' is not a valid option. See '%s -h'.\n",
                argv[0], argv[i], argv[0]);
        exit(EXIT_FAILURE);
      }
    } else {
      fprintf(stderr, "%s: '%s' is not a valid option. See '%s -h'.\n", argv[0],
              argv[i], argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

int shouldDisplay(const char* name, ParameterList* queries) {
  for (; queries; queries = queries->next) {
    if (strcmp(name, queries->name) == 0) {
      return 1;
    }
  }
  return 0;
}

void printPlatformInfo(const char* argv0, cl_platform_id platform,
                       cl_platform_info param, const char* name, size_t indent,
                       Formatter format, Printer print) {
  size_t buffer_size;
  cl_int status = clGetPlatformInfo(platform, param, 0, NULL, &buffer_size);
  if (status != CL_SUCCESS) {
    fprintf(stderr, "%s: Cannot get the size of the '%s' platform parameter.\n",
            argv0, name);
    exit(EXIT_FAILURE);
  }

  char* buffer = malloc(buffer_size);
  status = clGetPlatformInfo(platform, param, buffer_size, buffer, NULL);
  if (status != CL_SUCCESS) {
    fprintf(stderr, "%s: Cannot get the '%s' platform parameter.\n", argv0,
            name);
    exit(EXIT_FAILURE);
  }

  format(indent, name, buffer, buffer_size, print);

  free(buffer);
}

void printDeviceInfo(const char* argv0, cl_device_id device,
                     cl_device_info param, const char* name, size_t indent,
                     Formatter format, Printer print) {
  size_t buffer_size;
  cl_int status = clGetDeviceInfo(device, param, 0, NULL, &buffer_size);
  if (status != CL_SUCCESS) {
    fprintf(stderr, "%s: Cannot get the size of the '%s' device parameter.\n",
            argv0, name);
    exit(EXIT_FAILURE);
  }

  char* buffer = malloc(buffer_size);
  status = clGetDeviceInfo(device, param, buffer_size, buffer, NULL);
  if (status != CL_SUCCESS) {
    fprintf(stderr, "%s: Cannot get the '%s' device parameter.\n", argv0, name);
    exit(EXIT_FAILURE);
  }

  format(indent, name, buffer, buffer_size, print);

  free(buffer);
}

/*****************************************************************************\
▏ Main                                                                        ▕
\*****************************************************************************/

int main(int argc, char* argv[]) {
  Level level = BASIC;
  Style style = PRETTY;
  long specific_platform = -1;
  long specific_device = -1;
  int list = 0;
  ParameterList* queries = NULL;

  parseOptions(argc, argv, &level, &style, &specific_platform, &specific_device,
               &list, &queries);

  cl_int status;

  cl_uint num_platforms;
  status = clGetPlatformIDs(0, NULL, &num_platforms);
  if (status != CL_SUCCESS) {
    fprintf(stderr,
            "%s: Cannot get the number of OpenCL platforms available.\n",
            argv[0]);
    exit(EXIT_FAILURE);
  }

  if (specific_platform >= num_platforms) {
    fprintf(stderr, "%s: platform #%ld does not exist.\n", argv[0],
            specific_platform);
    exit(EXIT_FAILURE);
  }

  cl_platform_id platforms[num_platforms];
  status = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (status != CL_SUCCESS) {
    fprintf(stderr, "%s: Cannot get the list of OpenCL platforms.\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  size_t indent = 0;
  for (size_t i = 0; i < num_platforms; ++i) {
    if (specific_platform > -1 && i != specific_platform) continue;

    if (list) {
      if (style == RAW) {
        printf("%zu", i);
      } else {
        printf(BOLD "Platform #%zu:" NORMAL, i);
      }
      printPlatformInfo(argv[0], platforms[i], CL_PLATFORM_NAME, "Name", 0,
                        printString, printValue);
      putchar('\n');
    } else {
      if (style == PRETTY) {
        for (size_t n = 0; n < indent * indent_size; ++n) putchar(' ');
        printf(BOLD "Platform #%zu" NORMAL "\n", i);
        ++indent;
      }

      for (size_t k = 0; k < LENGTH(platform_parameters); ++k) {
        const PlatformParameter p = platform_parameters[k];
        if ((queries == NULL && p.level <= level) ||
            shouldDisplay(p.raw_name, queries)) {
          if (style == PRETTY) {
            printPlatformInfo(argv[0], platforms[i], p.id, p.pretty_name,
                              indent, p.prettyPrint, printWithKey);
          } else {
            printf("%zu %s", i, p.raw_name);
            printPlatformInfo(argv[0], platforms[i], p.id, p.raw_name, 0,
                              p.rawPrint, printValue);
            putchar('\n');
          }
        }
      }

      if (style == PRETTY) putchar('\n');
    }

    cl_uint num_devices;
    status =
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (status != CL_SUCCESS) {
      fprintf(stderr,
              "%s: Cannot get the number of OpenCL devices available on this "
              "platform.\n",
              argv[0]);
      exit(EXIT_FAILURE);
    }

    if (specific_device >= num_devices) {
      fprintf(stderr, "%s: device #%ld of platform #%ld does not exist.\n",
              argv[0], specific_device, specific_platform);
      exit(EXIT_FAILURE);
    }

    cl_device_id devices[num_devices];
    status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices,
                            devices, NULL);
    if (status != CL_SUCCESS) {
      fprintf(stderr, "%s: Cannot get the list of OpenCL devices.\n", argv[0]);
      exit(EXIT_FAILURE);
    }

    for (size_t j = 0; j < num_devices; ++j) {
      if (specific_device > -1 && j != specific_device) continue;

      if (list) {
        if (style == RAW) {
          printf("%zu:%zu", i, j);
        } else {
          printf("%s── " BOLD "Device #%zu:" NORMAL,
                 j + 1 < num_devices ? "├" : "└", j);
        }
        printDeviceInfo(argv[0], devices[j], CL_DEVICE_NAME, "Name", 0,
                        printString, printValue);
        putchar('\n');
      } else {
        if (style == PRETTY) {
          for (size_t n = 0; n < indent * indent_size; ++n) putchar(' ');
          printf(BOLD "Device #%zu" NORMAL "\n", j);
          ++indent;
        }

        for (size_t k = 0; k < LENGTH(device_parameters); ++k) {
          const DeviceParameter p = device_parameters[k];
          if ((queries == NULL && p.level <= level) ||
              shouldDisplay(p.raw_name, queries)) {
            if (style == PRETTY) {
              printDeviceInfo(argv[0], devices[j], p.id, p.pretty_name, indent,
                              p.prettyPrint, printWithKey);
            } else {
              printf("%zu:%zu %s", i, j, p.raw_name);
              printDeviceInfo(argv[0], devices[j], p.id, p.raw_name, 0,
                              p.rawPrint, printValue);
              putchar('\n');
            }
          }
        }

        if (style == PRETTY) {
          putchar('\n');
          --indent;
        }
      }
    }
    if (style == PRETTY) --indent;
  }

  while (queries) {
    ParameterList* query = queries;
    queries = queries->next;
    free(query);
  }

  return EXIT_SUCCESS;
}

/*****************************************************************************\
▏ Printers                                                                    ▕
\*****************************************************************************/

void printValue(size_t indent, const char* key, const char* value) {
  printf(" %s", value);
}

void printWithKey(size_t indent, const char* key, const char* value) {
  size_t n;
  for (n = 0; n < indent * indent_size; ++n) putchar(' ');
  for (n += key ? printf("%s:", key) : 0; n < column_size; ++n) putchar(' ');
  printf(" %s\n", value);
}

/*****************************************************************************\
▏ Formatters                                                                  ▕
\*****************************************************************************/

void printString(size_t indent, const char* key, void* value, size_t size,
                 Printer print) {
  print(indent, key, value);
}

void printStringValue(size_t indent, const char* key, void* value, size_t size,
                      Printer print) {
  printf("%s\n", value);
}

void printDeviceType(size_t indent, const char* key, void* value, size_t size,
                     Printer print) {
  const cl_device_type type = *((cl_device_type*)value);
  struct {
    cl_device_type type;
    const char* name;
  } list[] = {
    {CL_DEVICE_TYPE_CPU, "CPU"},
    {CL_DEVICE_TYPE_GPU, "GPU"},
    {CL_DEVICE_TYPE_ACCELERATOR, "Accelerator"},
#if __OPENCL_VERSION__ >= 120
    {CL_DEVICE_TYPE_CUSTOM, "Custom"},
#endif
    {CL_DEVICE_TYPE_DEFAULT, "Default"}
  };
  char buffer[45], *p = buffer;
  for (size_t i = 0; i < LENGTH(list); ++i) {
    if (type & list[i].type) {
      p += sprintf(p, "%s | ", list[i].name);
    }
  }
  if (p == buffer) {
    print(indent, key, "Unknown");
  } else {
    *(p - 3) = '\0';
    print(indent, key, buffer);
  }
}

void printBool(size_t indent, const char* key, void* value, size_t size,
               Printer print) {
  if (*((cl_bool*)value))
    print(indent, key, "Yes");
  else
    print(indent, key, "No");
}

void printUint(size_t indent, const char* key, void* value, size_t size,
               Printer print) {
  const cl_uint num = *((cl_uint*)value);
  char buffer[(num > 0 ? lrint(log10(num)) + 1 : 1) + 1];
  sprintf(buffer, "%u", num);
  print(indent, key, buffer);
}

void printUlong(size_t indent, const char* key, void* value, size_t size,
                Printer print) {
  const cl_ulong num = *((cl_ulong*)value);
  char buffer[(num > 0 ? lrint(log10(num)) + 1 : 1) + 1];
  sprintf(buffer, "%llu", num);
  print(indent, key, buffer);
}

void printSize(size_t indent, const char* key, void* value, size_t size,
               Printer print) {
  const size_t num = *((size_t*)value);
  char buffer[(num > 0 ? lrint(log10(num)) + 1 : 1) + 1];
  sprintf(buffer, "%zu", num);
  print(indent, key, buffer);
}

void printMemSize(size_t indent, const char* key, void* value, size_t size,
                  Printer print) {
  const cl_ulong mem_size = *((cl_ulong*)value);
  if (mem_size == 0) {
    print(indent, key, "0 B");
    return;
  }
  char buffer[((mem_size >> 40) > 0 ? lrint(log10(mem_size >> 40)) + 1 : 1) +
              38];
  int num, n = 0;
  if ((num = mem_size >> 40)) n += sprintf(&buffer[n], "%d TB ", num);
  if ((num = mem_size >> 30 & 1023)) n += sprintf(&buffer[n], "%d GB ", num);
  if ((num = mem_size >> 20 & 1023)) n += sprintf(&buffer[n], "%d MB ", num);
  if ((num = mem_size >> 10 & 1023)) n += sprintf(&buffer[n], "%d kB ", num);
  if ((num = mem_size & 1023)) n += sprintf(&buffer[n], "%d B", num);
  print(indent, key, buffer);
}

void printDimensions(size_t indent, const char* key, void* value, size_t size,
                     Printer print) {
  const size_t ndims = size / sizeof(size_t);
  const size_t* dims = *((size_t(*)[])value);
  size_t buffer_size = 1;
  for (size_t i = 0; i < ndims; ++i) {
    buffer_size += (dims[i] > 0 ? lrint(log10(dims[i])) + 1 : 1) + 2;
  }
  char buffer[buffer_size], *p = buffer;
  *p++ = '(';
  for (size_t i = 0; i < ndims; ++i) {
    p += sprintf(p, "%zu, ", dims[i]);
  }
  if (p + 1 > buffer) {
    *(p - 2) = ')';
    *(p - 1) = '\0';
  } else {
    *p++ = ')';
    *p++ = '\0';
  }
  print(indent, key, buffer);
}

void printExtensions(size_t indent, const char* key, void* value, size_t size,
                     Printer print) {
  char* item = strtok(value, " ");
  print(indent, key, item);
  while ((item = strtok(NULL, " "))) {
    print(indent, NULL, item);
  }
}

void printFPConfig(size_t indent, const char* key, void* value, size_t size,
                   Printer print) {
  const cl_device_fp_config config = *((cl_device_type*)value);
  const struct {
    cl_device_fp_config flag;
    const char* name;
  } list[] = {
#if __OPENCL_VERSION__ >= 110
    {CL_FP_SOFT_FLOAT,
     "Basic floating-point operations implemented in software"},
#endif
#if __OPENCL_VERSION__ >= 120
    {CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT,
     "Divide and sqrt are correctly rounded"},
#endif
    {CL_FP_DENORM, "Denorms"},
    {CL_FP_INF_NAN, "Inf and NaNs"},
    {CL_FP_ROUND_TO_NEAREST, "Round to nearest even rounding mode"},
    {CL_FP_ROUND_TO_ZERO, "Round to zero rounding mode"},
    {CL_FP_ROUND_TO_INF, "Round to +ve and -ve infinity rounding modes"},
    {CL_FP_FMA, "IEEE754-2008 fused multiply-add"}
  };
  for (size_t i = 0; i < LENGTH(list); ++i) {
    if (config & list[i].flag) {
      print(indent, key, list[i].name);
      key = NULL;
    }
  }
  if (key) {
    print(indent, key, "Not supported");
  }
}

void printQueueProperties(size_t indent, const char* key, void* value,
                          size_t size, Printer print) {
  const cl_command_queue_properties props =
      *((cl_command_queue_properties*)value);
  const struct {
    cl_command_queue_properties flag;
    const char* name;
  } list[] = {
      {CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, "Out of order execution"},
      {CL_QUEUE_PROFILING_ENABLE, "Profiling"}};
  for (size_t i = 0; i < LENGTH(list); ++i) {
    if (props & list[i].flag) {
      print(indent, key, list[i].name);
      key = NULL;
    }
  }
  if (key) {
    print(indent, key, "Not supported");
  }
}

void printLocalMemType(size_t indent, const char* key, void* value, size_t size,
                       Printer print) {
  switch (*((cl_device_local_mem_type*)value)) {
#if __OPENCL_VERSION__ >= 120
    case CL_NONE:
      print(indent, key, "None");
      break;
#endif
    case CL_LOCAL:
      print(indent, key, "Local");
      break;
    case CL_GLOBAL:
      print(indent, key, "Global");
      break;
    default:
      print(indent, key, "Unknown");
  }
}

void printGlobalMemCacheType(size_t indent, const char* key, void* value,
                             size_t size, Printer print) {
  switch (*((cl_device_mem_cache_type*)value)) {
    case CL_NONE:
      print(indent, key, "None");
      break;
    case CL_READ_ONLY_CACHE:
      print(indent, key, "Read only");
      break;
    case CL_READ_WRITE_CACHE:
      print(indent, key, "Read write");
      break;
    default:
      print(indent, key, "Unknown");
  }
}

void printExecutionCapabilities(size_t indent, const char* key, void* value,
                                size_t size, Printer print) {
  const cl_device_exec_capabilities props =
      *((cl_device_exec_capabilities*)value);
  const struct {
    cl_device_exec_capabilities flag;
    const char* name;
  } list[] = {{CL_EXEC_KERNEL, "OpenCL kernels"},
              {CL_EXEC_NATIVE_KERNEL, "Native kernels"}};
  for (size_t i = 0; i < LENGTH(list); ++i) {
    if (props & list[i].flag) {
      print(indent, key, list[i].name);
      key = NULL;
    }
  }
  if (key) {
    print(indent, key, "Not supported");
  }
}

#if __OPENCL_VERSION__ >= 120

void printKernels(size_t indent, const char* key, void* value, size_t size,
                  Printer print) {
  if (size > 0 && *((char*)value) != '\0') {
    char* item = strtok(value, ";");
    print(indent, key, item);
    while ((item = strtok(NULL, ";"))) {
      print(indent, NULL, item);
    }
  } else {
    print(indent, key, "None");
  }
}

void printParentDevice(size_t indent, const char* key, void* value, size_t size,
                       Printer print) {
  const cl_device_id id = size > 0 ? *((cl_device_id*)value) : NULL;
  if (id) {
    cl_platform_id platform;
    clGetDeviceInfo(id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform,
                    NULL);

    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

    cl_device_id devices[num_devices];
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

    for (size_t i = 0; i < num_devices; ++i) {
      if (devices[i] == id) {
        char buffer[(i > 0 ? lrint(log10(i)) + 1 : 1) + 2];
        sprintf(buffer, "#%zu", i);
        print(indent, key, buffer);
        return;
      }
    }
    print(indent, key, "Not found");
  } else {
    print(indent, key, "None");
  }
}

void printPartitionProperties(size_t indent, const char* key, void* value,
                              size_t size, Printer print) {
  const size_t ndims = size / sizeof(cl_device_partition_property);
  const cl_device_partition_property* props =
      *((cl_device_partition_property(*)[])value);
  for (size_t i = 0; i < ndims && props[i]; ++i) {
    switch (props[i]) {
      case CL_DEVICE_PARTITION_EQUALLY:
        print(indent, key, "Equally");
        break;
      case CL_DEVICE_PARTITION_BY_COUNTS:
        print(indent, key, "By counts");
        break;
      case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
        print(indent, key, "By affinity domain");
        break;
      default:
        print(indent, key, "Unknown partition property");
    }
    key = NULL;
  }
  if (key) {
    print(indent, key, "None");
  }
}

void printPartitionAffinityDomain(size_t indent, const char* key, void* value,
                                  size_t size, Printer print) {
  const cl_device_affinity_domain props = *((cl_device_affinity_domain*)value);
  const struct {
    cl_device_affinity_domain flag;
    const char* name;
  } list[] = {
      {CL_DEVICE_AFFINITY_DOMAIN_NUMA, "NUMA"},
      {CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE, "L4 cache"},
      {CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE, "L3 cache"},
      {CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE, "L2 cache"},
      {CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE, "L1 cache"},
      {CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE, "Next partitionable"}};
  for (size_t i = 0; i < LENGTH(list); ++i) {
    if (props & list[i].flag) {
      print(indent, key, list[i].name);
      key = NULL;
    }
  }
  if (key) {
    print(indent, key, "None");
  }
}

void printPartitionType(size_t indent, const char* key, void* value,
                        size_t size, Printer print) {
  const size_t ndims = size / sizeof(cl_device_partition_property);
  const cl_device_partition_property* props =
      *((cl_device_partition_property(*)[])value);
  switch (props[0]) {
    case CL_DEVICE_PARTITION_EQUALLY: {
      const cl_uint num = (cl_uint)props[1];
      char buffer[(num > 0 ? lrint(log10(num)) + 1 : 1) + 11];
      sprintf(buffer, "Equally (%u)", num);
      print(indent, key, buffer);
    } break;
    case CL_DEVICE_PARTITION_BY_COUNTS:
      print(indent, key, "By counts");
      for (size_t i = 1; i < ndims - 1; ++i) {
        const cl_uint num = (cl_uint)props[i];
        char buffer[(num > 0 ? lrint(log10(num)) + 1 : 1) + 1];
        sprintf(buffer, "%u", num);
        print(indent, NULL, buffer);
      }
      break;
    case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN: {
      const cl_device_affinity_domain dom = (cl_device_affinity_domain)props[1];
      const struct {
        cl_device_affinity_domain flag;
        const char* name;
      } list[] = {
          {CL_DEVICE_AFFINITY_DOMAIN_NUMA, "NUMA"},
          {CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE, "L4 cache"},
          {CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE, "L3 cache"},
          {CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE, "L2 cache"},
          {CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE, "L1 cache"},
          {CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE, "Next partitionable"}};
      char buffer[40] = "By affinity domain (Unknown)";
      for (size_t i = 0; i < LENGTH(list); ++i) {
        if (dom == list[i].flag) {
          sprintf(buffer, "By affinity domain (%s)", list[i].name);
          break;
        }
      }
      print(indent, key, buffer);
    } break;
    case 0:
      print(indent, key, "Not a subdevice");
      break;
    default:
      print(indent, key, "Unknown partition property");
  }
}

#endif
