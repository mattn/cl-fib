#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <CL/opencl.h>

int
main(int argc, char *argv[]) {
  if (argc != 2) {
	fprintf(stderr, "usage of %s: [number]\n", argv[0]);
	return 1;
  }
  int n = atoi(argv[1]);

  char* source_data = \
					  "float _fib(float n) {                  \n" \
					  "  if(n <= 0) return 0;                 \n" \
					  "  if(n > 0 && n < 3) return 1;         \n" \
					  "  float r = 0, n1 = 1, n2 = 1;         \n" \
					  "  for (int i = 2; i < n; i++) {        \n" \
					  "    r = n1 + n2;                       \n" \
					  "    n1 = n2;                           \n" \
					  "    n2 = r;                            \n" \
					  "  }                                    \n" \
					  "  return r;                            \n" \
					  "}                                      \n" \
					  "                                       \n" \
					  "kernel void fib(                       \n" \
					  "    global const float *a,             \n" \
					  "    global float *r) {                 \n" \
					  "  unsigned int id = get_global_id(0);  \n" \
					  "  r[id] = _fib(a[id]);                 \n" \
					  "}                                      \n" \
					  "";
  size_t source_size = strlen(source_data);

  cl_platform_id platform_id = NULL;
  cl_int ret;
  cl_uint num_devices, num_platforms;

  ret = clGetPlatformIDs(
	  1,
	  &platform_id,
	  &num_platforms);
  if (ret != CL_SUCCESS) {
	fprintf(stderr, "cannot get platform IDs: %d\n", ret);
	return 1;
  }

  cl_device_id device_id = NULL;
  ret = clGetDeviceIDs(
	  platform_id,
	  CL_DEVICE_TYPE_DEFAULT,
	  1,
	  &device_id,
	  &num_devices);
  if (ret != CL_SUCCESS) {
	fprintf(stderr, "cannot get device ID: %d\n", ret);
	return 1;
  }

  cl_context context = clCreateContext(
	  NULL,
	  1,
	  &device_id,
	  NULL,
	  NULL,
	  &ret);
  if (context == NULL) {
	fprintf(stderr, "cannot create context\n");
	return 1;
  }

  cl_command_queue command_queue = clCreateCommandQueue(
	  context,
	  device_id,
	  0,
	  &ret);
  if (command_queue == NULL) {
	fprintf(stderr, "cannot create command: %d\n", ret);
	return 1;
  }

  cl_program program = clCreateProgramWithSource(
	  context,
	  1,
	  (const char **) &source_data,
	  (const size_t *) &source_size,
	  &ret);
  if (program == NULL) {
	fprintf(stderr, "cannot create program\n");
	return 1;
  }

  ret = clBuildProgram(
	  program,
	  1,
	  &device_id,
	  NULL,
	  NULL,
	  NULL);
  if (ret != CL_SUCCESS) {
	size_t log_size;
	ret = clGetProgramBuildInfo(
		program,
		device_id,
		CL_PROGRAM_BUILD_LOG,
		0,
		NULL,
		&log_size);
	char *log_data = (char* )malloc(log_size + 1);
	ret = clGetProgramBuildInfo(
		program,
		device_id,
		CL_PROGRAM_BUILD_LOG,
		log_size,
		log_data,
		NULL);
	log_data[log_size] = 0;
	fprintf(stderr, "%s\n", log_data);
	free(log_data);
	return 1;
  }

  cl_kernel kernel = clCreateKernel(
	  program,
	  "fib",
	  &ret);
  if (program == NULL) {
	fprintf(stderr, "cannot create kernel: %d\n", ret);
	return 1;
  }

  float a[1] = { (float)n };
  float r[1] = { 0 };

  cl_mem ma = clCreateBuffer(
	  context,
	  CL_MEM_READ_WRITE,
	  sizeof(a),
	  NULL,
	  &ret);
  if (ma == NULL) {
	fprintf(stderr, "cannot create buffer: %d\n", ret);
	return 1;
  }
  cl_mem mr = clCreateBuffer(
	  context,
	  CL_MEM_READ_WRITE,
	  sizeof(r),
	  NULL,
	  &ret);
  if (mr == NULL) {
	fprintf(stderr, "cannot create buffer: %d\n", ret);
	return 1;
  }

  ret = clEnqueueWriteBuffer(
	  command_queue,
	  ma,
	  CL_TRUE,
	  0,
	  sizeof(a),
	  a,
	  0,
	  NULL,
	  NULL);
  if (ret != CL_SUCCESS) {
	fprintf(stderr, "cannot write buffer: %d\n", ret);
	return 1;
  }
  ret = clEnqueueWriteBuffer(
	  command_queue,
	  mr,
	  CL_TRUE,
	  0,
	  sizeof(r),
	  r,
	  0,
	  NULL,
	  NULL);
  if (ret != CL_SUCCESS) {
	fprintf(stderr, "cannot write buffer: %d\n", ret);
	return 1;
  }

  ret = clSetKernelArg(
	  kernel,
	  0,
	  sizeof(ma),
	  (void *)&ma);
  if (ret != CL_SUCCESS) {
	fprintf(stderr, "cannot set argument: %d\n", ret);
	return 1;
  }
  ret = clSetKernelArg(
	  kernel,
	  1,
	  sizeof(mr),
	  (void *)&mr);
  if (ret != CL_SUCCESS) {
	fprintf(stderr, "cannot set argument: %d\n", ret);
	return 1;
  }

  size_t work[2] = { 1, 0 };
  ret = clEnqueueNDRangeKernel(
	  command_queue,
	  kernel,
	  1,
	  NULL,
	  work,
	  work,
	  0,
	  NULL,
	  NULL);
  if (ret != CL_SUCCESS) {
	fprintf(stderr, "cannot enqueue task: %d\n", ret);
	return 1;
  }
  ret = clEnqueueReadBuffer(
	  command_queue,
	  mr,
	  CL_TRUE,
	  0,
	  sizeof(r),
	  r,
	  0,
	  NULL,
	  NULL);

  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  ret = clReleaseMemObject(ma);
  ret = clReleaseMemObject(mr);

  printf("%ld\n", (long)r[0]);
  return 0;
}
