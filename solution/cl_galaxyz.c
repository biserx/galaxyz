/*
	This is basically "combination" of examples I got at the beginning.
	It is initial version, so I wouldn't be surprised event if something don't work well.

	Changes made relative to original files
		- Function for loading kernel source from file
		- Function for loading input data (removed repetitive code)
		- Using malloc instead of calloc for x,y,z (no need for initialization)
		- Removed unnecessary histogram initialization to zero (calloc will do that)
		- Correction regarding measuring execution time (if I got things right in my head)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>

#include "CLErrorToStr.h"

const unsigned int binsperdegree = 4;		/* Nr of bins per degree */
#define totaldegrees 64							/* Nr of degrees */
#define KERNEL_FILE_NAME "kernel.cl"

/* Count how many lines the input file has */
int count_lines (FILE *infile) {
	char readline[80];      /* Buffer for file input */
	int lines=0;
	while( fgets(readline,80,infile) != NULL ) lines++;
		rewind(infile);  /* Reset the file to the beginning */
	return(lines);
}

/* Read input data from the file, convert to cartesian coordinates 
and write them to arrays x, y and z */
void read_data(FILE *infile, int n, float *x, float *y, float *z) {
	char readline[80];      /* Buffer for file input */
	float ra, dec, theta, phi, dpi;
	int i=0;
	dpi = acos(-1.0);
	while( fgets(readline,80,infile) != NULL )  /* Read a line */
	{
		sscanf(readline,"%f %f",&ra, &dec);  /* Read a coordinate pair */
		/* Debug */
		/*if ( i == 0 ) printf("     first item: %3.6f %3.6f\n",ra,dec); */
		/* Convert to cartesian coordinates */
		phi   = ra * dpi/180.0;
		theta = (90.0-dec)*dpi/180;
		x[i] = sinf(theta)*cosf(phi);
		y[i] = sinf(theta)*sinf(phi);
		z[i] = cosf(theta);
		/*
		NOTE: there was a bug in the code here. The following ststement
		is not correct.
		z[i] = sinf(cosf(theta));
		*/
		++i;
	}
	/* Debug */
	/* printf("      last item: %3.6f %3.6f\n\n",ra,dec); */
}


int load_kernel_from_file(const char fileName[], char** source_str, size_t *source_size)
{	
	FILE *fp;
	// Load kernel source code
	fp = fopen(fileName, "r");
	if (!fp)
		return -1;
	fseek(fp, 0L, SEEK_END);
	*source_size = ftell(fp);
	fseek(fp, 0L, SEEK_SET);

	*source_str = (char *)malloc(*source_size * sizeof(char));
	fread(*source_str, 1, *source_size, fp);
	fclose(fp);
}


int load_input_data(const char fileName[], int *number_of_lines, float **x, float **y, float **z)
{

	FILE *fp;
	/* Open the real data input file */
	fp = fopen(fileName, "r");
	if (!fp) {
		printf("Unable to open %s\n", fileName);
		return -1;
	}

	/* Count how many lines the input file has */
	*number_of_lines = count_lines(fp);
	printf("%s contains %d lines\n", fileName, *number_of_lines);

	/* Allocate arrays for x, y and z values */
	*x = (float *)malloc(*number_of_lines * sizeof(float));
	*y = (float *)malloc(*number_of_lines * sizeof(float));
	*z = (float *)malloc(*number_of_lines * sizeof(float)); 

	/* Read the file with real input data */
	read_data(fp, *number_of_lines, *x, *y, *z);

	fclose(fp);
	return 0;
}

int main(int argc, char *argv[])
{
	/* Check that we have 4 command line arguments */
	if ( argc != 4 ) {
		printf("Usage: %s real_data sim_data output_file\n", argv[0]);
		return(0);
	}
	
	time_t wc_starttime;
	time_t starttime, stoptime;
	wc_starttime = time(NULL);
	starttime = clock();  // Start measuring time (use MPI_Wtime in the parallel program)

	int number_of_lines_real;	/* Nr of lines in real data */
	int number_of_lines_sim;	/* Nr of lines in random data */
	float *xd_real, *yd_real, *zd_real;	/* Arrays for real data */
	float *xd_sim , *yd_sim , *zd_sim;	/* Arrays for random data */

	int err;
	err = load_input_data(argv[1], &number_of_lines_real, &xd_real, &yd_real, &zd_real);
	err |= load_input_data(argv[2], &number_of_lines_sim, &xd_sim, &yd_sim, &zd_sim);
	if (err < 0)
	{
		printf("Loading input data failed!\n");
		return -1;
	}

	int nr_of_bins = binsperdegree * totaldegrees;  /* Total number of bins */
	unsigned int *histogramDD, *histogramDR, *histogramRR; /* Arrays for histograms */
	
	/* Allocate arrays for the histograms */
	histogramDD = (unsigned int *)calloc(nr_of_bins+1, sizeof(unsigned int));
	histogramDR = (unsigned int *)calloc(nr_of_bins+1, sizeof(unsigned int));
	histogramRR = (unsigned int *)calloc(nr_of_bins+1, sizeof(unsigned int));

	float pi, costotaldegrees;
	pi = acosf(-1.0);
	costotaldegrees = (float)(cos(totaldegrees/180.0*pi));

	// =================================================================
	// OpenCL code starts here

	cl_device_id device_id; // compute device id
	cl_context context; // compute context
	cl_command_queue commands; // compute command queue
	cl_program program; // compute program
	cl_kernel kernel; // compute kernel

	cl_platform_id platform;
	unsigned int no_plat;

	// Get available platforms (this example uses the first available platform.
	// It may be a problem if the first platform does not include device we are looking for)
	err = clGetPlatformIDs(1, &platform, &no_plat);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }
	
	// Obtain the list of GPU devices available on the platform
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }
	
	// Create OpenCL context for the GPU device
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }
	if (!context) return -1;

	// This command is deprecated in OpenCL 2.0. 
	// clCreateCommandQueue() should be replaced with clCreateCommandQueueProperties(), and without declaring CL_USE_DEPRECATED_OPENCL_2_0_APIS a warning is raised
	commands = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }
	if (!commands) return -1;
	
	// Load kernel source from file
	char *kernel_source;
	size_t kernel_source_size;
	char kernel_file_name[] = KERNEL_FILE_NAME;	
	err = load_kernel_from_file(kernel_file_name, &kernel_source, &kernel_source_size);
	if (err < 0) { printf("Kernel loading from file failed!\n"); return -1; }
	
	// Create program object for the context that stores the source code specified by the kernel_source text string
	program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, &kernel_source_size, &err);
	free(kernel_source);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }
	if (!program) return -1;
	
	// Compile and link the kernel program
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		char *log = (char*) malloc(len * sizeof(char)); //or whatever you use
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, log, NULL);
		printf("Build log:\n%s\n", log);
		free(log);
	}
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }

	// Create the OpenCL kernel  
	kernel = clCreateKernel(program, "galaxyz", &err);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }	
	if (!kernel) return -1;
	
	cl_mem cl_xd_real, cl_yd_real, cl_zd_real;
	cl_mem cl_xd_sim, cl_yd_sim, cl_zd_sim;
	cl_mem cl_histogram_DD, cl_histogram_DR, cl_histogram_RR;

	// Create device memory buffer for the input data
	cl_xd_real = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * number_of_lines_real, xd_real, NULL);
	cl_yd_real = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * number_of_lines_real, yd_real, NULL);
	cl_zd_real = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * number_of_lines_real, zd_real, NULL);

	cl_xd_sim  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * number_of_lines_sim, xd_sim, NULL);
	cl_yd_sim  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * number_of_lines_sim, yd_sim, NULL);
	cl_zd_sim  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * number_of_lines_sim, zd_sim, NULL);

	// Create device memory buffer for the output data
	cl_histogram_DD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int) * (nr_of_bins+1), histogramDD, NULL);
	cl_histogram_DR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int) * (nr_of_bins+1), histogramDR, NULL);
	cl_histogram_RR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int) * (nr_of_bins+1), histogramRR, NULL);

	if (!cl_xd_real || !cl_yd_real || !cl_zd_real ||
		 !cl_xd_sim || !cl_yd_sim || !cl_zd_sim ||
		 !cl_histogram_DD || !cl_histogram_DR || !cl_histogram_RR)
	{
		printf("Buffer allocation failed!\n");
		return -1;
	} 
	
	int count;
	size_t local;
	size_t global;

	// ===================================================
	// 				Calculating DD
	// ===================================================

	count = number_of_lines_real;
	// Set the kernel arguments
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_xd_real);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_yd_real);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_zd_real);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_xd_real);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_yd_real);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_zd_real);
	err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_histogram_DD);
	err |= clSetKernelArg(kernel, 7, sizeof(int), &binsperdegree);
	err |= clSetKernelArg(kernel, 8, sizeof(float), &pi);
	err |= clSetKernelArg(kernel, 9, sizeof(float), &costotaldegrees);
	err |= clSetKernelArg(kernel, 10, sizeof(unsigned int), &count);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }
	
	local = 10;
	global = count * count; // count;
	// Execute the OpenCL kernel in data parallel
	err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }
	
	// Wait for the commands to get executed before reading back the results 
	clFinish(commands);

	// Read kernel results to the host memory buffer
	err = clEnqueueReadBuffer(commands, cl_histogram_DD, CL_TRUE, 0, sizeof(unsigned int) * (nr_of_bins + 1), histogramDD, 0, NULL, NULL);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }

	// ===================================================
	// 				Calculating DR
	// ===================================================

	count = number_of_lines_real;
	// Set the kernel arguments
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_xd_real);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_yd_real);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_zd_real);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_xd_sim);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_yd_sim);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_zd_sim);
	err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_histogram_DR);
	err |= clSetKernelArg(kernel, 7, sizeof(int), &binsperdegree);
	err |= clSetKernelArg(kernel, 8, sizeof(float), &pi);
	err |= clSetKernelArg(kernel, 9, sizeof(float), &costotaldegrees);
	err |= clSetKernelArg(kernel, 10, sizeof(unsigned int), &count);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }
	
	local = 10;
	global = count * count; // count;
	// Execute the OpenCL kernel in data parallel
	err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }
	
	// Wait for the commands to get executed before reading back the results 
	clFinish(commands);

	// Read kernel results to the host memory buffer
	err = clEnqueueReadBuffer(commands, cl_histogram_DR, CL_TRUE, 0, sizeof(unsigned int) * (nr_of_bins + 1), histogramDR, 0, NULL, NULL);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }

	// ===================================================
	// 				Calculating RR
	// ===================================================

	count = number_of_lines_sim;
	// Set the kernel arguments
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_xd_sim);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_yd_sim);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_zd_sim);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_xd_sim);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_yd_sim);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_zd_sim);
	err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_histogram_RR);
	err |= clSetKernelArg(kernel, 7, sizeof(int), &binsperdegree);
	err |= clSetKernelArg(kernel, 8, sizeof(float), &pi);
	err |= clSetKernelArg(kernel, 9, sizeof(float), &costotaldegrees);
	err |= clSetKernelArg(kernel, 10, sizeof(unsigned int), &count);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }
	
	local = 10;
	global = count * count; // count;
	// Execute the OpenCL kernel in data parallel
	err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }
	
	// Wait for the commands to get executed before reading back the results 
	clFinish(commands);

	// Read kernel results to the host memory buffer
	err = clEnqueueReadBuffer(commands, cl_histogram_RR, CL_TRUE, 0, sizeof(unsigned int) * (nr_of_bins + 1), histogramRR, 0, NULL, NULL);
	if (err != CL_SUCCESS) { printf("Error no: %d say: %s\n", err, getErrorString(err)); return -1; }

	// ===================================================
	// 				End Calculations
	// ===================================================

	clReleaseMemObject(cl_xd_real);
	clReleaseMemObject(cl_yd_real);
	clReleaseMemObject(cl_zd_real);
	clReleaseMemObject(cl_xd_sim);
	clReleaseMemObject(cl_yd_sim);
	clReleaseMemObject(cl_zd_sim);

	clReleaseMemObject(cl_histogram_DD);
	clReleaseMemObject(cl_histogram_DR);
	clReleaseMemObject(cl_histogram_RR);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	// OpenCL code ends here
	// =================================================================

	int i,j;

	long TotalCountDD, TotalCountDR, TotalCountRR; /* Counters */
	
	printf("Calculating DD angle histogram...\n");
	/* Multiply DD histogram with 2 since we only calculate (i,j) pair, not (j,i) */
	/*	for ( i = 0; i <= nr_of_bins; ++i ) 
		histogramDD[i] *= 2L;
	histogramDD[0] += ((long)(number_of_lines_real));*/
		
	/* Count the total nr of values in the DD histograms */
	TotalCountDD = 0L;
	for ( i = 0; i <= nr_of_bins; ++i ) 
		TotalCountDD += histogramDD[i]; 
	printf("  histogram count = %ld\n\n", TotalCountDD);
	
	printf("Calculating DR angle histogram...\n");
	/* Count the total nr of values in the DR histograms */
	TotalCountDR = 0L;
	for ( i = 0; i <= nr_of_bins; ++i ) 
		TotalCountDR += histogramDR[i]; 
	printf("DR angle                         histogram count = %ld\n\n", TotalCountDR);

	/* Multiply RR histogram with 2 since we only calculate (i,j) pair, not (j,i) */
	/*for ( i = 0; i <= nr_of_bins; ++i ) 
		histogramRR[i] *= 2L; 
	histogramRR[0] += ((long)(number_of_lines_sim));*/
	printf("Calculating RR angle histogram...\n");
	/* Count the total nr of values in the RR histograms */
	TotalCountRR = 0L;
	for ( i = 0; i <= nr_of_bins; ++i ) 
		TotalCountRR += histogramRR[i]; 
	printf("  histogram count = %ld\n\n", TotalCountRR);

	free(xd_real);
	free(yd_real);
	free(zd_real);
	free(xd_sim);
	free(yd_sim);
	free(zd_sim);

	printf("\n\n");

	FILE *outfile;
	/* Open the output file */
	outfile = fopen(argv[3],"w");
	if ( outfile == NULL ) {
		printf("Unable to open %s\n",argv[3]);
		return -1;
	}
	/* Write the histograms both to display and outfile */
	//printf("bin center\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
	fprintf(outfile,"bin center\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
	double NSimdivNReal, w;
	for ( i = 0; i < nr_of_bins; ++i ) 
	{
		NSimdivNReal = ((double)(number_of_lines_sim))/((double)(number_of_lines_real));
		w = 1.0 + NSimdivNReal*NSimdivNReal*histogramDD[i]/histogramRR[i]-2.0*NSimdivNReal*histogramDR[i]/((double)(histogramRR[i]));
		//printf(" %6.3f      %3.6f\t%15u\t%15u\t%15u\n",((float)i+0.5)/binsperdegree, w, histogramDD[i], histogramDR[i], histogramRR[i]);
		fprintf(outfile,"%6.3f\t%15lf\t%15u\t%15u\t%15u\n",((float)i+0.5)/binsperdegree, w, histogramDD[i], histogramDR[i], histogramRR[i]);
	}
	fclose(outfile);

	free(histogramDD);
	free(histogramDR);
	free(histogramRR);

	printf("\nCPU time = %6.2f seconds\n", ((double) (clock()-starttime))/ CLOCKS_PER_SEC);
	printf("\nWall clock time = %6.2f\n", (double)(time(NULL) - wc_starttime));

	return 0;
}