/*
	Compile with gcc -O3 cl_galaxyz.c -o cl_galaxyz -lm -lOpenCL
	Run sequentially with srun -n 1 galaxyz small.txt small_rand.txt outfile.txt
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>

#include "CLErrorToStr.h"

#define KERNEL_FILE_NAME "kernel.cl"

void make_factors(size_t in, size_t *a, size_t *b)
{
	int x = 2;
	*a = 1;
	*b = 1;
	while(1)
	{
		while(in % x == 0)
		{
			if (x > *a * *b) { *a = *a * *b; *b = 1; }
			if (*a > *b) *b *= x; else *a *= x;
			in /= x;
		}
		++x;
		if (x > in) break;
	}
}

void get_global_sizes(size_t local, size_t global[], unsigned int matrix_size)
{
	size_t a, b;
	size_t tmp = matrix_size / local + (matrix_size % local > 0);
	tmp = tmp * (tmp + 1) / 2;
	make_factors(tmp, &a, &b);
	global[0] = a * local;
	global[1] = b * local;
}

long max(long a, long b)
{
	return a>b?a:b;
}

/* Count how many lines the input file has */
int count_lines (FILE *infile) 
{
	char readline[80];	/* Buffer for file input */
	int lines=0;
	while( fgets(readline,80,infile) != NULL ) lines++;
		rewind(infile);	/* Reset the file to the beginning */
	return(lines);
}

/* Read input data from the file, convert to cartesian coordinates 
and write them to arrays x, y and z */
void read_data(FILE *infile, int n, float *x, float *y, float *z) 
{
	char readline[80];	/* Buffer for file input */
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
	*source_size = fread(*source_str, 1, *source_size, fp);
	fclose(fp);
}


int load_input_data(const char fileName[], long *number_of_lines, float **x, float **y, float **z)
{
	FILE *fp;
	/* Open the real data input file */
	fp = fopen(fileName, "r");
	if (!fp) 
	{
		printf("Unable to open %s\n", fileName);
		return -1;
	}

	/* Count how many lines the input file has */
	*number_of_lines = count_lines(fp);
	printf("%s contains %lu lines\n", fileName, *number_of_lines);

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
	if ( argc != 4 ) 
	{
		printf("Usage: %s real_data sim_data output_file\n", argv[0]);
		return(0);
	}

	unsigned int default_work_group_size = 10;
	unsigned int binsperdegree = 4;	/* Nr of bins per degree */
	unsigned int totaldegrees = 64;	/* Nr of degrees */
	unsigned int nr_of_bins = binsperdegree * totaldegrees + 1;  /* Total number of bins */

	time_t wc_starttime;
	time_t starttime, stoptime;
	wc_starttime = time(NULL);
	starttime = clock();  // Start measuring time (use MPI_Wtime in the parallel program)

	unsigned long number_of_lines_real;	/* Nr of lines in real data */
	unsigned long number_of_lines_sim;	/* Nr of lines in random data */
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

	unsigned int *histogramDD, *histogramDR, *histogramRR; /* Arrays for histograms */
	
	/* Allocate arrays for the histograms */
	histogramDD = (unsigned int *)calloc(nr_of_bins, sizeof(unsigned int));
	histogramDR = (unsigned int *)calloc(nr_of_bins, sizeof(unsigned int));
	histogramRR = (unsigned int *)calloc(nr_of_bins, sizeof(unsigned int));

	float pi, costotaldegrees;
	pi = acosf(-1.0);
	costotaldegrees = (float)(cos(totaldegrees/180.0*pi));

	// =================================================================
	// OpenCL code starts here

	cl_device_id device_id; // compute device id
	cl_context context; // compute context
	cl_command_queue commands; // compute command queue
	cl_program program; // compute program
	cl_kernel kernel_1, kernel_2; // compute kernel

	cl_platform_id platform;
	unsigned int no_plat;

	// Get available platforms (this example uses the first available platform.
	// It may be a problem if the first platform does not include device we are looking for)
	err = clGetPlatformIDs(1, &platform, &no_plat);
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }
	
	// Obtain the list of GPU devices available on the platform
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }
	
	// Create OpenCL context for the GPU device
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }
	if (!context) return -1;

	// This command is deprecated in OpenCL 2.0. 
	// clCreateCommandQueue() should be replaced with clCreateCommandQueueProperties(), and without declaring CL_USE_DEPRECATED_OPENCL_2_0_APIS a warning is raised
	commands = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }
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
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }
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
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }

	// Create the OpenCL kernel  
	kernel_1 = clCreateKernel(program, "galaxyz_1", &err);
	kernel_2 = clCreateKernel(program, "galaxyz_2", &err);
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }	
	if (!kernel_1 || !kernel_2) return -1;
	
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
	cl_histogram_DD = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int) * nr_of_bins, histogramDD, NULL);
	cl_histogram_DR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int) * nr_of_bins, histogramDR, NULL);
	cl_histogram_RR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int) * nr_of_bins, histogramRR, NULL);

	if (!cl_xd_real || !cl_yd_real || !cl_zd_real ||
		 !cl_xd_sim || !cl_yd_sim || !cl_zd_sim ||
		 !cl_histogram_DD || !cl_histogram_DR || !cl_histogram_RR)
	{
		printf("Buffer allocation failed!\n");
		return -1;
	} 
	
	// Calculating wrok-groups sizes and total work-items count
	size_t local_DD[2];
	size_t local_DR[2];
	size_t local_RR[2];
	size_t global_DD[2];
	size_t global_DR[2];
	size_t global_RR[2];

	local_DD[0] = default_work_group_size;
	local_DD[1] = local_DD[0];
	get_global_sizes(local_DD[0], global_DD, number_of_lines_real);

	printf("DD:\n Total work-groups: %zu\n Work-group size: %zux%zu\n Total work-items: %zu\n", 
			global_DD[0] * global_DD[1] / local_DD[0] / local_DD[1],
			local_DD[0], local_DD[1],
			global_DD[0] * global_DD[1]);

	local_DR[0] = default_work_group_size;
	local_DR[1] = default_work_group_size;
	global_DR[0] = number_of_lines_real;
	global_DR[1] = number_of_lines_sim;
	printf("DR:\n Total work-groups: %zu\n Work-group size: %zux%zu\n Total work-items: %zu\n", 
			global_DR[0] * global_DR[1] / local_DR[0] / local_DR[1],
			local_DR[0], local_DR[1],
			global_DR[0] * global_DR[1]);

	local_RR[0] = default_work_group_size;
	local_RR[1] = local_RR[0];
	get_global_sizes(local_RR[0], global_RR, number_of_lines_sim);
	printf("RR:\n Total work-groups: %zu\n Work-group size: %zux%zu\n Total work-items: %zu\n", 
			global_RR[0] * global_RR[1] / local_RR[0] / local_RR[1],
			local_RR[0], local_RR[1],
			global_RR[0] * global_RR[1]);

	float degreefactor = 180.0 / pi * binsperdegree;

	// ===================================================
	// 			Calculating DD, DR, RR
	// ===================================================

	// Set the kernel arguments DD
	err  = clSetKernelArg(kernel_1, 0, sizeof(cl_mem), &cl_xd_real);
	err |= clSetKernelArg(kernel_1, 1, sizeof(cl_mem), &cl_yd_real);
	err |= clSetKernelArg(kernel_1, 2, sizeof(cl_mem), &cl_zd_real);
	err |= clSetKernelArg(kernel_1, 3, sizeof(cl_mem), &cl_histogram_DD); // Histogram global
	err |= clSetKernelArg(kernel_1, 4, sizeof(unsigned int) * nr_of_bins, NULL); // Histogram local memory
	err |= clSetKernelArg(kernel_1, 5, sizeof(unsigned int), &nr_of_bins); // Number of bins in histogram
	err |= clSetKernelArg(kernel_1, 6, sizeof(float), &degreefactor);
	err |= clSetKernelArg(kernel_1, 7, sizeof(float), &costotaldegrees);
	err |= clSetKernelArg(kernel_1, 8, sizeof(unsigned long), &number_of_lines_real);
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }

	// Execute the OpenCL kernel in data parallel
	err = clEnqueueNDRangeKernel(commands, kernel_1, 2, NULL, global_DD, local_DD, 0, NULL, NULL);
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }

	// Set the kernel arguments RR
	err  = clSetKernelArg(kernel_1, 0, sizeof(cl_mem), &cl_xd_sim);
	err |= clSetKernelArg(kernel_1, 1, sizeof(cl_mem), &cl_yd_sim);
	err |= clSetKernelArg(kernel_1, 2, sizeof(cl_mem), &cl_zd_sim);
	err |= clSetKernelArg(kernel_1, 3, sizeof(cl_mem), &cl_histogram_RR); // Histogram global
	err |= clSetKernelArg(kernel_1, 4, sizeof(unsigned int) * nr_of_bins, NULL); // Histogram local memory
	err |= clSetKernelArg(kernel_1, 5, sizeof(unsigned int), &nr_of_bins); // Number of bins in histogram
	err |= clSetKernelArg(kernel_1, 6, sizeof(float), &degreefactor);
	err |= clSetKernelArg(kernel_1, 7, sizeof(float), &costotaldegrees);
	err |= clSetKernelArg(kernel_1, 8, sizeof(unsigned long), &number_of_lines_sim);
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }

	// Execute the OpenCL kernel in data parallel
	err = clEnqueueNDRangeKernel(commands, kernel_1, 2, NULL, global_RR, local_RR, 0, NULL, NULL);
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }
	
	// Set the kernel arguments DR
	err  = clSetKernelArg(kernel_2, 0, sizeof(cl_mem), &cl_xd_real);
	err |= clSetKernelArg(kernel_2, 1, sizeof(cl_mem), &cl_yd_real);
	err |= clSetKernelArg(kernel_2, 2, sizeof(cl_mem), &cl_zd_real);
	err |= clSetKernelArg(kernel_2, 3, sizeof(cl_mem), &cl_xd_sim);
	err |= clSetKernelArg(kernel_2, 4, sizeof(cl_mem), &cl_yd_sim);
	err |= clSetKernelArg(kernel_2, 5, sizeof(cl_mem), &cl_zd_sim);
	err |= clSetKernelArg(kernel_2, 6, sizeof(cl_mem), &cl_histogram_DR); // Histogram global
	err |= clSetKernelArg(kernel_2, 7, sizeof(unsigned int) * nr_of_bins, NULL); // Histogram local memory
	err |= clSetKernelArg(kernel_2, 8, sizeof(unsigned int), &nr_of_bins); // Number of bins in histogram
	err |= clSetKernelArg(kernel_2, 9, sizeof(float), &degreefactor);
	err |= clSetKernelArg(kernel_2, 10, sizeof(float), &costotaldegrees);
	err |= clSetKernelArg(kernel_2, 11, sizeof(unsigned long), &number_of_lines_real);
	err |= clSetKernelArg(kernel_2, 12, sizeof(unsigned long), &number_of_lines_sim);
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }

	// Execute the OpenCL kernel in data parallel
	err = clEnqueueNDRangeKernel(commands, kernel_2, 2, NULL, global_DR, local_DR, 0, NULL, NULL);
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }

	// Wait for the commands to get executed before reading back the results 
	clFinish(commands);

	// Read kernel results to the host memory buffer
	err  = clEnqueueReadBuffer(commands, cl_histogram_DD, CL_TRUE, 0, sizeof(unsigned int) * nr_of_bins, histogramDD, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(commands, cl_histogram_DR, CL_TRUE, 0, sizeof(unsigned int) * nr_of_bins, histogramDR, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(commands, cl_histogram_RR, CL_TRUE, 0, sizeof(unsigned int) * nr_of_bins, histogramRR, 0, NULL, NULL);
	if (err != CL_SUCCESS) { printf("Error (%d) no: %d say: %s\n", __LINE__, err, getErrorString(err)); return -1; }

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
	clReleaseKernel(kernel_1);
	clReleaseKernel(kernel_2);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	// OpenCL code ends here
	// =================================================================

	int i,j;

	long TotalCountDD, TotalCountDR, TotalCountRR; /* Counters */
	
	printf("Calculating DD angle histogram...\n");
	/* Multiply DD histogram with 2 since we only calculate (i,j) pair, not (j,i) */
	for ( i = 0; i < nr_of_bins; ++i ) 
		histogramDD[i] *= 2L;
	/*histogramDD[0] += ((long)(number_of_lines_real));*/
		
	/* Count the total nr of values in the DD histograms */
	TotalCountDD = 0L;
	for ( i = 0; i < nr_of_bins; ++i ) 
		TotalCountDD += histogramDD[i]; 
	printf("\thistogram count = %ld\n", TotalCountDD);
	
	printf("Calculating DR angle histogram...\n");
	/* Count the total nr of values in the DR histograms */
	TotalCountDR = 0L;
	for ( i = 0; i < nr_of_bins; ++i ) 
		TotalCountDR += histogramDR[i]; 
	printf("\thistogram count = %ld\n", TotalCountDR);

	/* Multiply RR histogram with 2 since we only calculate (i,j) pair, not (j,i) */
	for ( i = 0; i < nr_of_bins; ++i ) 
		histogramRR[i] *= 2L; 
	/*histogramRR[0] += ((long)(number_of_lines_sim));*/
	printf("Calculating RR angle histogram...\n");
	/* Count the total nr of values in the RR histograms */
	TotalCountRR = 0L;
	for ( i = 0; i < nr_of_bins; ++i ) 
		TotalCountRR += histogramRR[i]; 
	printf("\thistogram count = %ld\n", TotalCountRR);

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
	if ( outfile == NULL ) 
	{
		printf("Unable to open %s\n",argv[3]);
		return -1;
	}
	/* Write the histograms both to display and outfile */
	//printf("bin center\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
	fprintf(outfile,"bin center\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
	double NSimdivNReal, w;
	for ( i = 0; i < nr_of_bins - 1; ++i ) 
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

	printf("CPU time = %6.2f seconds\n", ((double) (clock()-starttime))/ CLOCKS_PER_SEC);
	printf("Wall clock time = %6.2f\n", (double)(time(NULL) - wc_starttime));

	return 0;
}
