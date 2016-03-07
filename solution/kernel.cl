__kernel void galaxyz_single (
	__global float *px, 
	__global float *py, 
	__global float *pz, 
	__global unsigned int *histogram,
	const float degreefactor,
	const float costotaldegrees,
	const unsigned int count)
{
	int bin;
	float theta;

	int i = get_global_id(0);

	int y = (int)((-1 + sqrt((float)8 * i + 1)) / 2);
	int x = i - y * (y + 1) / 2;
    
	theta = px[x] * px[y] + py[x] * py[y] + pz[x] * pz[y];
	if (theta >= costotaldegrees) 
	{
		if ( theta > 1.0 ) theta = 1.0;
		bin = (int)(acos(theta) * degreefactor); 
		atomic_inc(&histogram[bin]);
	}
}

__kernel void galaxyz_multi (
	__global float *px, 
	__global float *py, 
	__global float *pz, 
	__global float *qx, 
	__global float *qy, 
	__global float *qz, 
	__global unsigned int *histogramPQ,
	const float degreefactor,
	const float costotaldegrees,
	const unsigned int count_p)
{
	int i = get_global_id(0);
	int x = i % count_p;
	int y = i / count_p; 
	
	float theta = px[x] * qx[y] + py[x] * qy[y] + pz[x] * qz[y];
	if (theta >= costotaldegrees) 
	{
		if ( theta > 1.0 ) theta = 1.0; 
		int bin = (int)(acos(theta) * degreefactor); 
		atomic_inc(&histogramPQ[bin]);
	}
}
