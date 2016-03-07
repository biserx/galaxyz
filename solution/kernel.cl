
__kernel void galaxyz (
	__global float *px, 
	__global float *py, 
	__global float *pz, 
	__global float *qx, 
	__global float *qy, 
	__global float *qz, 
	__global unsigned int *histogramDD,
	__global unsigned int *histogramDR,
	__global unsigned int *histogramRR,
	const float degreefactor,
	const float costotaldegrees,
	const unsigned long totalX,
	const unsigned long totalY)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	int bin;
	float theta;
	
	if (x < totalX && y < totalX)
	{ 
		theta = px[x] * px[y] + py[x] * py[y] + pz[x] * pz[y];
		if (theta >= costotaldegrees) 
		{
			if (theta > 1.0) theta = 1.0; 
			bin = (int)(acos(theta) * degreefactor); 
			atomic_inc(&histogramDD[bin]); //histogram[bin]++;
		}
	}
	if (x < totalX && y < totalY)
	{
		theta = px[x] * qx[y] + py[x] * qy[y] + pz[x] * qz[y];
		if (theta >= costotaldegrees) 
		{
			if (theta > 1.0) theta = 1.0; 
			bin = (int)(acos(theta) * degreefactor); 
			atomic_inc(&histogramDR[bin]); //histogram[bin]++;
		}
	}
	if (x < totalY && y < totalY)
	{
		theta = qx[x] * qx[y] + qy[x] * qy[y] + qz[x] * qz[y];
		if (theta >= costotaldegrees) 
		{
			if (theta > 1.0) theta = 1.0; 
			bin = (int)(acos(theta) * degreefactor); 
			atomic_inc(&histogramRR[bin]); //histogram[bin]++;
		}
	}
}
