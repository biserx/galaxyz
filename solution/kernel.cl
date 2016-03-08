
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
	__local unsigned int *lHistogramDD,
	__local unsigned int *lHistogramDR,
	__local unsigned int *lHistogramRR,
	const float degreefactor,
	const float costotaldegrees,
	const unsigned long totalX,
	const unsigned long totalY)
{	
	int i;
	
	int lx = get_local_id(0);
	int ly = get_local_id(1);
	
	if (lx == 0 && ly == 0)
	{
		for (i = 0; i <= 256; ++i)
		{
			lHistogramDD[i] = 0;
			lHistogramDR[i] = 0;
			lHistogramRR[i] = 0;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	
	int x = get_global_id(0);
	int y = get_global_id(1);

	int bin;
	float theta;
	
	if (x < totalX && y <= x)
	{ 
		theta = px[x] * px[y] + py[x] * py[y] + pz[x] * pz[y];
		if (theta >= costotaldegrees) 
		{
			if (theta > 1.0) theta = 1.0; 
			bin = (int)(acos(theta) * degreefactor); 
			atomic_add(&lHistogramDD[bin], 2);
		}
	}
	
	if (x < totalX && y < totalY)
	{
		theta = px[x] * qx[y] + py[x] * qy[y] + pz[x] * qz[y];
		if (theta >= costotaldegrees) 
		{
			if (theta > 1.0) theta = 1.0; 
			bin = (int)(acos(theta) * degreefactor); 
			atomic_inc(&lHistogramDR[bin]);
		}
	}
	
	if (x < totalY && y <= x)
	{
		theta = qx[x] * qx[y] + qy[x] * qy[y] + qz[x] * qz[y];
		if (theta >= costotaldegrees) 
		{
			if (theta > 1.0) theta = 1.0; 
			bin = (int)(acos(theta) * degreefactor); 
			atomic_add(&lHistogramRR[bin], 2);
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	if (lx == 0 && ly == 0)
	{
		for (i = 0; i <= 256; ++i)
		{
			atomic_add(&histogramDD[i], lHistogramDD[i]);
			atomic_add(&histogramDR[i], lHistogramDR[i]);
			atomic_add(&histogramRR[i], lHistogramRR[i]);
		}
	}
}
