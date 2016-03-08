__kernel void galaxyz_1 (
	__global float *px, 
	__global float *py, 
	__global float *pz, 
	__global unsigned int *histogram,
	__local unsigned int *lHistogram,
	const unsigned int binsCount,
	const float degreefactor,
	const float costotaldegrees,
	const unsigned long total)
{	
	/*int i;
	
	int lx = get_local_id(0);
	int ly = get_local_id(1);
	
	if (lx == 0 && ly == 0)
	{
		for (i = 0; i < binsCount; ++i)
		{
			lHistogram[i] = 0;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);*/
	
	// Determine x, y relative to group ID
	size_t group_x_count = get_num_groups(0);
	//size_t group_y_count = get_num_groups(1); 
	size_t group_x = get_group_id(0);
	size_t group_y = get_group_id(1);
	
	size_t group_i = group_x + group_y * group_x_count;

	group_y = (size_t)((-1 + sqrt((float)8 * group_i + 1)) / 2);
	group_x = group_i - group_y * (group_y + 1) / 2;

	size_t group_x_size = get_local_size(0);
	size_t group_y_size = get_local_size(1);
	size_t local_x = get_local_id(0);
	size_t local_y = get_local_id(1);

	size_t x = group_x * group_x_size + local_x;
	size_t y = group_y * group_y_size + local_y;

	int bin;
	float theta;
	
	if (x < total && y < total && y >= x)
	{ 
		theta = px[x] * px[y] + py[x] * py[y] + pz[x] * pz[y];
		if (theta >= costotaldegrees) 
		{
			if (theta > 1.0) theta = 1.0; 
			bin = (int)(acos(theta) * degreefactor); 
			atomic_inc(&histogram[bin]);
		}
	}
	
	/*barrier(CLK_LOCAL_MEM_FENCE);

	if (lx == 0 && ly == 0)
	{
		for (i = 0; i < binsCount; ++i)
		{
			atomic_add(&histogram[i], lHistogram[i]);
		}
	}*/
}

__kernel void galaxyz_2 (
	__global float *px, 
	__global float *py, 
	__global float *pz, 
	__global float *qx, 
	__global float *qy, 
	__global float *qz, 
	__global unsigned int *histogram,
	__local unsigned int *lHistogram,
	const unsigned int binsCount,
	const float degreefactor,
	const float costotaldegrees,
	const unsigned long totalP,
	const unsigned long totalQ)
{	
	/*int i;
	
	int lx = get_local_id(0);
	int ly = get_local_id(1);
	
	if (lx == 0 && ly == 0)
	{
		for (i = 0; i < binsCount; ++i)
		{
			lHistogram[i] = 0;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);*/
	
	int x = get_global_id(0);
	int y = get_global_id(1);

	int bin;
	float theta;
	
	if (x < totalP && y < totalQ)
	{
		theta = px[x] * qx[y] + py[x] * qy[y] + pz[x] * qz[y];
		if (theta >= costotaldegrees) 
		{
			if (theta > 1.0) theta = 1.0; 
			bin = (int)(acos(theta) * degreefactor); 
			atomic_inc(&histogram[bin]);
		}
	}
	
	/*barrier(CLK_LOCAL_MEM_FENCE);

	if (lx == 0 && ly == 0)
	{
		for (i = 0; i < binsCount; ++i)
		{
			atomic_add(&histogram[i], lHistogram[i]);
		}
	}*/
}
