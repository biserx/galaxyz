__kernel void galaxyz_1 (
	__global float *px, 
	__global float *py, 
	__global float *pz, 
	__global unsigned int *histogram,
	__local unsigned int *lHistogram,
	const unsigned int binsCount,
	const float degreefactor,
	const float costotaldegrees,
	const unsigned long total,
	const unsigned int coverage)
{	
	size_t i;

	size_t local_x = get_local_id(0);
	size_t local_y = get_local_id(1);

	/*
	if (local_x == 0 && local_y == 0)
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

	size_t group_x_size = get_local_size(0);
	size_t group_y_size = get_local_size(1);

	size_t local_wi_i = local_x + local_y * group_x_size;

	// This might be calculated otherwise
	size_t global_wi_i = group_i * group_x_size * group_y_size + local_wi_i;

	size_t wi_y = (size_t)((-1 + sqrt((float)8 * global_wi_i + 1)) / 2);
	size_t wi_x = global_wi_i - wi_y * (wi_y + 1) / 2;
	
	size_t totalCoverage = coverage * coverage;
	for (i = 0; i < totalCoverage; i++)
	{
		int x = wi_x * coverage + i % coverage;
		int y = wi_y * coverage + i / coverage;
		if (x < total && y < total && y >= x)
		{
			float theta = px[x] * px[y] + py[x] * py[y] + pz[x] * pz[y];
			if (theta >= costotaldegrees)
			{
				if (theta > 1.0) theta = 1.0;
				int bin = (int)(acos(theta) * degreefactor);
				atomic_inc(&histogram[bin]);
			}
		}
	}
	
	/*barrier(CLK_LOCAL_MEM_FENCE);

	if (local_x == 0 && local_y == 0)
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
	const unsigned long totalQ,
	const unsigned int coverage)
{	
	size_t i; 
	/*
	size_t local_x = get_local_id(0);
	size_t local_y = get_local_id(1);

	
	if (local_x == 0 && local_y == 0)
	{
		for (i = 0; i < binsCount; ++i)
		{
			lHistogram[i] = 0;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);*/
	
	int wi_x = get_global_id(0);
	int wi_y = get_global_id(1);
	
	size_t totalCoverage = coverage * coverage;
	for (i = 0; i < totalCoverage; i++)
	{
		int x = wi_x * coverage + i % coverage;
		int y = wi_y * coverage + i / coverage;
		if (x < totalP && y < totalQ)
		{
			float theta = px[x] * qx[y] + py[x] * qy[y] + pz[x] * qz[y];
			if (theta >= costotaldegrees)
			{
				if (theta > 1.0) theta = 1.0;
				int bin = (int)(acos(theta) * degreefactor);
				atomic_inc(&histogram[bin]);
			}
		}
	}

	/*barrier(CLK_LOCAL_MEM_FENCE);

	if (local_x == 0 && local_y == 0)
	{
		for (i = 0; i < binsCount; ++i)
		{
			atomic_add(&histogram[i], lHistogram[i]);
		}
	}*/
}
