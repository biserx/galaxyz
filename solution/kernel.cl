__kernel void galaxyz_1 (
	__global float *px, 
	__global float *py, 
	__global float *pz, 
	__global unsigned int *histogram,
	__local float *lpx,
	__local float *lpy,
	__local float *lpz,
	__local float *lqx,
	__local float *lqy,
	__local float *lqz,
	__local unsigned int *lHistogram,
	const unsigned int binsCount,
	const float degreefactor,
	const float costotaldegrees,
	const unsigned long total,
	const unsigned int coverage)
{	
	size_t i;

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

	size_t wi_x = group_x * group_x_size + local_x;
	size_t wi_y = group_y * group_y_size + local_y;

	size_t x_offset = wi_x * coverage;
	size_t y_offset = wi_y * coverage;

	// First work item in each work group is chaarged to copy
	// data from global to local memory, and to initialize memory
	// used for histogram
	if (local_x == 0 && local_y == 0)
	{
		for (i = 0; i < binsCount; ++i)
		{
			lHistogram[i] = 0;
		}

		size_t memory_chunk_size = coverage * group_x_size;
		for (i = 0; i < memory_chunk_size; ++i)
		{
			size_t p_offset = x_offset + i;
			size_t q_offset = y_offset + i;
			if (p_offset < total)
			{
				lpx[i] = px[p_offset];
				lpy[i] = py[p_offset];
				lpz[i] = pz[p_offset];
			}
			if (q_offset < total)
			{
				lqx[i] = px[q_offset];
				lqy[i] = py[q_offset];
				lqz[i] = pz[q_offset];
			}
		}
	}

	// Wait for the first work item in the group to finish
	// copying data to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform calculations
	size_t totalCoverage = coverage * coverage;
	for (i = 0; i < totalCoverage; i++)
	{
		int x = i % coverage;
		int y = i / coverage;
		int gX = x_offset + x;
		int gY = y_offset + y;
		x += local_x * coverage;
		y += local_y * coverage;
		if (gX < total && gY < total && gY >= gX)
		{
			float theta = lpx[x] * lqx[y] + lpy[x] * lqy[y] + lpz[x] * lqz[y];
			if (theta >= costotaldegrees)
			{
				if (theta > 1.0) theta = 1.0;
				int bin = (int)(acos(theta) * degreefactor);
				atomic_inc(&lHistogram[bin]);
			}
		}
	}
	
	// Wait for all workitems to finish calculations
	// So first work item can copy results back to the main memory
	barrier(CLK_LOCAL_MEM_FENCE);

	if (local_x == 0 && local_y == 0)
	{
		for (i = 0; i < binsCount; ++i)
		{
			atomic_add(&histogram[i], lHistogram[i]);
		}
	}
}

__kernel void galaxyz_2 (
	__global float *px, 
	__global float *py, 
	__global float *pz, 
	__global float *qx, 
	__global float *qy, 
	__global float *qz, 
	__global unsigned int *histogram,
	__local float *lpx,
	__local float *lpy,
	__local float *lpz,
	__local float *lqx,
	__local float *lqy,
	__local float *lqz,
	__local unsigned int *lHistogram,
	const unsigned int binsCount,
	const float degreefactor,
	const float costotaldegrees,
	const unsigned long totalP,
	const unsigned long totalQ,
	const unsigned int coverage)
{	
	size_t i; 
	
	size_t local_x = get_local_id(0);
	size_t local_y = get_local_id(1);

	size_t group_x_size = get_local_size(0);
	//size_t group_y_size = get_local_size(1);

	size_t group_x = get_group_id(0);
	size_t group_y = get_group_id(1);

	int wi_x = get_global_id(0);
	int wi_y = get_global_id(1);

	size_t x_offset = wi_x * coverage;
	size_t y_offset = wi_y * coverage;
	
	if (local_x == 0 && local_y == 0)
	{
		for (i = 0; i < binsCount; ++i)
		{
			lHistogram[i] = 0;
		}
		size_t memory_chunk_size = coverage * group_x_size;
		for (i = 0; i < memory_chunk_size; ++i)
		{
			size_t p_offset = x_offset + i;
			size_t q_offset = y_offset + i;
			if (p_offset < totalP)
			{
				lpx[i] = px[p_offset];
				lpy[i] = py[p_offset];
				lpz[i] = pz[p_offset];
			}
			if (q_offset < totalQ)
			{
				lqx[i] = qx[q_offset];
				lqy[i] = qy[q_offset];
				lqz[i] = qz[q_offset];
			}
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	size_t totalCoverage = coverage * coverage;
	for (i = 0; i < totalCoverage; ++i)
	{
		int x = i % coverage;
		int y = i / coverage;
		int gX = x_offset + x;
		int gY = y_offset + y;
		x += local_x * coverage;
		y += local_y * coverage;
		if (gX < totalP && gY < totalQ)
		{
			float theta = lpx[x] * lqx[y] + lpy[x] * lqy[y] + lpz[x] * lqz[y];
			if (theta >= costotaldegrees)
			{
				if (theta > 1.0) theta = 1.0;
				int bin = (int)(acos(theta) * degreefactor);
				atomic_inc(&lHistogram[bin]);
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (local_x == 0 && local_y == 0)
	{
		for (i = 0; i < binsCount; ++i)
		{
			atomic_add(&histogram[i], lHistogram[i]);
		}
	}
}
