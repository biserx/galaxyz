
__kernel void galaxyz (
	__global float *px, 
	__global float *py, 
	__global float *pz, 
	__global float *qx, 
	__global float *qy, 
	__global float *qz, 
	__global unsigned int *histogram,
	const int binsperdegree,
	const float pi,
	const float costotaldegrees,
	const unsigned int count)
{
	int i = get_global_id(0);
	int x = i % count;
	int y = i / count; 

	//if(y < count)
	//{
		float theta;
		float degreefactor = 180.0/pi*binsperdegree;
		int bin;
		theta = px[x] * qx[y] + py[x] * qy[y] + pz[x] * qz[y];
		if (theta >= costotaldegrees) 
		{
			if ( theta > 1.0 )
				theta = 1.0; 
			bin = (int)(acos(theta) * degreefactor); 
			atomic_inc(&histogram[bin]); //histogram[bin]++;
		}
	//}

}
