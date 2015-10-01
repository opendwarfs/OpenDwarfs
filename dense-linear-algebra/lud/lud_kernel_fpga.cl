#define BLOCK_SIZE 16
	__kernel void 
lud_diagonal(__global float *m, int matrix_dim, int offset)
{
	int i,j;
	int array_offset = offset*matrix_dim+offset;

	for(i=0; i < BLOCK_SIZE; i++) {

		if (get_local_id(0)>i){
			for(j=0; j < i; j++)
				m[array_offset+get_local_id(0)*matrix_dim+i] -= m[array_offset+get_local_id(0)*matrix_dim+j]*m[array_offset+j*matrix_dim+i];
			m[array_offset+get_local_id(0)*matrix_dim+i] /= m[array_offset+i*matrix_dim+i];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
		if (get_local_id(0)>i){
			for(j=0; j < i+1; j++)
				m[array_offset+(i+1)*matrix_dim+get_local_id(0)] -= m[array_offset+(i+1)*matrix_dim+j]*m[array_offset+j*matrix_dim+get_local_id(0)];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
		
	}

}

	__kernel void
lud_perimeter(__global float *m, int matrix_dim, int offset)
{
	//__local float dia[BLOCK_SIZE][BLOCK_SIZE];
	//__local float peri_row[BLOCK_SIZE][BLOCK_SIZE];
	//__local float peri_col[BLOCK_SIZE][BLOCK_SIZE];

	int i,j, array_offset;
	//int array_offset_dia;
	int array_offset_peri;
	int idx;
	/*
	   if (get_local_id(0) < BLOCK_SIZE) {

	   idx = get_local_id(0);

	   array_offset = offset*matrix_dim+offset;
	   for (i=0; i < BLOCK_SIZE/2; i++){
	   dia[i][idx]=m[array_offset+idx];
	   array_offset += matrix_dim;
	   }

	   array_offset = offset*matrix_dim+offset;
	   for (i=0; i < BLOCK_SIZE; i++) {
	   peri_row[i][idx]=m[array_offset+(get_group_id(0)+1)*BLOCK_SIZE+idx];
	   array_offset += matrix_dim;
	   }

	   } else {

	   idx = get_local_id(0)-BLOCK_SIZE;

	   array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
	   for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
	   dia[i][idx]=m[array_offset+idx];
	   array_offset += matrix_dim;
	   }

	   array_offset = (offset+(get_group_id(0)+1)*BLOCK_SIZE)*matrix_dim+offset;
	   for (i=0; i < BLOCK_SIZE; i++) {
	   peri_col[i][idx] = m[array_offset+idx];
	   array_offset += matrix_dim;
	   }

	   }
	   barrier(CLK_LOCAL_MEM_FENCE);
	 */
	if (get_local_id(0) < BLOCK_SIZE) { //peri-row

		idx=get_local_id(0);

		array_offset = offset*matrix_dim+offset;

		for(i=1; i < BLOCK_SIZE; i++){
			for (j=0; j < i; j++)
				m[array_offset+i*matrix_dim+(get_group_id(0)+1)*BLOCK_SIZE+idx] -= m[array_offset+i*matrix_dim+j]*m[array_offset+j*matrix_dim+(get_group_id(0)+1)*BLOCK_SIZE+idx];
			//peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
		}

		//array_offset = (offset+1)*matrix_dim+offset;
		//for(i=1; i < BLOCK_SIZE; i++){
		//  m[array_offset+(get_group_id(0)+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
		//  array_offset += matrix_dim;
		//}
	} else { //peri-col

		idx=get_local_id(0) - BLOCK_SIZE;

		array_offset_peri = (offset+(get_group_id(0)+1)*BLOCK_SIZE)*matrix_dim+offset;
		//array_offset_dia = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
		array_offset = offset*matrix_dim+offset;

		for(i=0; i < BLOCK_SIZE; i++){
			for(j=0; j < i; j++)
				m[array_offset_peri+idx*matrix_dim+i] -= m[array_offset_peri+idx*matrix_dim+j] * m[array_offset+j*matrix_dim+i];
			//peri_col[idx][i]-=peri_col[idx][j]*m[array_offset+j*matrix_dim+i];//diko mou doulevei mono dia[] allagmeno
			//peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];//original
			m[array_offset_peri+idx*matrix_dim+i] /= m[array_offset+i*matrix_dim+i];
			//peri_col[idx][i] /= m[array_offset+i*matrix_dim+i];//diko mou doulevei mono dia[] allagmeno
			//peri_col[idx][i] /= dia[i][i];//original
		}

		//barrier(CLK_LOCAL_MEM_FENCE);

		//array_offset = (offset+(get_group_id(0)+1)*BLOCK_SIZE)*matrix_dim+offset;
		//for(i=0; i < BLOCK_SIZE; i++){
		//  m[array_offset+idx] =  peri_col[i][idx];
		//  array_offset += matrix_dim;
		//}
	}

}

	__kernel void
lud_internal(__global float *m, int matrix_dim, int offset)
{

	int i;
	float sum;

	int global_row_id = offset + (get_group_id(1)+1)*BLOCK_SIZE;
	int global_col_id = offset + (get_group_id(0)+1)*BLOCK_SIZE;

	sum = 0;
	for (i=0; i < BLOCK_SIZE; i++)
		sum += m[(global_row_id+get_local_id(1))*matrix_dim+offset+i] * m[(offset+i)*matrix_dim+global_col_id+get_local_id(0)];
	m[(global_row_id+get_local_id(1))*matrix_dim+global_col_id+get_local_id(0)] -= sum;


}

