/****************************************************************************
 * GEM -- electrostatics calculations and visualization                     *
 * Copyright (C) 2006  John C. Gordon                                       *
 *                                                                          *
 * This program is free software; you can redistribute it and/or modify     *
 * it under the terms of the GNU General Public License as published by     *
 * the Free Software Foundation; either version 2 of the License, or        *
 * (at your option) any later version.                                      *
 *                                                                          *
 * This program is distributed in the hope that it will be useful,          *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
 * GNU General Public License for more details.                             *
 *                                                                          *
 * You should have received a copy of the GNU General Public License along  *
 * with this program; if not, write to the Free Software Foundation, Inc.,  *
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.              *
 ****************************************************************************/

#include "calculations.h"
#include "defines.h"
#undef min
#undef max
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/time.h>
#include <malloc.h>

#include "../../include/rdtsc.h"
#include "../../include/common_args.h"
#define AOCL_ALIGNMENT 64

#ifdef __FPGA__
    #include "cl_ext.h"
#else
    #define CL_MEM_BANK_1_ALTERA              (0)
    #define CL_MEM_BANK_2_ALTERA              (0)
    #define CL_MEM_BANK_3_ALTERA              (0)
    #define CL_MEM_BANK_4_ALTERA              (0)
    #define CL_MEM_BANK_5_ALTERA              (0)
    #define CL_MEM_BANK_6_ALTERA              (0)
    #define CL_MEM_BANK_7_ALTERA              (0)
#endif


typedef long long int64;
static struct timeval start_time;

void init_time() {
	gettimeofday(&start_time, NULL);
}

int64 get_time() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return (int64) (t.tv_sec - start_time.tv_sec) * 1000000
		+ (t.tv_usec - start_time.tv_usec);
}
// #define BLOCK_DIM_X 8
// #define BLOCK_DIM_Y 8
int  BLOCK_DIM_X=16;
int  BLOCK_DIM_Y=16;
//nthreads/block 64
//#define BLOCKS 390
#define BLOCKS 120
//multiple of 30

cl_program program;

/* This program uses only one kernel and this serves as a handle to it */
cl_kernel  kernel;

/*
 * Converts the contents of a file into a string
 */
	std::string
convertToString(const char *filename)
{
	size_t size;
	char*  str;
	std::string s;

	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if(f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = f.tellg();
		f.seekg(0, std::fstream::beg);

		str = new char[size+1];
		if(!str)
		{
			f.close();
			return NULL;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';

		s = str;

		return s;
	}
	return NULL;
}


/* Calculates potential for surface out of a molecule -- this is the free
 * space solution. --JCG */
void calc_potential_single_step(residue *residues,
		int nres,
		vertx *vert,
		int nvert,
		float A,
		float proj_len,
		float diel_int,
		float diel_ext,
		float sal,
		float ion_exc_rad,
		int phiType,
		int *i,
		int step_size)
{
	int it, eye, bound;
	//residue *residues_s;
	float *res_c, *res_x, *res_y, *res_z,
	      *at_c, *at_x, *at_y, *at_z,
	      *vert_c, *vert_x, *vert_y, *vert_z,
	      *vert_x_p, *vert_y_p, *vert_z_p;
	cl_mem res_c_s, res_x_s, res_y_s, res_z_s,
	       at_c_s, at_x_s, at_y_s, at_z_s,
	       vert_c_s, vert_x_s, vert_y_s, vert_z_s,
	       vert_x_p_s, vert_y_p_s, vert_z_p_s;
	cl_mem atom_addrs_s;
	cl_mem atom_lengths_s;
	int natoms;
	unsigned int *atom_addrs;
	unsigned int *atom_lengths;

	cl_int status = 0;
	cl_int errcode;
	ocd_initCL();

	/////////////////////////////////////////////////////////////////
	// Load CL file, build CL program object, create CL kernel object
	/////////////////////////////////////////////////////////////////
/*	const char * filename  = "calculate_potential.cl";
	std::string  sourceStr = convertToString(filename);
	const char * source    = sourceStr.c_str();
	size_t sourceSize[]
= { strlen(source) };

	program = clCreateProgramWithSource(
			context,
			1,
			&source,
			sourceSize,
			&status);
	if(status != CL_SUCCESS)
	{
		std::cout<<
			"Error: Loading Binary into cl_program \
			(clCreateProgramWithBinary)\n";
		exit(-1);
	}

*/

	/* create a cl program executable for all the devices specified */
	//status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	char* kernel_files;
	int num_kernels = 1;
    std::string kernel_file_name;
	kernel_files = (char*) malloc(sizeof(char*)*num_kernels);
    kernel_file_name = "calculate_potential";


    kernel_files = new char[kernel_file_name.length()+1];
    strcpy(kernel_files,kernel_file_name.c_str());

    cl_program program = ocdBuildProgramFromFile(context,device_id,kernel_files, NULL);

	if(status != CL_SUCCESS)
	{
		std::cout<<"Error: Building Program (clBuildProgram):"<<status<<std::endl;
		char * errorbuf = (char*)calloc(sizeof(char),1024*1024);
		size_t size;
		clGetProgramBuildInfo(program,
				device_id,
				CL_PROGRAM_BUILD_LOG,
				1024*1024,
				errorbuf,
				&size);

		std::cout << errorbuf << std::endl;
		exit(-1);
	}

	/* get a kernel object handle for a kernel with the given name */
	kernel = clCreateKernel(program, "calc_potential_single_step_dev", &status);
	if(status != CL_SUCCESS)
	{
		std::cout<<"Error: Creating Kernel from program. (clCreateKernel)\n";
		exit(-1);
	}


	init_time();

	eye = *i;

	bound = eye + step_size;

	if (nvert < bound) bound = nvert;

	natoms = 0;
	for (it = 0; it < nres; it++)
	{
		natoms += residues[it].natoms;
	}

	cl_int err[17];
	    res_c = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * nres);
	    res_x = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * nres);
	    res_y = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * nres);
	    res_z = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * nres);
	    at_c = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * natoms);
	    at_x = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * natoms);
	    at_y = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * natoms);
	    at_z = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * natoms);
	    vert_c = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * nvert);
	    vert_x = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * nvert);
	    vert_y = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * nvert);
	    vert_z = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * nvert);
	    vert_x_p = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * nvert);
	    vert_y_p = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * nvert);
	    vert_z_p = (float *)  memalign ( AOCL_ALIGNMENT,sizeof(float) * nvert);
	    /* allocate temporary arrays for atoms and start addresses */
	    atom_addrs = (unsigned int *) memalign ( AOCL_ALIGNMENT,nres * sizeof(unsigned int));
	    atom_lengths = (unsigned int *) memalign ( AOCL_ALIGNMENT,nres * sizeof(unsigned int));

        //res_c = (float *) malloc(sizeof(float) * nres);
	    //res_x = (float *) malloc(sizeof(float) * nres);
	    //res_y = (float *) malloc(sizeof(float) * nres);
	    //res_z = (float *) malloc(sizeof(float) * nres);
	    //at_c = (float *) malloc(sizeof(float) * natoms);
	    //at_x = (float *) malloc(sizeof(float) * natoms);
	    //at_y = (float *) malloc(sizeof(float) * natoms);
	    //at_z = (float *) malloc(sizeof(float) * natoms);
	    //vert_c = (float *) malloc(sizeof(float) * nvert);
	    //vert_x = (float *) malloc(sizeof(float) * nvert);
	    //vert_y = (float *) malloc(sizeof(float) * nvert);
	    //vert_z = (float *) malloc(sizeof(float) * nvert);
	    //vert_x_p = (float *) malloc(sizeof(float) * nvert);
	    //vert_y_p = (float *) malloc(sizeof(float) * nvert);
	    //vert_z_p = (float *) malloc(sizeof(float) * nvert);
	    //
	    //atom_addrs = (unsigned int *)malloc(nres * sizeof(unsigned int));
	    //atom_lengths = (unsigned int *)malloc(nres * sizeof(unsigned int));


	/* copy the atom structure to the device */
	natoms = 0;
	for (it = 0; it < nres; it++)
	{
		atom_addrs[it] = natoms;

		res_c[it] = residues[it].total_charge;
		res_x[it] = residues[it].x;
		res_y[it] = residues[it].y;
		res_z[it] = residues[it].z;

		for(int j=0; j<residues[it].natoms; j++)
		{
			at_c[natoms+j] = residues[it].atoms[j].charge;
			at_x[natoms+j] = residues[it].atoms[j].x;
			at_y[natoms+j] = residues[it].atoms[j].y;
			at_z[natoms+j] = residues[it].atoms[j].z;
		}
		natoms += residues[it].natoms;
		atom_lengths[it] = residues[it].natoms;
	}

	// free(res_c);
	// free(res_x);
	// free(res_y);
	// free(res_z);
	// free(at_c );
	// free(at_x );
	// free(at_y );
	// free(at_z );


	for (it = 0; it < nvert; it++)
	{
		vert_c[it] = 0;
		/* offset point for calculation of d */
		vert_x[it] = vert[it].x + proj_len * vert[it].xNorm;
		vert_y[it] = vert[it].y + proj_len * vert[it].yNorm;
		vert_z[it] = vert[it].z + proj_len * vert[it].zNorm;

		/* offset point for calculation of d' only */
		vert_x_p[it] = vert[it].x + ion_exc_rad * vert[it].xNorm;
		vert_y_p[it] = vert[it].y + ion_exc_rad * vert[it].yNorm;
		vert_z_p[it] = vert[it].z + ion_exc_rad * vert[it].zNorm;
	}


	//TODO: fix this
	//free(vert_x);
	//free(vert_y);
	//free(vert_z);
	//free(vert_x_p);
	//free(vert_y_p);
	//free(vert_z_p);

	/* free temporary arrays for atoms and start addresses */
	// free(atom_addrs);
	// free(atom_lengths);


	float r, r0, rprime;
	r = A + proj_len;
	rprime = A + ion_exc_rad;


	/* need to estimate r0, dist from atom to surface */
	/* if we intend to support negative projection lengths <><> */
	r0 = A/2.;  /* might be tricky ... hrmmm */
/*
	res_c_s         = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(cl_float)*nres,    NULL, &err[0]);
	res_x_s         = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*nres,    NULL, &err[1]);
	res_y_s         = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*nres,    NULL, &err[2]);
	res_z_s         = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*nres,    NULL, &err[3]);
	at_c_s          = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*natoms,  NULL, &err[4]);
	at_x_s          = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*natoms,  NULL, &err[5]);
	at_y_s          = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*natoms,  NULL, &err[6]);
	at_z_s          = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*natoms,  NULL, &err[7]);
	vert_c_s        = clCreateBuffer( context, CL_MEM_READ_WRITE, sizeof(cl_float)*nvert,   NULL, &err[8]);
	vert_x_s        = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*nvert,   NULL, &err[9]);
	vert_y_s        = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*nvert,   NULL, &err[10]);
	vert_z_s        = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*nvert,   NULL, &err[11]);
	vert_x_p_s      = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*nvert,   NULL, &err[12]);
	vert_y_p_s      = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*nvert,   NULL, &err[13]);
	vert_z_p_s      = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_float)*nvert,   NULL, &err[14]);
	atom_addrs_s    = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_int)*nres,      NULL, &err[15]);
	atom_lengths_s  = clCreateBuffer( context, CL_MEM_READ_ONLY , sizeof(cl_int)*nres,      NULL, &err[16]);
*/

	res_c_s         = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_1_ALTERA, sizeof(cl_float)*nres,    NULL, &err[0]);
	res_x_s         = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_2_ALTERA , sizeof(cl_float)*nres,    NULL, &err[1]);
	res_y_s         = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_1_ALTERA , sizeof(cl_float)*nres,    NULL, &err[2]);
	res_z_s         = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_2_ALTERA , sizeof(cl_float)*nres,    NULL, &err[3]);
	at_c_s          = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_1_ALTERA , sizeof(cl_float)*natoms,  NULL, &err[4]);
	at_x_s          = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_2_ALTERA , sizeof(cl_float)*natoms,  NULL, &err[5]);
	at_y_s          = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_1_ALTERA , sizeof(cl_float)*natoms,  NULL, &err[6]);
	at_z_s          = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_2_ALTERA , sizeof(cl_float)*natoms,  NULL, &err[7]);
	vert_c_s        = clCreateBuffer( context, CL_MEM_READ_WRITE|CL_MEM_BANK_1_ALTERA, sizeof(cl_float)*nvert,   NULL, &err[8]);
	vert_x_s        = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_2_ALTERA , sizeof(cl_float)*nvert,   NULL, &err[9]);
	vert_y_s        = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_1_ALTERA , sizeof(cl_float)*nvert,   NULL, &err[10]);
	vert_z_s        = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_2_ALTERA , sizeof(cl_float)*nvert,   NULL, &err[11]);
	vert_x_p_s      = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_1_ALTERA , sizeof(cl_float)*nvert,   NULL, &err[12]);
	vert_y_p_s      = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_2_ALTERA , sizeof(cl_float)*nvert,   NULL, &err[13]);
	vert_z_p_s      = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_1_ALTERA , sizeof(cl_float)*nvert,   NULL, &err[14]);
	atom_addrs_s    = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_2_ALTERA , sizeof(cl_int)*nres,      NULL, &err[15]);
	atom_lengths_s  = clCreateBuffer( context, CL_MEM_READ_ONLY|CL_MEM_BANK_1_ALTERA , sizeof(cl_int)*nres,      NULL, &err[16]);


	clEnqueueWriteBuffer ( commands, res_c_s       , CL_TRUE, 0, sizeof(cl_float)*nres,   res_c,        0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Res Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, res_x_s       , CL_TRUE, 0, sizeof(cl_float)*nres,   res_x,        0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Res Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, res_y_s       , CL_TRUE, 0, sizeof(cl_float)*nres,   res_y,        0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Res Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, res_z_s       , CL_TRUE, 0, sizeof(cl_float)*nres,   res_z,        0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Res Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, at_c_s        , CL_TRUE, 0, sizeof(cl_float)*natoms, at_c,         0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Atom Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, at_x_s        , CL_TRUE, 0, sizeof(cl_float)*natoms, at_x,         0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Atom Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, at_y_s        , CL_TRUE, 0, sizeof(cl_float)*natoms, at_y,         0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Atom Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, at_z_s        , CL_TRUE, 0, sizeof(cl_float)*natoms, at_z,         0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Atom Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, vert_c_s      , CL_TRUE, 0, sizeof(cl_float)*nvert,  vert_c,       0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Vertex Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, vert_x_s      , CL_TRUE, 0, sizeof(cl_float)*nvert,  vert_x,       0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Vertex Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, vert_y_s      , CL_TRUE, 0, sizeof(cl_float)*nvert,  vert_y,       0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Vertex Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, vert_z_s      , CL_TRUE, 0, sizeof(cl_float)*nvert,  vert_z,       0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Vertex Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, vert_x_p_s    , CL_TRUE, 0, sizeof(cl_float)*nvert,  vert_x_p,     0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Vertex Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, vert_y_p_s    , CL_TRUE, 0, sizeof(cl_float)*nvert,  vert_y_p,     0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Vertex Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, vert_z_p_s    , CL_TRUE, 0, sizeof(cl_float)*nvert,  vert_z_p,     0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Vertex Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, atom_addrs_s  , CL_TRUE, 0, sizeof(cl_int)*nres,     atom_addrs,   0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Address Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	clEnqueueWriteBuffer ( commands, atom_lengths_s, CL_TRUE, 0, sizeof(cl_int)*nres,     atom_lengths, 0, NULL, &ocdTempEvent);
	clFinish(commands);
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Length Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	// clSetKernelArg( kernel, 0, sizeof(cl_mem), (void *)&outputBuffer);
	clSetKernelArg( kernel, 0, sizeof(cl_int), (void *)&    nres);// int nres,
	clSetKernelArg( kernel, 1, sizeof(cl_int), (void *)&    nvert);// int nvert,
	clSetKernelArg( kernel, 2, sizeof(cl_float), (void *)&  A);// float A,
	clSetKernelArg( kernel, 3, sizeof(cl_float), (void *)&  proj_len);// float proj_len,
	clSetKernelArg( kernel, 4, sizeof(cl_float), (void *)&  diel_int);// float diel_int,
	clSetKernelArg( kernel, 5, sizeof(cl_float), (void *)&  diel_ext);// float diel_ext,
	clSetKernelArg( kernel, 6, sizeof(cl_float), (void *)&  sal);// float sal,
	clSetKernelArg( kernel, 7, sizeof(cl_float), (void *)&  ion_exc_rad);// float ion_exc_rad,
	clSetKernelArg( kernel, 8, sizeof(cl_int), (void *)&    phiType);// int phiType,
	clSetKernelArg( kernel, 10, sizeof(cl_int), (void *)&    step_size);// int step_size,
	clSetKernelArg( kernel, 11, sizeof(cl_mem), (void *)&    atom_addrs_s);// __global unsigned int * atom_addrs,
	clSetKernelArg( kernel, 12, sizeof(cl_mem), (void *)&    atom_lengths_s);// __global unsigned int * atom_lengths,
	clSetKernelArg( kernel, 13, sizeof(cl_float), (void *)&  r);// float r,
	clSetKernelArg( kernel, 14, sizeof(cl_float), (void *)&  r0);// float r0,
	clSetKernelArg( kernel, 15, sizeof(cl_float), (void *)&  rprime);// float rprime,
	clSetKernelArg( kernel, 16, sizeof(cl_mem), (void *)&    res_c_s   );//__global float *res_c_s
	clSetKernelArg( kernel, 17, sizeof(cl_mem), (void *)&    res_x_s   );//__global float *res_x_s
	clSetKernelArg( kernel, 18, sizeof(cl_mem), (void *)&    res_y_s   );//__global float *res_y_s
	clSetKernelArg( kernel, 19, sizeof(cl_mem), (void *)&    res_z_s  );//__global float *res_z_s,
	clSetKernelArg( kernel, 20, sizeof(cl_mem), (void *)&    at_c_s    );//__global float *at_c_s
	clSetKernelArg( kernel, 21, sizeof(cl_mem), (void *)&    at_x_s    );//__global float *at_x_s
	clSetKernelArg( kernel, 22, sizeof(cl_mem), (void *)&    at_y_s    );//__global float *at_y_s
	clSetKernelArg( kernel, 23, sizeof(cl_mem), (void *)&    at_z_s   );//__global float *at_z_s,
	clSetKernelArg( kernel, 24, sizeof(cl_mem), (void *)&    vert_c_s  );//__global float *vert_c_s
	clSetKernelArg( kernel, 25, sizeof(cl_mem), (void *)&    vert_x_s  );//__global float *vert_x_s
	clSetKernelArg( kernel, 26, sizeof(cl_mem), (void *)&    vert_y_s  );//__global float *vert_y_s
	clSetKernelArg( kernel, 27, sizeof(cl_mem), (void *)&    vert_z_s );//__global float *vert_z_s,
	clSetKernelArg( kernel, 28, sizeof(cl_mem), (void *)&    vert_x_p_s);//__global float *vert_x_p_s
	clSetKernelArg( kernel, 29, sizeof(cl_mem), (void *)&    vert_y_p_s);//__global float *vert_y_p_s
	clSetKernelArg( kernel, 30, sizeof(cl_mem), (void *)&    vert_z_p_s);//__global float *vert_z_p_s)

	size_t maxWorkItemSizes[3];
	clGetDeviceInfo(
			device_id,
			CL_DEVICE_MAX_WORK_ITEM_SIZES,
			sizeof(size_t)*3,
			(void*)maxWorkItemSizes,
			NULL);

	while(BLOCK_DIM_X*BLOCK_DIM_X>maxWorkItemSizes[0])
		BLOCK_DIM_X = BLOCK_DIM_X/2;
	BLOCK_DIM_Y = BLOCK_DIM_X;


	//cl_int   status;
	cl_uint maxDims;
	size_t globalThreads[2] = { BLOCK_DIM_X * BLOCK_DIM_Y, BLOCKS };
	size_t localThreads[2]  = { BLOCK_DIM_X * BLOCK_DIM_Y, 1 };
	size_t maxWorkGroupSize;


	clGetDeviceInfo(
			device_id,
			CL_DEVICE_MAX_WORK_GROUP_SIZE,
			sizeof(size_t),
			(void*)&maxWorkGroupSize,
			NULL);


	clGetDeviceInfo(
			device_id,
			CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
			sizeof(cl_uint),
			(void*)&maxDims,
			NULL);


	if(globalThreads[0] > maxWorkGroupSize ||
			localThreads[0] > maxWorkItemSizes[0])
	{
		std::cout<<"Unsupported: Device does not support requested number of work items."<<std::endl;
		std::cout<<"workgroup:"<< maxWorkGroupSize<<std::endl;
		std::cout<<"workitem :"<< maxWorkItemSizes[0]<<std::endl;
		return;
	}


	for (;  eye/*+((BLOCK_DIM_X*BLOCK_DIM_Y)*BLOCKS)*/ < bound; eye+=((BLOCK_DIM_X*BLOCK_DIM_Y)*BLOCKS))
		//for(it=0; it<3; it++)
	{
		clSetKernelArg( kernel, 9, sizeof(cl_int), (void *)&    eye);// int eye,
		fprintf(stdout,"finished first %d of %d\n", eye, nvert);
		/* copy the vert data of the current block to device */
		status = clEnqueueNDRangeKernel(
				commands,
				kernel,
				2,
				NULL,
				globalThreads,
				localThreads,
				0,
				NULL,
				&ocdTempEvent);
		if(status != CL_SUCCESS)
		{
			std::cout<<
				"Error: Enqueueing kernel onto command queue. \
				(clEnqueueNDRangeKernel): "<< status << std::endl;
					return;
		}

		/* wait for the kernel call to finish execution */
		status = clWaitForEvents(1, &ocdTempEvent);
		START_TIMER(ocdTempEvent, OCD_TIMER_KERNEL, "GEM Kernel", ocdTempTimer)
		END_TIMER(ocdTempTimer)
		if(status != CL_SUCCESS)
		{
				std::cout<<
					"Error: Waiting for kernel run to finish. \
					(clWaitForEvents)\n";
				return;
		}

		if(eye + step_size > bound)
			step_size = bound - eye;


		/* copy back the calculation result */
	}

	if(eye > bound)
		eye = bound;
	status = clEnqueueReadBuffer(
			commands,
			vert_c_s,
			CL_TRUE,
			0,
			nvert * sizeof(cl_float),
			vert_c,
			0,
			NULL,
			&ocdTempEvent);
	clFinish(commands);
	if(status != CL_SUCCESS)
	{
		std::cout <<
			"Error: clEnqueueReadBuffer failed. \
			(clEnqueueReadBuffer)\n";

		return ;
	}

	/* Wait for the read buffer to finish execution */
	status = clWaitForEvents(1, &ocdTempEvent);
	START_TIMER(ocdTempEvent, OCD_TIMER_D2H, "Vertex Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)
	if(status != CL_SUCCESS)
	{
			std::cout<<
				"Error: Waiting for read buffer call to finish. \
				(clWaitForEvents)\n";
			return ;
	}

	clFinish(commands);
	printf("runtime:%llu\n", get_time());


	for(it = 0; it < nvert; it++)
	{
		vert[it].potential = vert_c[it];
	}

	*i = eye;
}

