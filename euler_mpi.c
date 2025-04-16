#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include "euler.h"
#define N 128
#define M 2000
#define SUBREGION_LEN 32
#define GATHER_BLK_SIZE 4096

int main(int argc, char* argv[])
{
    printf("#### Linearized euler equation for computational fluid dynamics using MPI and Lapacke ####\n");
    int scatter_blk_size = SUBREGION_LEN * SUBREGION_LEN; // 32 * 32   
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size); // should be 4
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // measuring efficiency
    double precision = MPI_Wtick();
    double starttime = MPI_Wtime();
    const double delta_x = 2./N;
    const double gamma = 1.4;
    const double initial_pt = -1.;
    const double delta_t = 0.001;
    const double d = delta_t / (delta_x * 4);
    int step_count = 0;
     // every process responsible for the subregion grid
    double **x_local; // 32 x 512
    double x_t[512][32];
    double r[128][128], u[128][128], v[128][128], p[128][128];
    x_local = (double**) malloc(sizeof(double*)*32);    
    /* Variables for gaussian Elimination */
    lapack_int lda = 512;
    lapack_int ipiv[512];
    lapack_int ldb = 512;	
    lapack_int info = 0;
    lapack_int n = 512;
    lapack_int nrhs = 1;
    double temp_matrix[512][512];

    /* Variables for scatter communication */
    double r_t_local_1[32][32], r_t_local_2[32][32], r_t_local_3[32][32], r_t_local_4[32][32];
    double u_t_local_1[32][32], u_t_local_2[32][32], u_t_local_3[32][32], u_t_local_4[32][32];
    double v_t_local_1[32][32], v_t_local_2[32][32], v_t_local_3[32][32], v_t_local_4[32][32];
    double p_t_local_1[32][32], p_t_local_2[32][32], p_t_local_3[32][32], p_t_local_4[32][32];	
    // these local array can then be organized into a 32x512 array

    // Initialization process
    for(int i = 0; i < 32; i++)
    {
	x_local[i] = (double *) malloc(sizeof(double)*512);
    }

    for(int j = 0; j < 32; j++)
    {
	// corresponding to every region for each process
	double y_coord = initial_pt + (SUBREGION_LEN*rank + j)*delta_x;
	
	for(int k = 0; k < 128; k++)
	{
	    double x_coord = initial_pt + k*delta_x;
	    double value = 2*exp(-100*(x_coord*x_coord + y_coord*y_coord));
	    x_local[j][k] = value/gamma; // rho
	    x_local[j][k+128] = 0;
	    x_local[j][k+256] = 0;// u and v
	    x_local[j][k+384] = value; // p	   
	}
    }
    for(int step = 0; step < 1000; step++)
    {
	if(rank == 0)
	    printf("current time step = %d\n", step); 
	// step 1: explicit update for x direction
	for(int j = 0; j < 32; j++)
	{
	    double temp_p[128], temp_u[128];
	    for(int k = 0; k < 128; k++)
	    {
		int index_1 = (k+1) % 128;
		int index_2 = k == 0 ? 127 : (k-1)%128;
		x_local[j][k] =  x_local[j][k] - d*(x_local[j][index_1 + 128] - x_local[j][index_2+128]);
		temp_u[k] = x_local[j][128+k] - d*(x_local[j][384 + index_1] - x_local[j][384+index_2]);
		temp_p[k] = x_local[j][384+k] - gamma*d*(x_local[j][index_1 + 128] - x_local[j][index_2+128]);
	    }
	    for(int k = 0; k < 128; k++)
	    {
		x_local[j][k+128] = temp_u[k];
		x_local[j][k+384] = temp_p[k];
	    }
	}

	// transpose x before scatter for y direction
	for(int i = 0; i < 512; i++)
	    for(int j = 0; j < 32; j++)
		x_t[i][j] = x_local[j][i];


	// scatter from rank 0 (sending rank)
	// each process then receives a 32x32 block

	MPI_Scatter(&(x_t[0][0]), scatter_blk_size, MPI_DOUBLE, r_t_local_1, scatter_blk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[128][0]), scatter_blk_size, MPI_DOUBLE, u_t_local_1, scatter_blk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	MPI_Scatter(&(x_t[256][0]), scatter_blk_size, MPI_DOUBLE, v_t_local_1, scatter_blk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[384][0]), scatter_blk_size, MPI_DOUBLE, p_t_local_1, scatter_blk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	// scatter from rank 1 (sending rank)

	MPI_Scatter(&(x_t[0][0]), scatter_blk_size, MPI_DOUBLE, r_t_local_2, scatter_blk_size, MPI_DOUBLE, 1, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[128][0]), scatter_blk_size, MPI_DOUBLE, u_t_local_2, scatter_blk_size, MPI_DOUBLE, 1, MPI_COMM_WORLD);	
	MPI_Scatter(&(x_t[256][0]), scatter_blk_size, MPI_DOUBLE, v_t_local_2, scatter_blk_size, MPI_DOUBLE, 1, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[384][0]), scatter_blk_size, MPI_DOUBLE, p_t_local_2, scatter_blk_size, MPI_DOUBLE, 1, MPI_COMM_WORLD);

	// scatter from rank 2 (sending rank)

	MPI_Scatter(&(x_t[0][0]), scatter_blk_size, MPI_DOUBLE, r_t_local_3, scatter_blk_size, MPI_DOUBLE, 2, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[128][0]), scatter_blk_size, MPI_DOUBLE, u_t_local_3, scatter_blk_size, MPI_DOUBLE, 2, MPI_COMM_WORLD);	
	MPI_Scatter(&(x_t[256][0]), scatter_blk_size, MPI_DOUBLE, v_t_local_3, scatter_blk_size, MPI_DOUBLE, 2, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[384][0]), scatter_blk_size, MPI_DOUBLE, p_t_local_3, scatter_blk_size, MPI_DOUBLE, 2, MPI_COMM_WORLD);

	// scatter from rank 3 (sending rank)

	MPI_Scatter(&(x_t[0][0]), scatter_blk_size, MPI_DOUBLE, r_t_local_4, scatter_blk_size, MPI_DOUBLE, 3, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[128][0]), scatter_blk_size, MPI_DOUBLE, u_t_local_4, scatter_blk_size, MPI_DOUBLE, 3, MPI_COMM_WORLD);	
	MPI_Scatter(&(x_t[256][0]), scatter_blk_size, MPI_DOUBLE, v_t_local_4, scatter_blk_size, MPI_DOUBLE, 3, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[384][0]), scatter_blk_size, MPI_DOUBLE, p_t_local_4, scatter_blk_size, MPI_DOUBLE, 3, MPI_COMM_WORLD);


	for(int k = 0; k < 32; k++)
	{
	    for(int j = 0; j < 32; j++)
	    {
		x_local[k][j] = r_t_local_1[k][j];
		x_local[k][j+32] = r_t_local_2[k][j];
		x_local[k][j+64] = r_t_local_3[k][j];
		x_local[k][j+96] = r_t_local_4[k][j];
		x_local[k][j+128] = u_t_local_1[k][j];
		x_local[k][j+160] = u_t_local_2[k][j];
		x_local[k][j+192] = u_t_local_3[k][j];
		x_local[k][j+224] = u_t_local_4[k][j];
		x_local[k][j+256] = v_t_local_1[k][j];
		x_local[k][j+288] = v_t_local_2[k][j];
		x_local[k][j+320] = v_t_local_3[k][j];
		x_local[k][j+352] = v_t_local_4[k][j];
		x_local[k][j+384] = p_t_local_1[k][j];
		x_local[k][j+416] = p_t_local_2[k][j];
		x_local[k][j+448] = p_t_local_3[k][j];
		x_local[k][j+480] = p_t_local_4[k][j];				
	    }
	}


	// step 2: implicit update for y direction

	for(int i = 0; i < SUBREGION_LEN; i++)
	{

	    for(int j = 0; j < 512; j++)
	    {
		for(int k = 0; k < 512; k++)
		    temp_matrix[j][k] = matrix_1_col_maj[j][k];
	    }

	    double *A = &(temp_matrix[0][0]);
	    info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, A, lda, ipiv, x_local[i], ldb);

	}

	// step 3: explicit update for y direction


	for(int j = 0; j < 32; j++)
	{
	    double temp_p[128], temp_v[128];
	    for(int k = 0; k < 128; k++)
	    {
		int index_1 = (k+1) % 128;
		int index_2 = k == 0 ? 127 : (k-1)%128;
		x_local[j][k] =  x_local[j][k] - d*(x_local[j][256+ index_1] - x_local[j][256+ index_2]);

		temp_v[k] = x_local[j][256+k] - d*(x_local[j][384 + index_1] - x_local[j][384+ index_2]);
		temp_p[k] = x_local[j][384+k] - gamma*d*(x_local[j][256 + index_1] - x_local[j][256+index_2]);
	    }
	    for(int k = 0; k < 128; k++)
	    {
		x_local[j][k+256] = temp_v[k];
		x_local[j][k+384] = temp_p[k];
	    }
	}

	// transpose and scatter for x direction in step 4 
	// implicit update for x direction
	for(int i = 0; i < 512; i++)
	    for(int j = 0; j < 32; j++)
		x_t[i][j] = x_local[j][i];
	// these local array can then be organized into a 32x512 array

	// scatter from rank 0 (sending rank)
	// each process then receives a 32x32 block
	MPI_Scatter(&(x_t[0][0]), scatter_blk_size, MPI_DOUBLE, r_t_local_1, scatter_blk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[128][0]), scatter_blk_size, MPI_DOUBLE, u_t_local_1, scatter_blk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	MPI_Scatter(&(x_t[256][0]), scatter_blk_size, MPI_DOUBLE, v_t_local_1, scatter_blk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[384][0]), scatter_blk_size, MPI_DOUBLE, p_t_local_1, scatter_blk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// scatter from rank 1 (sending rank)
	MPI_Scatter(&(x_t[0][0]), scatter_blk_size, MPI_DOUBLE, r_t_local_2, scatter_blk_size, MPI_DOUBLE, 1, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[128][0]), scatter_blk_size, MPI_DOUBLE, u_t_local_2, scatter_blk_size, MPI_DOUBLE, 1, MPI_COMM_WORLD);	
	MPI_Scatter(&(x_t[256][0]), scatter_blk_size, MPI_DOUBLE, v_t_local_2, scatter_blk_size, MPI_DOUBLE, 1, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[384][0]), scatter_blk_size, MPI_DOUBLE, p_t_local_2, scatter_blk_size, MPI_DOUBLE, 1, MPI_COMM_WORLD);
	// scatter from rank 2 (sending rank)
	MPI_Scatter(&(x_t[0][0]), scatter_blk_size, MPI_DOUBLE, r_t_local_3, scatter_blk_size, MPI_DOUBLE, 2, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[128][0]), scatter_blk_size, MPI_DOUBLE, u_t_local_3, scatter_blk_size, MPI_DOUBLE, 2, MPI_COMM_WORLD);	
	MPI_Scatter(&(x_t[256][0]), scatter_blk_size, MPI_DOUBLE, v_t_local_3, scatter_blk_size, MPI_DOUBLE, 2, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[384][0]), scatter_blk_size, MPI_DOUBLE, p_t_local_3, scatter_blk_size, MPI_DOUBLE, 2, MPI_COMM_WORLD);
	// scatter from rank 3 (sending rank)
	MPI_Scatter(&(x_t[0][0]), scatter_blk_size, MPI_DOUBLE, r_t_local_4, scatter_blk_size, MPI_DOUBLE, 3, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[128][0]), scatter_blk_size, MPI_DOUBLE, u_t_local_4, scatter_blk_size, MPI_DOUBLE, 3, MPI_COMM_WORLD);	
	MPI_Scatter(&(x_t[256][0]), scatter_blk_size, MPI_DOUBLE, v_t_local_4, scatter_blk_size, MPI_DOUBLE, 3, MPI_COMM_WORLD);
	MPI_Scatter(&(x_t[384][0]), scatter_blk_size, MPI_DOUBLE, p_t_local_4, scatter_blk_size, MPI_DOUBLE, 3, MPI_COMM_WORLD);

	for(int k = 0; k < 32; k++)
	{
	    for(int j = 0; j < 32; j++)
	    {
		x_local[k][j] = r_t_local_1[k][j];
		x_local[k][j+32] = r_t_local_2[k][j];
		x_local[k][j+64] = r_t_local_3[k][j];
		x_local[k][j+96] = r_t_local_4[k][j];
		x_local[k][j+128] = u_t_local_1[k][j];
		x_local[k][j+160] = u_t_local_2[k][j];
		x_local[k][j+192] = u_t_local_3[k][j];
		x_local[k][j+224] = u_t_local_4[k][j];
		x_local[k][j+256] = v_t_local_1[k][j];
		x_local[k][j+288] = v_t_local_2[k][j];
		x_local[k][j+320] = v_t_local_3[k][j];
		x_local[k][j+352] = v_t_local_4[k][j];
		x_local[k][j+384] = p_t_local_1[k][j];
		x_local[k][j+416] = p_t_local_2[k][j];
		x_local[k][j+448] = p_t_local_3[k][j];
		x_local[k][j+480] = p_t_local_4[k][j];				
	    }
	}	

	// step 4: implicit update for x direction
	for(int i = 0; i < SUBREGION_LEN; i++)
	{

	    for(int j = 0; j < 512; j++)
	    {
		for(int k = 0; k < 512; k++)
		    temp_matrix[j][k] = matrix_2_col_maj[j][k];
	    }

	    double *A = &(temp_matrix[0][0]);
	    info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, A, lda, ipiv, x_local[i], ldb);
	}

	if(step % 200 == 0)
	{
	    // gather communication for printing to files
	    double r_local[32][128],u_local[32][128],v_local[32][128],p_local[32][128];
	    for(int i = 0; i < 32; i++)
	    {
		for(int j = 0; j < 128; j++)
		{
		    r_local[i][j] = x_local[i][j];
		    u_local[i][j] = x_local[i][j+128];
		    v_local[i][j] = x_local[i][j+256];
		    p_local[i][j] = x_local[i][j+384];
		}
	    }
	    MPI_Gather(r_local, GATHER_BLK_SIZE, MPI_DOUBLE, r, GATHER_BLK_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Gather(u_local, GATHER_BLK_SIZE, MPI_DOUBLE, u, GATHER_BLK_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Gather(v_local, GATHER_BLK_SIZE, MPI_DOUBLE, v, GATHER_BLK_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Gather(p_local, GATHER_BLK_SIZE, MPI_DOUBLE, p, GATHER_BLK_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	    if(rank == 0)
	    {
		printf("k = %d: printing to files %s %s %s %s", 10, filelist_r[step_count], filelist_u[step_count], filelist_v[step_count], filelist_p[step_count]);

		FILE *fileid_u = fopen(filelist_u[step_count],"w");
		FILE *fileid_v = fopen(filelist_v[step_count],"w");
		FILE *fileid_r = fopen(filelist_r[step_count],"w");
		FILE *fileid_p = fopen(filelist_p[step_count],"w");	    
		step_count++;
		for(int i = 0; i < N; i++)
		{
		    for(int j = 0; j < N; j++)
		    {
			fprintf(fileid_r, "%f ", r[i][j]);	
			fprintf(fileid_u, "%f ", u[i][j]);
			fprintf(fileid_v, "%f ", v[i][j]);
			fprintf(fileid_p, "%f ", p[i][j]);
		    }
		    fprintf(fileid_u, "\n");
		    fprintf(fileid_v, "\n");
		    fprintf(fileid_r, "\n");
		    fprintf(fileid_p, "\n");

		}
		fclose(fileid_u);
		fclose(fileid_v);
		fclose(fileid_p);
		fclose(fileid_r);
	    }
	}
    }
    
    for(int i = 0; i < 32; i++)
    {
	free(x_local[i]);
    }
    free(x_local);

    // benchmark executime time
    double time_elapsed = MPI_Wtime() - starttime;
    printf("Execution time = %le seconds, with precision % le seconds\n",
	   time_elapsed, precision);
    MPI_Finalize();
    return 0;
}
