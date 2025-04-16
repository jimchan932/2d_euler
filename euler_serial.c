#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "euler.h"
#define N 128
#define M 2000

int main(int argc, char* argv[])
{
    printf("#### Linearized euler equation for computational fluid dynamics using Lapacke (serial version) ####\n");
    
    const char *filelist_u[10] = {"EulerU_1.out","EulerU_2.out","EulerU_3.out","EulerU_4.out","EulerU_5.out","EulerU_6.out","EulerU_7.out","EulerU_8.out","EulerU_9.out","EulerU_10.out"};
    const char *filelist_v[10] = {"EulerV_1.out","EulerV_2.out","EulerV_3.out","EulerV_4.out","EulerV_5.out","EulerV_6.out","EulerV_7.out","EulerV_8.out","EulerV_9.out","EulerV_10.out"};
    const char *filelist_r[10] = {"EulerR_1.out","EulerR_2.out","EulerR_3.out","EulerR_4.out","EulerR_5.out","EulerR_6.out","EulerR_7.out","EulerR_8.out","EulerR_9.out","EulerR_10.out"};
    const char *filelist_p[10] = {"EulerP_1.out","EulerP_2.out","EulerP_3.out","EulerP_4.out","EulerP_5.out","EulerP_6.out","EulerP_7.out","EulerP_8.out","EulerP_9.out","EulerP_10.out"};    
    const double delta_x = 2./N;
    const double gamma = 1.4;
    const double initial_pt = -1.;
    const double delta_t = 0.001;
    const double d = delta_t / (delta_x * 4);
    double r[128][128], u[128][128], v[128][128], p[128][128];

    lapack_int lda = 512;
    lapack_int ipiv_1[512], ipiv_2[512];
    lapack_int ldb = 512;
    
    lapack_int info = 0;
    lapack_int n = 512;
    lapack_int nrhs = 1;
    int step_count = 0;

   
    for(int i = 0; i < 128; i++)
    {
	
	double y = initial_pt + i*delta_x;
	for(int j = 0; j < 128; j++)
	{
	    double x = initial_pt + j*delta_x;
	    
	    double value = 2*exp(-100*(x*x + y*y));
	    
	    r[i][j] = value / gamma;
	    p[i][j] = value;
	    u[i][j] = v[i][j] = 0;
	}	
    }
    double start = clock();
    for(int step = 0; step < 1000; step++)
    {
	printf("current step = %d\n", step); 
	// explicit in x
	// for each row 0 <= k <= K: do operation for *,k
	for(int row = 0; row < 128; row++)
	{
	    double new_p[128], new_u[128];
	    for(int j = 0 ; j < 128; j++)
	    {
		int index_1 = (j+1) %128;
		int index_2 = j == 0 ? 127 : (j-1) % 128;
		r[row][j] = r[row][j] - d*(u[row][index_1] - u[row][index_2]);
		new_u[j] = u[row][j] - d*(p[row][index_1] - p[row][index_2]);
		new_p[j] = p[row][j] - gamma*d*(u[row][index_1] - u[row][index_2]);
		
	    }
	    
	    for(int k = 0; k < 128; k++)
	    {
	
		u[row][k] = new_u[k];
		p[row][k] = new_p[k];
	    }
	}

// implicit in y
	// first construct column major matrix for multiplication
	double r_t[128][128], u_t[128][128], v_t[128][128], p_t[128][128];
	double x[128][512];
	for(int i = 0; i < 128; i++)
	{
	    for(int j = 0; j < 128; j++)
	    {
		x[i][j] = r[j][i];
		x[i][j+128] = u[j][i];
		x[i][j+256] = v[j][i];
		x[i][j+384] = p[j][i];
	    }
	}

	double temp_matrix[512][512];
	for(int i = 0; i < 128; i++)
	{
	    
	    for(int j = 0; j < 512; j++)
	    {
		for(int k = 0; k < 512; k++)
		    temp_matrix[j][k] = matrix_1_col_maj[j][k];
	    }

	    double *A = &(temp_matrix[0][0]);
	    int flag = 0;

	    info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, A, lda, ipiv_1, x[i], ldb);
	    
			
	}

	for(int i = 0; i < 128; i++)
	{
	    for(int j = 0; j < 128; j++)
	    {
		r_t[i][j] = x[i][j];
		u_t[i][j] = x[i][j+128];
		v_t[i][j] = x[i][j+256];
		p_t[i][j] = x[i][j+384];
	    }
	}

	// explicit in y
	// for each col 0 <= j <= J: do operation for j,*
       
	for(int row = 0; row < 128; row++)
	{	    
	    double new_p[128], new_v[128];
	    for(int j = 0 ; j < 128; j++)
	    {
		int index_1 = (j+1) %128;
		int index_2 = j == 0 ? 127 : (j-1) % 128;

		r_t[row][j] = r_t[row][j] - d*(v_t[row][index_1] - v_t[row][index_2]);
	    
		new_v[j] = v_t[row][j] - d*(p_t[row][index_1] - p_t[row][index_2]);
		new_p[j] = p_t[row][j] - gamma*d*(v_t[row][index_1] - v_t[row][index_2]);
	    }
	    for(int k = 0; k < 128; k++)
	    {	
		v_t[row][k] = new_v[k];
		p_t[row][k] = new_p[k];
	    }
	}

	// implicit in x (need to transpose)
	for(int i = 0; i < 128; i++)
	{
	    for(int j = 0; j < 128; j++)
	    {
		x[i][j] = r_t[j][i];
		x[i][j+128] = u_t[j][i];
		x[i][j+256] = v_t[j][i];
		x[i][j+384] = p_t[j][i];
	    }
	}
	
	for(int i = 0; i < 128; i++)
	{

	    for(int j = 0; j < 512; j++)
	    {
		for(int k = 0; k < 512; k++)
		    temp_matrix[j][k] = matrix_2_col_maj[j][k];
 	    }
	    double *B = &(temp_matrix[0][0]);
	    
	    info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, B, lda, ipiv_2, x[i], ldb);
	}
	for(int i = 0; i < 128; i++)
	{
	    for(int j = 0; j < 128; j++)
	    {
		r[i][j] = x[i][j];
		u[i][j] = x[i][j+128];
		v[i][j] = x[i][j+256];
		p[i][j] = x[i][j+384];

	    }
	}
	if(step % 200 == 0)
	{
	    printf("k = %d: printing to files %s %s", step_count, filelist_u[step_count], filelist_v[step_count]);
		
	    FILE *fileid_u = fopen(filelist_u[step_count],"w");
	    FILE *fileid_v = fopen(filelist_v[step_count],"w");
	    FILE *fileid_r = fopen(filelist_r[step_count],"w");
	    FILE *fileid_p = fopen(filelist_p[step_count],"w");	    
	    step_count++;
	    for(int i = 0; i < N; i++)
	    {
		for(int j = 0; j < N; j++)
		{
		    fprintf(fileid_u, "%f ", u[i][j]);
		    fprintf(fileid_v, "%f ", v[i][j]);
		    fprintf(fileid_r, "%f ", r[i][j]);
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
    double end = clock();
    printf("Execution time(in seconds): %lf\n", ((double) (end - start)) / CLOCKS_PER_SEC);
    return 0;
}
