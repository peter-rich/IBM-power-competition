#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

struct query_info {
    int xr_xl_space_begin_index;
    
    int x_row_amount;
    int x_col_amount;

    int yr_yl_space_begin_index;
    
    int y_row_amount;
    int y_col_amount;

    int xl_xl_space_begin_index;
    
    int xl_yl_begin_index;

};


// Kernel that executes on the CUDA device
__global__ void square_array(struct query_info *gpu_qi,
                             float *gpu_x,
                             float *gpu_y,
                             float *gpu_x_transpose,
                             float *gpu_xt_mul_x,
                             float *gpu_xt_mul_x_inverse,
                             float *gpu_xt_mul_x_inverse_mul_xt,
                             float *gpu_result,
                             int N)
{  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) {
        return;
    }
    
    int xr_xl_space_begin_index = gpu_qi[idx].xr_xl_space_begin_index;
    int x_row_amount = gpu_qi[idx].x_row_amount;
    int x_col_amount = gpu_qi[idx].x_col_amount;

    int yr_yl_space_begin_index = gpu_qi[idx].yr_yl_space_begin_index;
    int y_row_amount = gpu_qi[idx].y_row_amount;
    int y_col_amount = gpu_qi[idx].y_col_amount;

    int xl_xl_space_begin_index = gpu_qi[idx].xl_xl_space_begin_index;
    
    int xl_yl_begin_index = gpu_qi[idx].xl_yl_begin_index;

    float *now_x = (gpu_x+xr_xl_space_begin_index);
    float *now_y = (gpu_y+yr_yl_space_begin_index);
    float *now_x_transpose = (gpu_x_transpose+xr_xl_space_begin_index);
    float *now_xt_mul_x = (gpu_xt_mul_x+xl_xl_space_begin_index);
    float *now_xt_mul_x_inverse = (gpu_xt_mul_x_inverse+xl_xl_space_begin_index);
    float *now_xt_mul_x_inverse_mul_xt = (gpu_xt_mul_x_inverse_mul_xt+xr_xl_space_begin_index);
    float *now_result = (gpu_result+xl_yl_begin_index);
    
    
    //set transpose
    for (int i = 0; i < x_col_amount; i++) {
        for (int j = 0; j < x_row_amount; j++) {
            now_x_transpose[i*x_row_amount+j] = now_x[j*x_col_amount+i];
        }
    }


    //set xt_mul_x
    float sum;
    for (int i = 0; i < x_col_amount; i++) {
        
        for (int k = 0; k < x_col_amount; k++) {
            sum = 0;
            for (int j = 0; j < x_row_amount; j++) {
                sum += now_x_transpose[i*x_row_amount+j] * now_x[j*x_col_amount+k];
            }
            now_xt_mul_x[i*x_col_amount+k] = sum;
        }
    }
    
    //set xt_mul_x_inverse
    
	for(int i = 0; i < x_col_amount; i ++) {
		
		for (int j = 0; j<x_col_amount; j++){
			if (i == j) {now_xt_mul_x_inverse[i*x_col_amount+i]=1.0;}
			else {now_xt_mul_x_inverse[i*x_col_amount+j]=0.0;}
		}
		
	}

	for(int i = 0 ; i< x_col_amount; i ++) {
		for(int j = i+1; j < x_col_amount;j++){
			if(now_xt_mul_x[j*x_col_amount+i]!=0){
				float temp = now_xt_mul_x[j*x_col_amount+i]/now_xt_mul_x[i*x_col_amount+i];
				for(int k = 0; k < x_col_amount; k ++) {
					now_xt_mul_x[j*x_col_amount+k]-=temp*now_xt_mul_x[i*x_col_amount+k];
					now_xt_mul_x_inverse[j*x_col_amount+k]-=temp*now_xt_mul_x_inverse[i*x_col_amount+k];		
				}
			}
		}
	}
	for(int i = 1; i < x_col_amount; i ++) {
		for(int j = i-1;j>=0;j--) {
			if(now_xt_mul_x[j* x_col_amount+i]!=0) {
				float temp = now_xt_mul_x[j* x_col_amount+i]/now_xt_mul_x[i* x_col_amount+i];
				for(int k = 0; k <  x_col_amount; k ++) {			
					now_xt_mul_x[j* x_col_amount+k]-=temp*now_xt_mul_x[i* x_col_amount+k];
					now_xt_mul_x_inverse[j* x_col_amount+k]-=temp*now_xt_mul_x_inverse[i* x_col_amount+k];
				}
			}
		}
	}

	
	for(int i = 0 ; i< x_col_amount; i ++) {
		for(int j = 0; j <x_col_amount; j ++) {
			now_xt_mul_x_inverse[i*x_col_amount+j]/=now_xt_mul_x[i*x_col_amount+i];
		}
	}    

    //set xt_mul_x_inverse_mul_xt
    
    //set result

}


// main routine that executes on the host
extern "C" float* test(int query_amount, int *row_info, float *x, float *y)
{
    struct query_info qi[query_amount];
    int x_index = 0;
    int y_index = 0;
    int xt_mul_x_inverse_index = 0;
    int result_index = 0;
    
    clock_t begin, end;
    double time_spent;
    
    printf("first matrix in cu\n");
    for (int i = 0; i < row_info[0]; i++) {
        printf("%f %f\n", x[2*i], x[2*i+1]);

    }
    


    for (int i = 0; i < query_amount; i++) {
        qi[i].xr_xl_space_begin_index = x_index;
        qi[i].x_row_amount = row_info[i];
        qi[i].x_col_amount = 2;
        x_index += qi[i].x_row_amount*qi[i].x_col_amount;
        
        qi[i].yr_yl_space_begin_index = y_index;
        qi[i].y_row_amount = row_info[i];
        qi[i].y_col_amount = 1;
        y_index += qi[i].y_row_amount*qi[i].y_col_amount;
        
        
        qi[i].xl_xl_space_begin_index = xt_mul_x_inverse_index;
        xt_mul_x_inverse_index += qi[i].x_col_amount*qi[i].x_col_amount;
        
        qi[i].xl_yl_begin_index = result_index;
        result_index += qi[i].x_col_amount*qi[i].y_col_amount;
    }
    
    
    
    struct query_info *gpu_qi;
    float *gpu_x, *gpu_y, *gpu_x_transpose, *gpu_xt_mul_x, *gpu_xt_mul_x_inverse, *gpu_xt_mul_x_inverse_mul_xt, *gpu_result;

    const int N = query_amount;  // Number of elements in arrays


    cudaMalloc((void **) &gpu_qi, sizeof(struct query_info)*query_amount);
    
    cudaMalloc((void **) &gpu_x, x_index*sizeof(float));
    cudaMalloc((void **) &gpu_y, y_index*sizeof(float));
    
    cudaMalloc((void **) &gpu_x_transpose, x_index*sizeof(float));
    
    cudaMalloc((void **) &gpu_xt_mul_x, xt_mul_x_inverse_index*sizeof(float));
    cudaMalloc((void **) &gpu_xt_mul_x_inverse, xt_mul_x_inverse_index*sizeof(float));
    cudaMalloc((void **) &gpu_xt_mul_x_inverse_mul_xt, x_index*sizeof(float));
    
    
    cudaMalloc((void **) &gpu_result, result_index*sizeof(float));
    
    cudaMemcpy(gpu_qi, qi, sizeof(struct query_info)*query_amount, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_x, x, x_index*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y, y, y_index*sizeof(float), cudaMemcpyHostToDevice);
    
    
    // Do calculation on device:
    int block_size = 4;
    int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
    
    begin = clock();
    square_array <<< n_blocks, block_size >>> (gpu_qi, gpu_x, gpu_y, gpu_x_transpose, gpu_xt_mul_x, gpu_xt_mul_x_inverse, gpu_xt_mul_x_inverse_mul_xt, gpu_result, N);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
    printf("time cost: %f\n", time_spent);
    
    float *result = (float *)malloc(result_index*sizeof(float));
    float *tmp_x = (float *)malloc(x_index*sizeof(float));
    float *x_t = (float *)malloc(x_index*sizeof(float));
    float *xt_mul_x = (float *)malloc(xt_mul_x_inverse_index*sizeof(float));
    float *xt_mul_x_inverse = (float *)malloc(xt_mul_x_inverse_index*sizeof(float));
    
    
    // Retrieve result from device and store it in host array
    cudaMemcpy(result, gpu_result, result_index*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmp_x, gpu_x, x_index*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x_t, gpu_x_transpose, x_index*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xt_mul_x, gpu_xt_mul_x, xt_mul_x_inverse_index*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xt_mul_x_inverse, gpu_xt_mul_x_inverse, xt_mul_x_inverse_index*sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("first matrix's x_t\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < row_info[0]; j++) {
            printf("%f ", x_t[i*row_info[0]+j]);
        }
        printf("\n");
    }
    
    printf("first matrix's xt_mul_x\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%f ", xt_mul_x[i*2+j]);
        }
        printf("\n");
    }
    
    printf("bye\n");

    return x_t;
}



