#define BLOCK_SIZE 16
#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

extern "C" __global__ void calculate_temp(int iteration,   //number of iteration
                                          float *power,    //power input
                                          float *temp_src, //temperature input/output
                                          float *temp_dst, //temperature input/output
                                          int grid_cols,   //Col of grid
                                          int grid_rows,   //Row of grid
                                          int border_cols, // border offset
                                          int border_rows, // border offset
                                          float Cap,       //Capacitance
                                          float Rx,
                                          float Ry,
                                          float Rz,
                                          float step,
                                          float time_elapsed) {

    __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

    float amb_temp = 80.0;
    float step_div_Cap;
    float Rx_1, Ry_1, Rz_1;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    step_div_Cap = step / Cap;

    Rx_1 = 1 / Rx;
    Ry_1 = 1 / Ry;
    Rz_1 = 1 / Rz;

    // each block finally computes result for a small block
    // after N iterations.
    // it is the non-overlapping small blocks that cover
    // all the input data

    // calculate the small block size
    int small_block_rows = BLOCK_SIZE - iteration * 2; //EXPAND_RATE
    int small_block_cols = BLOCK_SIZE - iteration * 2; //EXPAND_RATE

    // calculate the boundary for the block according to
    // the boundary of its small block
    int blkY = small_block_rows * by - border_rows;
    int blkX = small_block_cols * bx - border_cols;
    int blkYmax = blkY + BLOCK_SIZE - 1;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    // calculate the global thread coordination
    int yidx = blkY + ty;
    int xidx = blkX + tx;

    // load data if it is within the valid input range
    int loadYidx = yidx, loadXidx = xidx;
    int index = grid_cols * loadYidx + loadXidx;

    // RISKY ACCESS 1:
    // As usual, OoB accesses on global memory are not noticed by the CUDA runtime.
    // No OoB access on shared memory is done in this case;
    // Accesses on "temp_on_cuda" and "power_on_cuda" can be protected as their size is known at compile time,
    //   and "ty", "tx" are functions of threadIdx;
    // Protecting "temp_src" and "power" is harder as we don't know the size at compile time,
    //   and we need to associate the variable "size" to it. We can identify "index" as function of threadIdx.
    // We could end up with:
    // if (tx >= 0 && tx < BLOCK_SIZE && ty >= 0 && ty < BLOCK_SIZE && index < size)
    if (IN_RANGE(loadYidx, 0, grid_rows - 1) && IN_RANGE(loadXidx, 0, grid_cols - 1)) {
        temp_on_cuda[ty][tx] = temp_src[index]; // Load the temperature data from global memory to shared memory
        power_on_cuda[ty][tx] = power[index];   // Load the power data from global memory to shared memory
    }
    __syncthreads();

    // effective range within this block that falls within
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validYmin = (blkY < 0) ? -blkY : 0;
    int validYmax = (blkYmax > grid_rows - 1) ? BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1) : BLOCK_SIZE - 1;
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > grid_cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) : BLOCK_SIZE - 1;

    int N = ty - 1;
    int S = ty + 1;
    int W = tx - 1;
    int E = tx + 1;

    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool computed;
    for (int i = 0; i < iteration; i++) {
        computed = false;
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) &&
            IN_RANGE(ty, i + 1, BLOCK_SIZE - i - 2) &&
            IN_RANGE(tx, validXmin, validXmax) &&
            IN_RANGE(ty, validYmin, validYmax)) {
            computed = true;
            temp_t[ty][tx] = temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
                                                                    (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0 * temp_on_cuda[ty][tx]) * Ry_1 +
                                                                    (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0 * temp_on_cuda[ty][tx]) * Rx_1 +
                                                                    (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
        }
        __syncthreads();
        if (i == iteration - 1)
            break;
        if (computed) //Assign the computation range
            temp_on_cuda[ty][tx] = temp_t[ty][tx];
        __syncthreads();
    }

    // RISKY ACCESS 2:
    // Again, unchecked accesses on global memory are not identified, and the final result is correct.
    // Protecting this access using "computed" is hard, because its value is not updated in a predictable way.
    // Instead, we can use a relaxed constraint like "if (index < size)" if we can associate the array to its size;

    // update the global memory
    // after the last iteration, only threads coordinated within the
    // small block perform the calculation and switch on ``computed''
    if (computed) {
        temp_dst[index] = temp_t[ty][tx];
    }
}