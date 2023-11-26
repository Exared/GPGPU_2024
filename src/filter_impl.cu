#include "filter_impl.h"

#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

struct rgb {
    uint8_t r, g, b;
};

struct XYZ {
    double X, Y, Z;
};

struct CIELAB {
    double L, a, b;
};


__global__ void convert_to_cielab_kernel(rgb* src, CIELAB* dest, int width, int height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return; 

    rgb* lineptr = (rgb*) ((std::byte*) src + y * width * sizeof(rgb));
    CIELAB* dest_lineptr = dest + y * width;
    rgb pixel = lineptr[x];

    XYZ xyz;
    xyz.X = 0.412453 * pixel.r + 0.357580 * pixel.g + 0.180423 * pixel.b;
    xyz.Y = 0.212671 * pixel.r + 0.715160 * pixel.g + 0.072169 * pixel.b;
    xyz.Z = 0.019334 * pixel.r + 0.119193 * pixel.g + 0.950227 * pixel.b;

    xyz.X /= 255;
    xyz.Y /= 255;
    xyz.Z /= 255;

    xyz.X = xyz.X > 0.04045 ? pow((xyz.X + 0.055) / 1.055, 2.4) : xyz.X / 12.92;
    xyz.Y = xyz.Y > 0.04045 ? pow((xyz.Y + 0.055) / 1.055, 2.4) : xyz.Y / 12.92;
    xyz.Z = xyz.Z > 0.04045 ? pow((xyz.Z + 0.055) / 1.055, 2.4) : xyz.Z / 12.92;

    xyz.X *= 100;
    xyz.Y *= 100;
    xyz.Z *= 100;

    dest_lineptr[x].L = 116 * xyz.Y - 16;
    dest_lineptr[x].a = 500 * (xyz.X - xyz.Y);
    dest_lineptr[x].b = 200 * (xyz.Y - xyz.Z);
}

__global__ void compute_residual_kernel(CIELAB* background, CIELAB* current, double* residual, int width, int height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return; 

    CIELAB* background_lineptr = background + y * width;
    CIELAB* current_lineptr = current + y * width;
    double* residual_lineptr = residual + y * width;

    residual_lineptr[x] = sqrt(pow(background_lineptr[x].L - current_lineptr[x].L, 2) + pow(background_lineptr[x].a - current_lineptr[x].a, 2) + pow(background_lineptr[x].b - current_lineptr[x].b, 2));
}

__global__ void erosion_kernel(double* input, double* output, int width, int height, int radius) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return; 

    double* input_lineptr = input + y * width;
    double* output_lineptr = output + y * width;

    double min = 1e9;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int x2 = x + i;
            int y2 = y + j;
            if (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height) {
                double val = input_lineptr[x2 + y2 * width];
                if (val < min) {
                    min = val;
                }
            }
        }
    }

    output_lineptr[x] = min;
}

__global__ void dilatation_kernel(double* input, double* output, int width, int height, int radius) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return; 

    double* input_lineptr = input + y * width;
    double* output_lineptr = output + y * width;

    double max = -1e9;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int x2 = x + i;
            int y2 = y + j;
            if (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height) {
                double val = input_lineptr[x2 + y2 * width];
                if (val > max) {
                    max = val;
                }
            }
        }
    }

    output_lineptr[x] = max;
}


__global__ void hysteresis_kernel(double* input, int* output, int width, int height, double lowThresh, double highThresh) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return; 

    double* input_lineptr = input + y * width;
    int* output_lineptr = output + y * width;

    if (input_lineptr[x] > highThresh) {
        output_lineptr[x] = 1;
    } else if (input_lineptr[x] < lowThresh) {
        output_lineptr[x] = 0;
    } else {
        bool found = false;
        for (int i = -1; i <= 1 && !found; i++) {
            for (int j = -1; j <= 1 && !found; j++) {
                int x2 = x + i;
                int y2 = y + j;
                if (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height) {
                    if (input_lineptr[x2 + y2 * width] > highThresh) {
                        output_lineptr[x] = 1;
                        found = true;
                    }
                }
            }
        }
        if (!found) {
            output_lineptr[x] = 0;
        }
    }
}

__global__ void apply_mask_kernel(rgb* image, int* mask, int width, int height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return; 

    rgb* image_lineptr = image + y * width;
    int* mask_lineptr = mask + y * width;

    if (mask_lineptr[x] == 0) {
        image_lineptr[x].r = 0;
        image_lineptr[x].g = 0;
        image_lineptr[x].b = 0;
    }
    else {
        image_lineptr[x].r = 255;
        image_lineptr[x].g = 255;
        image_lineptr[x].b = 255;
    }
}


//define background
CIELAB* background = nullptr;
bool is_first_frame = true;
int frame_count = 0;

extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        // Allocate memory on the device
        if (is_first_frame) {
            CHECK_CUDA_ERROR(cudaMalloc(&background, width * height * sizeof(CIELAB)));
            is_first_frame = false;
        }

        // Copy the background to the device
        CHECK_CUDA_ERROR(cudaMemcpy(background, src_buffer, width * height * sizeof(CIELAB), cudaMemcpyHostToDevice));

        // Convert the image to CIELAB
        dim3 block_size(32, 32);
        dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
        convert_to_cielab_kernel<<<grid_size, block_size>>>((rgb*) src_buffer, background, width, height);

        // Compute the residual
        double* residual;
        CHECK_CUDA_ERROR(cudaMalloc(&residual, width * height * sizeof(double)));
        compute_residual_kernel<<<grid_size, block_size>>>(background, background, residual, width, height);

        // Erosion
        double* erosion;
        CHECK_CUDA_ERROR(cudaMalloc(&erosion, width * height * sizeof(double)));
        erosion_kernel<<<grid_size, block_size>>>(residual, erosion, width, height, 1);

        // Dilatation
        double* dilatation;
        CHECK_CUDA_ERROR(cudaMalloc(&dilatation, width * height * sizeof(double)));
        dilatation_kernel<<<grid_size, block_size>>>(erosion, dilatation, width, height, 1);

        // Hysteresis
        int* mask;
        CHECK_CUDA_ERROR(cudaMalloc(&mask, width * height * sizeof(int)));
        hysteresis_kernel<<<grid_size, block_size>>>(dilatation, mask, width, height, 0.1, 0.2);

        // Apply mask
        apply_mask_kernel<<<grid_size, block_size>>>((rgb*) src_buffer, mask, width, height);

        // Copy the background to the device
        CHECK_CUDA_ERROR(cudaMemcpy(background, src_buffer, width * height * sizeof(CIELAB), cudaMemcpyHostToDevice));

        // Copy the result back to the host
        CHECK_CUDA_ERROR(cudaMemcpy(src_buffer, background, width * height * sizeof(CIELAB), cudaMemcpyDeviceToHost));

        // Free memory
        CHECK_CUDA_ERROR(cudaFree(residual));
        CHECK_CUDA_ERROR(cudaFree(erosion));
    }   
}