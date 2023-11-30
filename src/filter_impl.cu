#include "filter_impl.h"

#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cfloat>

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

struct lab {
    float l, a, b;
};

struct xyz {
    float x, y, z;
};

__device__ xyz rgb_to_xyz(rgb rgb) {
    double r = rgb.r / 255.0;
    double g = rgb.g / 255.0;
    double b = rgb.b / 255.0;

    r = r > 0.04045 ? __powf((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = g > 0.04045 ? __powf((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = b > 0.04045 ? __powf((b + 0.055) / 1.055, 2.4) : b / 12.92;

    r *= 100.0;
    g *= 100.0;
    b *= 100.0;

    xyz xyz;
    xyz.x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    xyz.y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    xyz.z = r * 0.0193 + g * 0.1192 + b * 0.9505;

    return xyz;
}

__device__ lab xyz_to_lab(xyz xyz) {
    double x = xyz.x / 95.047;
    double y = xyz.y / 100.000;
    double z = xyz.z / 108.883;

    x = x > 0.008856 ? __powf(x, 1.0/3.0) : (7.787 * x) + (16.0 / 116.0);
    y = y > 0.008856 ? __powf(y, 1.0/3.0) : (7.787 * y) + (16.0 / 116.0);
    z = z > 0.008856 ? __powf(z, 1.0/3.0) : (7.787 * z) + (16.0 / 116.0);

    lab lab;
    lab.l = (116.0 * y) - 16.0;
    lab.a = 500.0 * (x - y);
    lab.b = 200.0 * (y - z);

    return lab;
}

__device__ lab rgb_to_lab(rgb rgb) {
    xyz xyz = rgb_to_xyz(rgb);
    return xyz_to_lab(xyz);
}

__global__ void convert_to_cielab(std::byte* buffer, int width, int height, int stridein, int strideout, std::byte* output) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return; 

    rgb* lineptr = (rgb*) (buffer + y * stridein);
    lab* outptr = (lab*) (output + y * strideout);
    outptr[x] = rgb_to_lab(lineptr[x]);
}

__global__ void compute_residual_image(
    std::byte* buffer1, //lab
    std::byte* buffer2, //lab
    std::byte* residual, //float
    int width,
    int height,
    int stride1,//lab stride
    int stride2) { ///float stride
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width || y >= height)
        return;
    
    lab *lineptr1 = (lab*) (buffer1 + y * stride1);
    lab *lineptr2 = (lab*) (buffer2 + y * stride1);
    float *outptr = (float*) (residual + y * stride2);
    outptr[x] = sqrt(__powf(lineptr1[x].l - lineptr2[x].l, 2) + __powf(lineptr1[x].a - lineptr2[x].a, 2) + __powf(lineptr1[x].b - lineptr2[x].b, 2));
}

// Kernel pour l'érosion
__global__ void erosion_kernel(
    float* input,
    float* output,
    int width,
    int height,
    int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float minVal = FLT_MAX;

    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            int ix = x + dx;
            int iy = y + dy;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                float val = input[iy * stride / sizeof(float) + ix];
                minVal = fminf(minVal, val);
            }
        }
    }

    output[y * stride / sizeof(float) + x] = minVal;
}

// Kernel pour la dilatation
__global__ void dilatation_kernel(float* input,
    float* output,
    int width,
    int height,
    int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float maxVal = -FLT_MAX;

    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            int ix = x + dx;
            int iy = y + dy;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                float val = input[iy * stride / sizeof(float) + ix];
                maxVal = fmaxf(maxVal, val);
            }
        }
    }

    output[y * stride / sizeof(float) + x] = maxVal;
}

__global__ void hysteresis_threshold_kernel(
    float* dilatation, 
    unsigned char* mask, 
    int width, 
    int height, 
    int dilatation_stride,
    int mask_stride,
    float low_threshold,
    float high_threshold) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float pixelValue = dilatation[y * dilatation_stride / sizeof(float) + x];
    unsigned char maskValue = 0;  // Initialisation à 0 (arrière-plan)

    // Seuillage d'hystérésis
    if (pixelValue > high_threshold) {
        maskValue = 1;  // Premier plan
    } else if (pixelValue > low_threshold) {
        // Vérification des voisins pour les valeurs entre les deux seuils
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int ix = x + dx;
                int iy = y + dy;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    float neighborValue = dilatation[iy * dilatation_stride / sizeof(float) + ix];
                    if (neighborValue > high_threshold) {
                        maskValue = 1;  // Premier plan si connecté à un voisin de premier plan
                        break;
                    }
                }
            }
            if (maskValue == 1) break;  // Arrêter la recherche si un voisin de premier plan est trouvé
        }
    }

    mask[y * mask_stride + x] = maskValue;
}

__global__ void apply_mask(
    std::byte* buffer, 
    unsigned char* mask, 
    int width, 
    int height, 
    int buffer_stride,
    int mask_stride) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    rgb* lineptr = (rgb*) (buffer + y * buffer_stride);
    unsigned char* maskptr = (unsigned char*) (mask + y * mask_stride);
    if (maskptr[x] == 1) {
        lineptr[x].r = 255;
    }
}

__global__ void recompute_mean_background(
    std::byte* background, 
    std::byte* buffer,
    int width, 
    int height, 
    int background_stride,
    int buffer_stride,
    int frame_count) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    lab* background_lineptr = (lab*) (background + y * background_stride);
    lab* buffer_lineptr = (lab*) (buffer + y * buffer_stride);
    background_lineptr[x].l = (background_lineptr[x].l * (frame_count - 1) + buffer_lineptr[x].l) / frame_count;
    background_lineptr[x].a = (background_lineptr[x].a * (frame_count - 1) + buffer_lineptr[x].a) / frame_count;
    background_lineptr[x].b = (background_lineptr[x].b * (frame_count - 1) + buffer_lineptr[x].b) / frame_count;
}

int first = 0;
std::byte* first_image_lab;
int frame_count = 0;

extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        assert(sizeof(rgb) == pixel_stride);
        std::byte* dBuffer;
        size_t pitch;
        std::byte* dlabBuffer;
        size_t labpitch;
        std::byte* dResidual;
        size_t residualpitch;
        std::byte* dErosion;
        size_t erosionpitch;
        std::byte* dDilatation;
        size_t dilatationpitch;
        std::byte* binary_mask;
        size_t binary_mask_pitch;

        cudaError_t err;
        frame_count++;

        dim3 blockSize(16,16);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);

        // Allocate a buffer on the GPU for the input
        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);
        // Copy the input buffer to the GPU buffer
        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);
        
        if (first == 0) {
            //allocate the first image
            err = cudaMallocPitch(&first_image_lab, &labpitch, width * sizeof(lab), height);
            CHECK_CUDA_ERROR(err);
            convert_to_cielab<<<gridSize, blockSize>>>(dBuffer, width, height, pitch, labpitch, first_image_lab);
            first = 1;
            cudaFree(dBuffer);
            return;
        }

        // Allocate a buffer on the GPU for the input
        err = cudaMallocPitch(&dlabBuffer, &labpitch, width * sizeof(lab), height);
        CHECK_CUDA_ERROR(err);

        convert_to_cielab<<<gridSize, blockSize>>>(dBuffer, width, height, pitch, labpitch, dlabBuffer);

        recompute_mean_background<<<gridSize, blockSize>>>(first_image_lab, dlabBuffer, width, height, labpitch, labpitch, frame_count);

        // Allocate a buffer on the GPU for the residual
        err = cudaMallocPitch(&dResidual, &residualpitch, width * sizeof(float), height);
        CHECK_CUDA_ERROR(err);

        compute_residual_image<<<gridSize, blockSize>>>(first_image_lab, dlabBuffer, dResidual, width, height, labpitch, residualpitch);
        
        // Allocate a buffer on the GPU for the erosion
        err = cudaMallocPitch(&dErosion, &erosionpitch, width * sizeof(float), height);
        CHECK_CUDA_ERROR(err);

        erosion_kernel<<<gridSize, blockSize>>>((float*)dResidual, (float*)dErosion, width, height, residualpitch);

        // Allocate a buffer on the GPU for the dilatation
        err = cudaMallocPitch(&dDilatation, &dilatationpitch, width * sizeof(float), height);
        CHECK_CUDA_ERROR(err);

        dilatation_kernel<<<gridSize, blockSize>>>((float*)dErosion, (float*)dDilatation, width, height, erosionpitch);

        // Allocate a buffer on the GPU for the binary mask
        err = cudaMallocPitch(&binary_mask, &binary_mask_pitch, width * sizeof(unsigned char), height);
        CHECK_CUDA_ERROR(err);

        hysteresis_threshold_kernel<<<gridSize, blockSize>>>((float*)dDilatation, (unsigned char*)binary_mask, width, height, dilatationpitch, binary_mask_pitch, 4, 30);

        apply_mask<<<gridSize, blockSize>>>(dBuffer, (unsigned char*)binary_mask, width, height, pitch, binary_mask_pitch);
        
        // Copy the result back to the CPU
        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        cudaFree(dBuffer);
        cudaFree(dlabBuffer);
        cudaFree(dResidual);
        cudaFree(dErosion);
        cudaFree(dDilatation);
        cudaFree(binary_mask);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);
    }   
}