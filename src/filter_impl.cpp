#include <chrono>
#include <thread>
#include <iostream>
#include <cstring>
#include <cmath>
#include "logo.h"

struct rgb {
    uint8_t r, g, b;
};

struct XYZ {
    double X, Y, Z;
};

struct CIELAB {
    double L, a, b;
};


static uint8_t* first_frame = nullptr;
static CIELAB* first_frame_lab = nullptr;
bool first = true;

XYZ rgb_to_XYZ(rgb rgb) {
    double r = rgb.r / 255.0;
    double g = rgb.g / 255.0;
    double b = rgb.b / 255.0;

    r = r > 0.04045 ? pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = g > 0.04045 ? pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = b > 0.04045 ? pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    r *= 100.0;
    g *= 100.0;
    b *= 100.0;

    XYZ xyz;
    xyz.X = r * 0.4124 + g * 0.3576 + b * 0.1805;
    xyz.Y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    xyz.Z = r * 0.0193 + g * 0.1192 + b * 0.9505;

    return xyz;
}

CIELAB xyz_to_CIELAB(XYZ xyz) {
    double x = xyz.X / 95.047;
    double y = xyz.Y / 100.000;
    double z = xyz.Z / 108.883;

    x = x > 0.008856 ? pow(x, 1.0/3.0) : (7.787 * x) + (16.0 / 116.0);
    y = y > 0.008856 ? pow(y, 1.0/3.0) : (7.787 * y) + (16.0 / 116.0);
    z = z > 0.008856 ? pow(z, 1.0/3.0) : (7.787 * z) + (16.0 / 116.0);

    CIELAB lab;
    lab.L = (116.0 * y) - 16.0;
    lab.a = 500.0 * (x - y);
    lab.b = 200.0 * (y - z);

    return lab;
}

CIELAB rgb_to_CIELAB(rgb rgb) {
    XYZ xyz = rgb_to_XYZ(rgb);
    return xyz_to_CIELAB(xyz);
}

CIELAB* convert_image_to_lab(rgb* rgb_image, int width, int height) {
    CIELAB* cielab = new CIELAB[height*width];
    for (int y = 0; y < height; ++y)
    {
        rgb* lineptr = (rgb*) (rgb_image + y * width);
        for (int x = 0; x < width; ++x)
        {
            cielab[y*width + x] = rgb_to_CIELAB(lineptr[x]);
        }
    }
    return cielab;
}

double* compute_residual_image(CIELAB* background, CIELAB* image, int width, int height) {
    double* residual_image = new double[height*width];
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x) {
            residual_image[y*width + x] = sqrt(pow(image[y*width + x].L - background[y*width + x].L, 2) + pow(image[y*width + x].a - background[y*width + x].a, 2) + pow(image[y*width + x].b - background[y*width + x].b, 2));
        }
    }
    return residual_image;
}

double* errosion_dilatation(double* residual_image, int width, int height, int radius) {
    double* temp_image = new double[width * height];
    double* output_image = new double[width * height];

    // Érosion
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double min_val = std::numeric_limits<double>::max();
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height && dx * dx + dy * dy <= radius * radius) {
                        min_val = std::min(min_val, residual_image[(y + dy) * width + (x + dx)]);
                    }
                }
            }
            temp_image[y * width + x] = min_val;
        }
    }

    // Dilatation
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double max_val = -std::numeric_limits<double>::max();
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height && dx * dx + dy * dy <= radius * radius) {
                        max_val = std::max(max_val, temp_image[(y + dy) * width + (x + dx)]);
                    }
                }
            }
            output_image[y * width + x] = max_val;
        }
    }

    delete[] temp_image;
    return output_image;
}

int* hysteresis(double* erosion_dilatation_image, int width, int height, double threshold_low, double threshold_high) {
    int* output_image = new int[width * height];
    std::fill_n(output_image, width * height, 0); // Initialise l'image de sortie à zéro

    // Marquer les pixels forts
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (erosion_dilatation_image[y * width + x] >= threshold_high) {
                output_image[y * width + x] = 1; // Marquer comme fort
            }
        }
    }

    // Propager les pixels forts
    bool has_changed;
    do {
        has_changed = false;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (output_image[y * width + x] == 1) continue; // Ignorer les pixels déjà forts
                if (erosion_dilatation_image[y * width + x] < threshold_low) continue; // Ignorer les pixels trop faibles

                // Vérifier les 8 voisins
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (output_image[ny * width + nx] == 1) {
                                output_image[y * width + x] = 1; // Marquer comme fort
                                has_changed = true;
                                break;
                            }
                        }
                    }
                    if (output_image[y * width + x] == 1) break; // Sortir dès qu'un pixel fort est trouvé
                }
            }
        }
    } while (has_changed);

    return output_image;
}

extern "C" {
    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        if (first) {
            first_frame = new uint8_t[height*stride*pixel_stride];
            memcpy(first_frame, buffer, height*stride);
            first_frame_lab = convert_image_to_lab((rgb*) first_frame, width, height);
            first = false;
        }

        CIELAB* image_lab = convert_image_to_lab((rgb*) buffer, width, height);
        double* residual_image = compute_residual_image(first_frame_lab, image_lab, width, height);
        double* errosion_dilatation_image = errosion_dilatation(residual_image, width, height, 3);
        int* hysteresis_image = hysteresis(errosion_dilatation_image, width, height, 4, 30);

        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*) (buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                if (hysteresis_image[y*width + x] == 1) {
                    lineptr[x].r = 255;
                }
            }
        }

        delete[] image_lab;
        delete[] residual_image;
        delete[] errosion_dilatation_image;
        delete[] hysteresis_image;
        
        

        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }
}
