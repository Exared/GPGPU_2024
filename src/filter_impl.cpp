
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
    double x = xyz.X / 95.047;         // référence X de l'illuminant D65
    double y = xyz.Y / 100.000;        // référence Y de l'illuminant D65
    double z = xyz.Z / 108.883;        // référence Z de l'illuminant D65

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

extern "C" {
    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        if (first) {
            first_frame = new uint8_t[3*height*width*sizeof(uint8_t)];
            memcpy(first_frame, buffer, 3*height*width*sizeof(uint8_t));
            first = false;
        }
        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*) (buffer + y * stride);
            rgb* first_ptr = (rgb*) (first_frame + y * stride);
            for (int x = 0; x < width; ++x)
            {
                CIELAB pixel_lab = rgb_to_CIELAB(lineptr[x]);
                CIELAB first_lab = rgb_to_CIELAB(first_ptr[x]);
                double deltaE = sqrt(pow(pixel_lab.L - first_lab.L, 2) + pow(pixel_lab.a - first_lab.a, 2) + pow(pixel_lab.b - first_lab.b, 2));
                if (deltaE > 10.0) {
                    lineptr[x].r = 255;
                    lineptr[x].g = 0;
                    lineptr[x].b = 0;
                }
            }
        }

        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }
}
