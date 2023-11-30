#include <chrono>
#include <thread>
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

struct rgb {
    uint8_t r, g, b;
};

struct XYZ {
    double X, Y, Z;
};

struct CIELAB {
    double L, a, b;
};

std::vector<double> createCorrectColorLUT() {
    std::vector<double> lut(256);
    for (int i = 0; i < 256; ++i) {
        double c = i / 255.0;
        lut[i] = c > 0.04045 ? pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
    }
    return lut;
}

inline XYZ rgb_to_XYZ(rgb rgb) {
    static const std::vector<double> correctColorLUT = createCorrectColorLUT();

    double r = correctColorLUT[rgb.r] * 100.0;
    double g = correctColorLUT[rgb.g] * 100.0;
    double b = correctColorLUT[rgb.b] * 100.0;

    // Direct computation
    XYZ xyz = {
            r * 0.4124 + g * 0.3576 + b * 0.1805,
            r * 0.2126 + g * 0.7152 + b * 0.0722,
            r * 0.0193 + g * 0.1192 + b * 0.9505
    };

    return xyz;
}

inline CIELAB xyz_to_CIELAB(XYZ xyz) {
    // Precompute division
    double x = xyz.X / 95.047;
    double y = xyz.Y / 100.0;
    double z = xyz.Z / 108.883;

    // Using a lambda for the repeated operation
    auto f = [](double t) -> double {
        return t > 0.008856 ? pow(t, 1.0 / 3.0) : (7.787 * t) + (16.0 / 116.0);
    };

    x = f(x);
    y = f(y);
    z = f(z);

    // Direct computation
    CIELAB lab = {
            (116.0 * y) - 16.0,
            500.0 * (x - y),
            200.0 * (y - z)
    };

    return lab;
}

inline CIELAB rgb_to_CIELAB(rgb rgb) {
    XYZ xyz = rgb_to_XYZ(rgb);
    return xyz_to_CIELAB(xyz);
}

inline CIELAB* convert_image_to_lab(const rgb* rgb_image, int width, int height) {
    CIELAB* cielab = new CIELAB[width * height];
    for (int i = 0; i < width * height; ++i) {
        cielab[i] = rgb_to_CIELAB(rgb_image[i]);
    }
    return cielab;
}

double* compute_residual_image(CIELAB* background, CIELAB* image, int width, int height) {
    double* residual_image = new double[height * width];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            double dL = image[idx].L - background[idx].L;
            double da = image[idx].a - background[idx].a;
            double db = image[idx].b - background[idx].b;
            residual_image[idx] = sqrt(dL * dL + da * da + db * db);
        }
    }
    return residual_image;
}

double* erosion_dilatation(double* residual_image, int width, int height, int radius) {
    double* temp_image = new double[width * height];
    double* output_image = new double[width * height];

    // Erosion
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double min_val = std::numeric_limits<double>::max();
            for (int dy = -radius; dy <= radius; ++dy) {
                int ny = y + dy;
                if (ny >= 0 && ny < height) {
                    for (int dx = -radius; dx <= radius; dx += 2) { // Loop unrolling
                        int nx1 = x + dx;
                        int nx2 = nx1 + 1;
                        if (nx1 >= 0 && nx1 < width) {
                            min_val = std::min(min_val, residual_image[ny * width + nx1]);
                        }
                        if (nx2 >= 0 && nx2 < width) {
                            min_val = std::min(min_val, residual_image[ny * width + nx2]);
                        }
                    }
                }
            }
            temp_image[y * width + x] = min_val;
        }
    }

    // Dilatation
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (temp_image[y * width + x] == std::numeric_limits<double>::max()) { // Early skip
                output_image[y * width + x] = temp_image[y * width + x];
                continue;
            }

            double max_val = -std::numeric_limits<double>::max();
            for (int dy = -radius; dy <= radius; ++dy) {
                int ny = y + dy;
                if (ny >= 0 && ny < height) {
                    for (int dx = -radius; dx <= radius; dx += 2) { // Loop unrolling
                        int nx1 = x + dx;
                        int nx2 = nx1 + 1;
                        if (nx1 >= 0 && nx1 < width) {
                            max_val = std::max(max_val, temp_image[ny * width + nx1]);
                        }
                        if (nx2 >= 0 && nx2 < width) {
                            max_val = std::max(max_val, temp_image[ny * width + nx2]);
                        }
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
    std::fill_n(output_image, width * height, 0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (erosion_dilatation_image[y * width + x] >= threshold_high) {
                output_image[y * width + x] = 1;
            }
        }
    }

    bool has_changed;
    do {
        has_changed = false;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (output_image[y * width + x] == 1) continue;
                if (erosion_dilatation_image[y * width + x] < threshold_low) continue;

                // Vérifier les 8 voisins
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (output_image[ny * width + nx] == 1) {
                                output_image[y * width + x] = 1;
                                has_changed = true;
                                break;
                            }
                        }
                    }
                    if (output_image[y * width + x] == 1) break;
                }
            }
        }
    } while (has_changed);

    return output_image;
}

void update_pixel_history(std::vector<std::vector<CIELAB>>& pixelHistories, const CIELAB& newPixel, int x, int y, int width, int maxHistoryLength) {
    std::vector<CIELAB>& history = pixelHistories[y * width + x];
    history.push_back(newPixel);
    if (history.size() > maxHistoryLength) {
        history.erase(history.begin());
    }
}

bool compareCIELAB(const CIELAB& a, const CIELAB& b) {
    if (a.L != b.L) return a.L < b.L;
    if (a.a != b.a) return a.a < b.a;
    return a.b < b.b;
}

CIELAB compute_median(std::vector<CIELAB>& history) {
    if (history.empty()) return {0, 0, 0};  // Handle empty history case

    std::sort(history.begin(), history.end(), compareCIELAB);

    size_t n = history.size();
    if (n % 2 == 0) {
        // If even, return the average of the two middle values
        CIELAB mid1 = history[n / 2 - 1];
        CIELAB mid2 = history[n / 2];
        return {(mid1.L + mid2.L) / 2, (mid1.a + mid2.a) / 2, (mid1.b + mid2.b) / 2};
    } else {
        // If odd, return the middle value
        return history[n / 2];
    }
}

void refresh_background_median(CIELAB* background, CIELAB* actual_image, int width, int height, std::vector<std::vector<CIELAB>>& pixelHistories, int maxHistoryLength) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            update_pixel_history(pixelHistories, actual_image[idx], x, y, width, maxHistoryLength);
            background[idx] = compute_median(pixelHistories[idx]);
        }
    }
}

//replace background with the new background based on the new image
void refresh_background_mean(CIELAB* background, CIELAB* actual_image, int width, int height, int frame_count) {
    for (int i = 0; i < width * height; ++i) {
        background[i].L += (actual_image[i].L - background[i].L) / frame_count;
        background[i].a += (actual_image[i].L - background[i].L) / frame_count;
        background[i].b += (actual_image[i].L - background[i].L) / frame_count;
    }
}

CIELAB* backgroud_image = nullptr;
bool first_frame_set = false;
int frame_count = 0;
std::vector<std::vector<CIELAB>> pixelHistories;
int maxHistoryLength = 5;

extern "C" {

void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
{
    frame_count++;
    CIELAB* image_lab = convert_image_to_lab((rgb*) buffer, width, height);
    if (!first_frame_set) {
        backgroud_image = new CIELAB[width * height];
        memcpy(backgroud_image, image_lab, width * height * sizeof(CIELAB));
        first_frame_set = true;

        // Initialize pixel histories
        pixelHistories.resize(width * height);
    }
    else {
        refresh_background_median(backgroud_image, image_lab, width, height, pixelHistories, maxHistoryLength);
        double* residual_image = compute_residual_image(backgroud_image, image_lab, width, height);
        double* erosion_dilatation_image = erosion_dilatation(residual_image, width, height, 3);
        int* hysteresis_image = hysteresis(erosion_dilatation_image, width, height, 4, 30);

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
        delete[] residual_image;
        delete[] erosion_dilatation_image;
        delete[] hysteresis_image;
    }
    delete[] image_lab;
}
}