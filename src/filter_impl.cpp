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
    float X, Y, Z;
};

struct CIELAB {
    float L, a, b;
};

std::vector<float> createCorrectColorLUT() {
    std::vector<float> lut(256);
    for (int i = 0; i < 256; ++i) {
        float c = i / 255.0f;
        lut[i] = c > 0.04045f ? powf((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f;
    }
    return lut;
}

inline XYZ rgb_to_XYZ(rgb rgb) {
    static const std::vector<float> correctColorLUT = createCorrectColorLUT();

    float r = correctColorLUT[rgb.r] * 100.0f;
    float g = correctColorLUT[rgb.g] * 100.0f;
    float b = correctColorLUT[rgb.b] * 100.0f;

    // Direct computation
    XYZ xyz = {
            r * 0.4124f + g * 0.3576f + b * 0.1805f,
            r * 0.2126f + g * 0.7152f + b * 0.0722f,
            r * 0.0193f + g * 0.1192f + b * 0.9505f
    };

    return xyz;
}

inline CIELAB xyz_to_CIELAB(XYZ xyz) {
    // Precompute division
    float x = xyz.X / 95.047f;
    float y = xyz.Y / 100.0f;
    float z = xyz.Z / 108.883f;

    // Using a lambda for the repeated operation
    auto f = [](float t) -> float {
        return t > 0.008856f ? powf(t, 1.0f / 3.0f) : (7.787f * t) + (16.0f / 116.0f);
    };

    x = f(x);
    y = f(y);
    z = f(z);

    // Direct computation
    CIELAB lab = {
            (116.0f * y) - 16.0f,
            500.0f * (x - y),
            200.0f * (y - z)
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

float* compute_residual_image(CIELAB* background, CIELAB* image, int width, int height) {
    float* residual_image = new float[height * width];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float dL = image[idx].L - background[idx].L;
            float da = image[idx].a - background[idx].a;
            float db = image[idx].b - background[idx].b;
            residual_image[idx] = sqrt(dL * dL + da * da + db * db);
        }
    }
    return residual_image;
}

float* erosion_dilatation(float* residual_image, int width, int height, int radius) {
    float* temp_image = new float[width * height];
    float* output_image = new float[width * height];

    // Erosion
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float min_val = std::numeric_limits<float>::max();
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
            if (temp_image[y * width + x] == std::numeric_limits<float>::max()) { // Early skip
                output_image[y * width + x] = temp_image[y * width + x];
                continue;
            }

            float max_val = -std::numeric_limits<float>::max();
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

int* hysteresis(float* erosion_dilatation_image, int width, int height, float threshold_low, float threshold_high) {
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
        float* residual_image = compute_residual_image(backgroud_image, image_lab, width, height);
        float* erosion_dilatation_image = erosion_dilatation(residual_image, width, height, 3);
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
