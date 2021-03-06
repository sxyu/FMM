/**
 *  Copyright 2019 Alex Yu
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Fast marching method implementation
 *  usage: fmm::fmm(image, seeds, weight_map_type = IDENTITY,
 *                  segmentation_threshold = disabled,
 *                  normalize_output_geodesic_distances = true,
 *                  output = nullptr)
 *  > returns an image, either geodesic distance map or, if
 *    segmentation_threshold is given, a segmentation mask
 *  image: input image (can be OpenCV Mat or fmm::Image<T>)
 *  seeds: std::vector of points (each can be OpenCV Point or fmm::Point)
 *  weight_map_type: transformation to apply to input image to use as FMM
*                    weight function. Can be one of:
 *                    fmm::weight::IDENTITY  no transformation (avoids a copy, faster)
 *                    fmm::weight::GRADIENT  recommended: image gradient magnitude (using Sobel gradients) 
 *                    fmm::weight::ABSDIFF   absolute difference from average
 *                                           grayscale value of seeds
 *                    fmm::weight::LAPLACIAN absolute value of image Laplacian
 *  segmentation_threshold: if specified, sets pixels with geodesic value less
 *                          than or equal to this threshold to 1 and others to 0
 *                          in the output image. If not given,the geodesic
 *                          distances map will be returned.
 *  normalize_output_geodesic_distances: if true, normalizes geodesic distances
 *                                       values to be in the interval [0, 1].
 *                                       If segmentation_threshold is specified,
 *                                       this occurs prior to segmentation.
 *                                       Default true.
 *  max_visits: maximum number of points to visit. Can help speed up the computation
 *              if objects larger than a certain area are eliminated (e.g. background)
 *  output: optionally, a pointer to an already-allocated output image.
 *          This allows you to avoid a copy if you already have one
 *          allocated. By default a new image is created, and this
 *          is not necessary.
 *
 *  fmm::Image<T> usage (optional)
 *  - To make owning image fmm::Image<T>(rows, cols)
 *  - To make non-owning image that maps to row-major data (of same type, or char/uchar):
 *    fmm::Image<T>(rows, cols, data_ptr)
 * */
#ifndef FMM_HPP_DDF4F5D0_B8EA_11E9_8C9D_FFE443C4FFC0
#define FMM_HPP_DDF4F5D0_B8EA_11E9_8C9D_FFE443C4FFC0

#include<cmath>
#include<cstring>
#include<algorithm>
#include<vector>
#include<queue>

namespace fmm {

// Very simple OpenCV compatible point
struct Point { int x, y; };
// Basic image with static field type
template <class T>
struct Image {
    T* data;
    int rows, cols;
    bool own_data = false;
    // Standard constructors/assignment operators
    Image() {}
    Image(Image& other) { *this = other; }
    Image(Image&& other) { *this = std::move(other); }
    Image& operator= (Image& other) {
        if (other.rows > rows || other.cols > cols) {
            fprintf(stderr, "Image copy assignment failed: input image is larger at "
                    "%dx%d (target  image is %dx%d)", other.cols, other.rows, cols, rows);
            return;
        }
        memcpy(data, other.data, other.rows * other.cols);
    }
    Image& operator= (Image&& other) {
        auto tmp = other.own_data;
        if (own_data) delete[] data;
        other.own_data = own_data = false;
        rows = other.rows; cols = other.cols; data = other.data;
        own_data = tmp;
        return *this;
    }

    // Create new image with own data
    Image(int r, int c) { create(r, c); }
    // Create non-owning image with size mapping to data pointer
    template<class U> Image(int r, int c, U* data) :
        data(reinterpret_cast<T*>(data)), rows(r), cols(c) {}
    // Destructor
    ~Image() { if (own_data) delete[] data; }
    // Create new image with own data
    void create(int r, int c) {
        rows= r; cols = c;
        data = new T[r*c];
        own_data = true;
    }
};

namespace __internal {
    // Cell for priority queue
    template<class T> struct __queue_cell {
        int id, x; 
        T dist;
        __queue_cell() {}
        __queue_cell(int id, int x, T dist) : id(id), x(x), dist(dist) { }
    };

    // Definitions: Create an image with same shape and type as other
    // You may add definitions here for custom types
    // OpenCV compatibility
    template<class T> T create_image_like(const T& other) {
        return T(other.rows, other.cols, other.type());
    }
    // Implementation for fmm::Image
    template<class U>
        Image<U> create_image_like(const Image<U>& other) {
        return Image<U>(other.rows, other.cols);
    }
}

namespace weight {
enum WeightMap {
    IDENTITY = 0,
    GRADIENT,
    ABSDIFF,
    LAPLACIAN,

    // Do not use this, this is for counting the number of enum values.
    // If you use this it will be considered to be IDENTITY
    _WEIGHT_MAP_COUNT
};
template<class T, class PointLike, class ImageLike>
ImageLike difference_weights(const ImageLike& image, const std::vector<PointLike>& seeds) {
    ImageLike diff_out = __internal::create_image_like(image);
    const T* image_data = reinterpret_cast<T*>(image.data);
    T* out_data = reinterpret_cast<T*>(diff_out.data);
    T ref = 0;
    for (const auto& seed: seeds) ref += image_data[seed.x + seed.y * image.cols];
    ref /= seeds.size();
    for (int i = 0; i < image.cols * image.rows; ++i) {
        out_data[i] = std::fabs(image_data[i] - ref);
    }
    return diff_out;
}

template<class T, class ImageLike>
ImageLike gradient_weights(const ImageLike& image, bool normalize_output = false) {
    ImageLike gradient_out = __internal::create_image_like(image);
    int idx = 0, j;
    const int area = image.rows * image.cols;
    const T* image_data = reinterpret_cast<T*>(image.data);
    T* out_data = reinterpret_cast<T*>(gradient_out.data);
    T sobelx, sobely;
    while (idx < area) {
        for (j = 0; j < image.cols; ++j) {
            sobelx = sobely = 0;
            if (j > 0) {
                if (idx - image.cols >= 0) {
                    sobelx = -3 * image_data[idx - image.cols - 1];
                    sobely = -3 * image_data[idx - image.cols - 1];
                }
                sobelx -= 10 * image_data[idx - 1];
                if (idx + image.cols < area) {
                    sobelx -= 3 * image_data[idx + image.cols - 1];
                    sobely += 3 *image_data[idx + image.cols - 1];
                }
            }
            if (j + 1 < image.cols) {
                if (idx - image.cols >= 0) {
                    sobelx += 3 * image_data[idx - image.cols + 1];
                    sobely -= 3 * image_data[idx - image.cols + 1];
                }
                sobelx += 10 * image_data[idx + 1];
                if (idx + image.cols < area) {
                    sobelx += 3 * image_data[idx + image.cols + 1];
                    sobely += 3 * image_data[idx + image.cols + 1];
                }
            }
            if (idx - image.cols >= 0) sobely -= 10 * image_data[idx - image.cols];
            if (idx + image.cols < area) sobely += 10 * image_data[idx + image.cols];
            out_data[idx] = std::sqrt(sobelx * sobelx + sobely * sobely);
            ++idx;
        }
    }
    if (normalize_output) {
        T maxval = *std::max_element(out_data, out_data + area);
        for (int i = 0; i < area; ++i) out_data[i] /= maxval;
    }
    return gradient_out;
}

template<class T, class ImageLike>
ImageLike laplacian_weights(const ImageLike& image, bool normalize_output = false) {
    ImageLike laplacian_out = __internal::create_image_like(image);
    int idx = 0, j;
    const int area = image.rows * image.cols;
    const T* image_data = reinterpret_cast<T*>(image.data);
    T* out_data = reinterpret_cast<T*>(laplacian_out.data);
    while (idx < area) {
        for (j = 0; j < image.cols; ++j) {
            auto& out = out_data[idx];
            out = -T(4) * image_data[idx];
            if (idx - image.cols >= 0) out += image_data[idx - image.cols];
            if (idx + image.cols < area) out += image_data[idx + image.cols];
            if (j > 0) out += image_data[idx - 1];
            if (j + 1 < image.cols) out += image_data[idx + 1];
            if (out < 0) out = - out;
            ++idx;
        }
    }
    if (normalize_output) {
        T maxval = *std::max_element(out_data, out_data + area);
        for (int i = 0; i < area; ++i) out_data[i] /= maxval;
    }
    return laplacian_out;
}
}

template<class T, class ImageLike, class PointLike>
ImageLike fmm(const ImageLike& image,
         const std::vector<PointLike>& seeds,
         weight::WeightMap weight_map_type = weight::IDENTITY,
         T segmentation_threshold = std::numeric_limits<T>::max(),
         bool normalize_output_geodesic_distances = true,
         int max_visits = -1,
         ImageLike* output = nullptr) {
    using namespace weight;
    using namespace __internal;

    ImageLike image_processed;
    switch(weight_map_type) {
        case GRADIENT:
            image_processed = gradient_weights<T>(image, false);
            break;
        case ABSDIFF:
            image_processed = difference_weights<T>(image, seeds);
            break;
        case LAPLACIAN:
            image_processed = laplacian_weights<T>(image, false);
            break;
        default:
            // Identity
            image_processed.rows = image.rows;
            image_processed.cols = image.cols;
            // HACK warning: Save a copy by using const cast. Please do not modify
            // image_processed.data in this function!
            image_processed.data = const_cast<decltype(image_processed.data)>(image.data);
    }

    ImageLike dist_out;
    if (output == nullptr) {
        // Allocate new image
        dist_out = __internal::create_image_like(image);
        output = &dist_out;
    }

    static constexpr T INF = std::numeric_limits<T>::max(); 
    const int area = image.cols * image.rows;

    T* dist = reinterpret_cast<T*>(output->data);
    std::fill(dist, dist + area, INF);
    
    auto __compare = [] (const __queue_cell<T>& a, const __queue_cell<T>& b) {
            return a.dist > b.dist;
        };
    std::priority_queue<__queue_cell<T>,
                        std::vector<__queue_cell<T>>,
                        decltype(__compare)> que(__compare);
    std::vector<char> cell_status(area);

    for (const auto& seed : seeds) {
        const auto id = seed.x + seed.y * image.cols;
        que.emplace(id, seed.x, 0.f);
        dist[id] = 0.;
    }

    const T* image_data = reinterpret_cast<T*>(image_processed.data);
    __queue_cell<T> u;

    constexpr char __CELL_VISITED = char(255),
                   __CELL_NOT_VISITED = char(0);
    auto __update_cell = [area, &image, &u, image_data, &que, dist, &cell_status,
                          __CELL_VISITED, __CELL_NOT_VISITED]() {
        if (cell_status[u.id] == __CELL_VISITED) return;

        const T dleft  = u.x > 0                  ? dist[u.id - 1]    : INF;
        const T dright = u.x + 1 < image.cols     ? dist[u.id + 1]    : INF;
        const T dup    = u.id - image.cols >= 0   ? dist[u.id - image.cols] : INF;
        const T ddown  = u.id + image.cols < area ? dist[u.id + image.cols] : INF;

        const T dhoriz = std::min(dleft, dright), dvert = std::min(dup, ddown);

        const T cell_val = image_data[u.id];
        const T det = 2*dvert*dhoriz - dvert*dvert - dhoriz*dhoriz + 2*cell_val*cell_val;

        const T estimate = det >= 0. ?
            0.5f * (dhoriz+dvert + std::sqrt(det)) :
            std::min(dhoriz, dvert) + cell_val;
        
        if (estimate < dist[u.id]) {
            dist[u.id] = estimate;
            if (cell_status[u.id] == __CELL_NOT_VISITED) {
                u.dist = estimate;
                que.emplace(u);
                if (!cell_status[u.id]) cell_status[u.id] = 1;
            }
        }
    };

    while (!que.empty() && max_visits--) {
        u = que.top(); que.pop();

        if (cell_status[u.id] == __CELL_VISITED ||
            dist[u.id] > segmentation_threshold) continue;
        cell_status[u.id] = __CELL_VISITED;

        --u.id; --u.x;
        if (u.x >= 0) __update_cell();
        u.id += 2; u.x += 2;
        if (u.x < image.cols) __update_cell();
        --u.id; --u.x;

        u.id -= image.cols;
        if (u.id >= 0) __update_cell();
        u.id += image.cols * 2;
        if (u.id < area) __update_cell();
    }
    if (normalize_output_geodesic_distances) {
        T maxval = *std::max_element(dist, dist + area);
        for (int i = 0; i < area; ++i) dist[i] /= maxval;
    }
    if (segmentation_threshold < INF) {
        for (int i = 0; i < area; ++i) dist[i] = T(dist[i] <= segmentation_threshold);
    }
    return *output;
}

}
#endif // FMM_HPP_DDF4F5D0_B8EA_11E9_8C9D_FFE443C4FFC0
