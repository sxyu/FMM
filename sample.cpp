/** Fast marching method sample project */
#include <string>
#include "fmm.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

constexpr char WIND_NAME[] = "Image";
const char* WEIGHT_MAP_NAMES[] = {"Identity",  "Gradient magnitude", "AbsDiff", "Laplacian"};
int weight_map = fmm::weight::GRADIENT;
bool segmentation_enabled = false;
float thresh = 0.072f;
std::vector<cv::Point> seeds;

void update(const cv::Mat& gray_float) {
    // The primary FMM call happens here
    cv::Mat result = fmm::fmm<float>(gray_float, seeds, fmm::weight::WeightMap(weight_map),
            segmentation_enabled ? thresh : std::numeric_limits<float>::max());
    cv::imshow(WIND_NAME, result);
}

// Handler for click to segment
static void mouse_handler(int event, int x, int y, int _, void* data){
    if (event != 1) return;
    printf("FMM from: %d %d\n", x, y); 
    seeds = std::vector<cv::Point> {{x, y}};
    update(*reinterpret_cast<cv::Mat*>(data));
}

int main(int argc, char** argv) {
    if (argc < 2 || argc == 3 || argc > 6 || (argc >= 2 && strcmp(argv[1], "--help") == 0)) {
        fprintf(stderr, "Usage: fmmtool image_path [seedx seedy [weight_map_type_int [segment_thresh]]]\n");
        return 0;
    }
    const std::string image_path = argv[1];
    if (argc > 3) seeds.emplace_back(std::atoi(argv[2]), std::atoi(argv[3]));
    if (argc > 4) weight_map = std::atoi(argv[4]);
    if (argc > 5) {
        segmentation_enabled = true;
        thresh = static_cast<float>(std::atof(argv[5]));
    }

    cv::Mat image, image_float;
    if (image_path.size() > 4 && !image_path.compare(image_path.size()-4, image_path.size(), ".exr")) {
        // Depth image
        image = cv::imread(image_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        if (image.channels() == 3) {
            // Was stored as XYZ map (point cloud), extract depth
            cv::extractChannel(image, image, 2);
        }
    } else {
        image = cv::imread(image_path);
        if (image.channels() == 3) {
            // Color to gray
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        }
    }
    if (image.type() != CV_32FC1) {
        // Byte to float
        image.convertTo(image_float, CV_32FC1, 1.0/255.0);
    } else {
        // Normalize floats
        cv::normalize(image, image_float,
                       0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
    }

    cv::namedWindow(WIND_NAME);

    if (argc > 2) {
        // Just show the result image if many command line arguments are specified
        printf("More than one command line argument specified, skipping UI and showing result...\n");
        update(image_float);
        cv::waitKey(0);
        cv::destroyWindow(WIND_NAME);
        return 0;
    }

    // Show initial image
    cv::imshow(WIND_NAME, image_float);
    // Add callbacks
    cv::setMouseCallback(WIND_NAME, mouse_handler, &image_float);

    // Create interface
    printf("Click on the image to start Fast Marching Method from that point\n"
           "Press r to view the initial image\n"
           "Press s to enable segmentation, then +- to adjust threshold\n"
           "Press keys 1-%d to switch to different weight map\n", fmm::weight::_WEIGHT_MAP_COUNT); 
    printf("Using weight map: %s\n", WEIGHT_MAP_NAMES[weight_map]); 

    while (true) {
        char k = cv::waitKey(0);
        if (k == 'q' || k == 27) break;
        if (seeds.empty()) continue; // Do not allow commands until user has clicked
        if (k == 'r') {
            cv::imshow(WIND_NAME, image_float);
            continue;
        } else if (k >= '1' && k <= '4') {
            weight_map = k - '1';
            printf("Using weight map: %s\n", WEIGHT_MAP_NAMES[weight_map]); 
        } else if (k == 's') {
            segmentation_enabled ^= 1;
            printf("Segmentation %sabled\n", segmentation_enabled ? "en" : "dis"); 
            if (segmentation_enabled) printf("Segmentation threshold: %f press +- to adjust\n", thresh); 
        } else if (segmentation_enabled && (k == '+' || k == '=' || k == '-')) {
            auto sign = -(2 * int(k== '-') - 1);
            thresh += thresh * 0.12f * sign;
            printf("Segmentation threshold: %f\n", thresh); 
        }
        update(image_float);
    }
    cv::destroyWindow(WIND_NAME);
    return 0;
}
