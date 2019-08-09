/** Fast marching method sample project */
#include <string>
#include "fmm.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

constexpr char WIND_NAME[] = "Image";
const char* WEIGHT_MAP_NAMES[] = {"Identity",  "Gradient magnitude", "AbsDiff", "Laplacian"};
int weight_map = fmm::weight::ABSDIFF;
bool segmentation_enabled = false;
float thresh = 0.2;
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
    if (argc < 2 || argc > 3 || (argc >= 2 && strcmp(argv[1], "--help") == 0)) {
        fprintf(stderr, "Usage: fmm-demo image_path [(int 1-4) initial_weight_map]\n");
        return 0;
    }
    const std::string image_path = argv[1];
    if (argc > 2) weight_map = std::atoi(argv[2]);

    cv::Mat image = cv::imread(image_path), image_gray, image_float;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    assert(image_gray.type() == CV_8UC1);
    image_gray.convertTo(image_float, CV_32FC1, 1.0/255.0);
    cv::namedWindow(WIND_NAME);
    cv::imshow(WIND_NAME, image_float.clone());
    cv::setMouseCallback(WIND_NAME, mouse_handler, &image_float);

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
            cv::imshow(WIND_NAME, image_float.clone());
        } else if (k >= '1' && k <= '4') {
            weight_map = k - '1';
            printf("Using weight map: %s\n", WEIGHT_MAP_NAMES[weight_map]); 
        } else if (k == 's') {
            segmentation_enabled ^= 1;
            printf("Segmentation %sabled\n", segmentation_enabled ? "en" : "dis"); 
            if (segmentation_enabled) printf("Segmentation threshold: %f press +- to adjust\n", thresh); 
        } else if (segmentation_enabled && (k == '+' || k == '=' || k == '-')) {
            thresh -= 0.005 * (2 * int(k== '-') - 1);
            printf("Segmentation threshold: %f\n", thresh); 
        }
        update(image_float);
    }
    cv::destroyWindow(WIND_NAME);
    return 0;
}
