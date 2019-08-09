## Fast marching method

- To use in a project, include fmm.hpp and use `fmm::fmm<image_data_type>(image, seeds, weight_map_type[, image_segmentation_threshold])`

- This is header-only and you do not need to build this, unless you want to build the sample program.

- All functions are templated to be OpenCV Mat compatible, also includes bare-bones Image struct which allows you to use any row-major contiguous matrix type: `Image<filed_type>(rows, cols, data_ptr)`

## License

Apache 2.0
