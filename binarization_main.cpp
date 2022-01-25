#include <opencv2/opencv.hpp>
#include "image_processing/image_processing_unit.h"

struct AdditionalInfo {
  std::string input_path;
  std::string output_path;
  int levels;
};

void ParseArguments(int argc, char** argv, AdditionalInfo& info) {
  for (int i = 1; i < argc; i += 2) {
    if ("-i" == std::string(argv[i])) {
      info.input_path = std::string(argv[i + 1]);
    } else if ("-o" == std::string(argv[i])) {
      info.output_path = std::string(argv[i + 1]);
    } else if ("-l" == std::string(argv[i])) {
      info.levels = std::stoi(argv[i + 1]);
    }
  }
  if ((info.input_path.empty()) || (info.output_path.empty()) ||
      (info.levels < 2) || (info.levels > 4))
    throw std::runtime_error(
        "Use program:\n path/to/binarization_main -i path/to/gray/image -o "
        "path/to/output/image -l /number/of/classes");
}

// ToDo: const in class
int main(int argc, char** argv) {
  try {
    AdditionalInfo add_info;
    ParseArguments(argc, argv, add_info);
    cv::Mat img = cv::imread(add_info.input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("Cannot read input image");
    MultiClassOtsuUnit bin_unit(img, add_info.levels);
    cv::Mat bin;
    bin_unit.Process(img, bin);
    cv::cvtColor(bin, bin, cv::COLOR_GRAY2BGR);
    if (!cv::imwrite(add_info.output_path, bin))
      throw std::runtime_error("Cannot save image");
  } catch (std::exception& w) {
    std::cout << w.what();
  } catch (...) {
  }
}
