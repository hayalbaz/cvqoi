#include <boost/filesystem/path.hpp>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <ostream>
#include <string>
#include <vector>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/stream.hpp>
#include "cvqoi/CVQoi.hpp"

#define QOI_IMPLEMENTATION
#include "qoi.h"

namespace bfs = boost::filesystem;
namespace bstrs = boost::iostreams;

int main(int argc, const char **argv) {
    if (argc < 2) {
        std::cout << "Please give path to the qoi_test_images as an argument to the program!" << std::endl;
        return 1;
    }
    std::vector<bfs::path> pngFiles, qoiFiles;
    bfs::path p(argv[1]);
    try {
        if (bfs::exists(p)) {
            if (bfs::is_regular_file(p)) {
                std::cout << "Input must be the qoi_test_images directory!" << std::endl;
                return 1;
            }
            else if (bfs::is_directory(p)) {
                for (auto& x : bfs::directory_iterator(p)) {
                    if (x.path().extension() == ".png") {
                        pngFiles.push_back(x.path());
                    }
                }
                for (auto& x : bfs::directory_iterator(p)) {
                    if (x.path().extension() == ".qoi") {
                        qoiFiles.push_back(x.path());
                    }
                }

                if (pngFiles.empty()) {
                    std::cout << "There are no png files inside the qoi_test_images directory!" << std::endl;
                    return 1;
                }
                if (qoiFiles.empty()) {
                    std::cout << "There are no qoi files inside the qoi_test_images directory!" << std::endl;
                    return 1;
                }
                if (pngFiles.size() != qoiFiles.size()) {
                    std::cout << "Number of qoi and png files do not match!" << std::endl;
                    return 1;
                }

                std::sort(pngFiles.begin(), pngFiles.end());
                std::sort(qoiFiles.begin(), qoiFiles.end());
                
                std::cout << "PNG files: " << std::endl;
                for (auto& x : pngFiles) {
                    std::cout << "    " << x.filename() << std::endl;
                }
                std::cout << "QOI files: " << std::endl;
                for (auto& x : qoiFiles) {
                    std::cout << "    " << x.filename() << std::endl;
                }
            }
            else {
                std::cout << p << " exists, but is not a regular file or directory" << std::endl;
            }
        }
        else {
            std::cout << p << " does not exist" << std::endl;
        }
    }
    catch (const bfs::filesystem_error& ex) {
        std::cout << ex.what() << std::endl;
    }

    std::vector<cv::Mat> pngImages;
    pngImages.reserve(pngFiles.size());
    for (auto &p : pngFiles) {
        pngImages.push_back(cv::imread(p.string(), cv::IMREAD_UNCHANGED));
    }
    std::cout << "Loaded in png files." << std::endl;

    std::vector<std::vector<char>> qoiImages;
    qoiImages.reserve(qoiFiles.size());
    for (auto &p : qoiFiles) {
        std::ifstream ifs(p.string(), std::ios::in | std::ios::binary);
        if (!ifs.is_open()) {
            std::cout << "Could not open " << p << " to read the qoi file!" << std::endl;
            return 1;
        }
        qoiImages.push_back(std::vector<char>(std::istreambuf_iterator<char>(ifs), 
                                                    std::istreambuf_iterator<char>()));
    }
    std::cout << "Loaded in qoi files." << std::endl;

    /*for (auto &q : qoiFiles) {
        qoi_desc qd;
        int outLen;
        auto *decodedImg = qoi_read(q.string().c_str(), &qd, 0);
        qoi_encode(decodedImg, &qd, &outLen);
    }*/

    for (std::size_t i = 0; i < pngImages.size(); ++i) {
        std::ofstream os(qoiFiles[i].string() + ".test");
        assert(os.is_open());
        if (pngImages[i].channels() == 4) {
            os << cvqoi::Encoder<true>(pngImages[i]);
        }
        else {
            os << cvqoi::Encoder<>(pngImages[i]);
        }
    }

    std::vector<std::vector<char>> cvQoiImages(pngFiles.size());
    for (std::size_t i = 0; i < pngImages.size(); ++i) {
        bstrs::back_insert_device<std::vector<char>> sink{cvQoiImages[i]};
        bstrs::stream<bstrs::back_insert_device<std::vector<char>>> os{sink};
        if (pngImages[i].channels() == 4) {
            os << cvqoi::Encoder<true>(pngImages[i]);
        }
        else {
            os << cvqoi::Encoder<>(pngImages[i]);
        }
    }

    std::cout << "Successfully encoded all PNG Images using CVQoi" << std::endl;

    for (std::size_t i = 0; i < qoiImages.size(); ++i) {
        std::cout << "File Name: " << qoiFiles[i].filename() << ", ";
        std::cout << "Reference QOI size: " << qoiImages[i].size() << ", ";
        std::cout << "CVQoi size: " << cvQoiImages[i].size() << std::endl;
    }

    for (std::size_t i = 0; i < qoiImages.size(); ++i) {
        qoi_desc qd;
        auto *decodedImg = qoi_decode(cvQoiImages[i].data(), cvQoiImages[i].size(), &qd, 0);
        if (decodedImg == nullptr) {
            std::cout << "Could not decode " << pngFiles[i].filename() << std::endl;
            continue;
        }
        std::cout << "File Name: " << pngFiles[i].filename() << ", ";
        std::cout << "Decoded channels: " << +qd.channels << ", Actual channels: " << pngImages[i].channels() << ", ";
        std::cout << "Decoded width: " << qd.width << ", Actual width: " << pngImages[i].cols << ", ";
        std::cout << "Decoded height: " << qd.height << ", Actual height: " << pngImages[i].rows << std::endl;

        cv::Mat mat(qd.height, qd.width, qd.channels == 3 ? CV_8UC3 : CV_8UC4, decodedImg);
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
        cv::imshow("Original Image", pngImages[i]);
        cv::imshow("Encoded/Decoded Image", mat);
        cv::waitKey();
        free(decodedImg);
    }
    return 0;
}