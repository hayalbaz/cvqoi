# CVQOI
## A QOI encoder/decoder for using with OpenCV in C++
This is a QOI implementation in C++ using OpenCV. It is currently WIP, the encoder is partly done but has bugs. The decoder will be implemented in the future. It expects the usual OpenCV format of BGR/A so no color conversion is needed. However it requires the bit depth of 8. Currently only dependency is boost::endian for making sure the byte order is big endian.

### Usage
This is a header only library. It is designed to be used with std::ostream and std::istream. Simply pass the Encoder/Decoder object to the stream for encoding/decoding.
```
cv::Mat mat;
//Do stuff with cv::Mat...

//To stream into a std::vector you can use boost::iostreams
namespace bstrs = boost::iostreams;
std::vector<char> encodedImage;
bstrs::back_insert_device<std::vector<char>> sink{encodedImage};
bstrs::stream<bstrs::back_insert_device<std::vector<char>>> os{sink};

//Or you can create std::ofstream to stream to a file
//std::ofstream os("myQoiFile.qoi", std::ios::binary)

if (mat.channels() == 4) {
    os << cvqoi::Encoder<true>(mat); //If cv::Mat has alpha channel need to set this template flag
}
else {
    os << cvqoi::Encoder<>(mat);
}
```