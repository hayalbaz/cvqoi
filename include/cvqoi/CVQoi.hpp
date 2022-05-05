#pragma once
#include <array>
#include <boost/endian/conversion.hpp>
#include <boost/endian/detail/order.hpp>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <ostream>
#include <stdexcept>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <vector>
#include <boost/endian.hpp>
#include <boost/endian/buffers.hpp>

//#define CVQOI_ASSERT_NO_CONSECUTIVE_INDEX

#ifdef CVQOI_ASSERT_NO_CONSECUTIVE_INDEX
#include <boost/optional.hpp>
#endif

namespace cvqoi
{
    template<bool hasAlpha>
    using PixelType = std::conditional_t<hasAlpha, cv::Vec<uint8_t, 4>, cv::Vec<uint8_t, 3>>;
    template<bool hasAlpha>
    using SignedPixelType = std::conditional_t<hasAlpha, cv::Vec<int8_t, 4>, cv::Vec<int8_t, 3>>;

    namespace qoi {
        constexpr std::array<char, 4> MAGIC{'q', 'o', 'i', 'f'};
        constexpr std::array<char, 8> EOS{0, 0, 0, 0, 0, 0, 0, 1};
    }

    namespace luma {
        namespace green {
            constexpr int8_t UPPER_RANGE = 32;
            constexpr int8_t LOWER_RANGE = -33;
            constexpr int8_t BIAS = 32;
        }
        namespace red_blue {
            constexpr int8_t UPPER_RANGE = 8;
            constexpr int8_t LOWER_RANGE = -9;
            constexpr int8_t BIAS = 8;
        }
        constexpr uint8_t TAG = 0x80;
        using chunk = std::array<uint8_t, 2>;
    }
    
    namespace diff {
        constexpr int8_t UPPER_RANGE = 2;
        constexpr int8_t LOWER_RANGE = -3;
        constexpr int8_t BIAS = 2;
        constexpr uint8_t TAG = 0x40;
        using chunk = uint8_t;
    }

    namespace rgb {
        constexpr uint8_t TAG = 0xfe;
        using chunk = std::array<uint8_t, 4>;
    }
    
    namespace rgba {
        constexpr uint8_t TAG = 0xff;
        using chunk = std::array<uint8_t, 5>;
    }

    namespace index {
        constexpr uint8_t TAG = 0x00;
        using chunk = uint8_t;
    }

    namespace run {
        constexpr uint8_t TAG = 0xc0;
        constexpr uint8_t LOWER_RANGE = 0;
        constexpr uint8_t UPPER_LIMIT = 62;
        constexpr uint8_t BIAS = 1;
        using chunk = uint8_t;
    }

    template<typename Pixel,
             typename SignedPixel,
             bool hasAlpha = false>
    struct util {
        static bool less(const SignedPixel &left, const int8_t &right) {
            return (blue(left) < right) && (green(left) < right) && (red(left) < right);
        }

        static bool greater(const SignedPixel &left, const int8_t &right) {
            return (blue(left) > right) && (green(left) > right) && (red(left) > right);
        }

        template<typename T>
        static const char* toChar(const T& t) {
            return reinterpret_cast<const char*>(&t);
        }

        static bool isInDiffRange(const SignedPixel &dp) {
            return less(dp, diff::UPPER_RANGE) && greater(dp, diff::LOWER_RANGE);
        }

        static bool isInLumaRange(const SignedPixel &dp, SignedPixel &lumaDp) {
            bool res{false};
            auto dgInRange = green(dp) < luma::green::UPPER_RANGE && green(dp) > luma::green::LOWER_RANGE;
            if (dgInRange) {
                green(lumaDp) = green(dp);
                blue(lumaDp) = blue(dp) - green(dp);
                red(lumaDp) = red(dp) - green(dp);

                auto dr_dgInRange = red(lumaDp) < luma::red_blue::UPPER_RANGE 
                                    && red(lumaDp) > luma::red_blue::LOWER_RANGE;
                auto db_dgInRange = blue(lumaDp) < luma::red_blue::UPPER_RANGE 
                                    && blue(lumaDp) > luma::red_blue::LOWER_RANGE; 

                res = dr_dgInRange && db_dgInRange;
            }
            return res;
        }

        // Below are accessor functions to indivual values of a Pixel
        // Done like this to easliy template const vs. non-const versions.
        template<typename T>
        static auto& blue(T &p) {
            return p[0];
        }

        template<typename T>
        static auto& green(T &p) {
            return p[1];
        }

        template<typename T>
        static auto& red(T &p) {
            return p[2];
        } 

        template<typename T>
        static auto& alpha(T &p) {
            return p[3];
        }

        static int hash(const Pixel &p) {
            int val =  (blue(p) * 7) + (green(p) * 5) + (red(p) * 3);
            if (hasAlpha) {
                val += alpha(p) * 11;
            }
            return val % 64;
        }

        template<typename T>
        struct is_std_array : std::false_type {};

        template<typename T, size_t N>
        struct is_std_array<std::array<T, N>> : std::true_type {};

        template<typename T,
                typename = std::enable_if_t<is_std_array<T>::value>>
        static void writeArrayToStream(const T &t, std::ostream &os) {
            static_assert(sizeof(typename T::value_type) == 1
                            , "Only arrays with elements that has sizeof 1 are allowed for endianness reasons.");
            os.write(reinterpret_cast<const char*>(t.data()), t.size() * sizeof(typename T::value_type));
        }

        template<typename T,
                typename = std::enable_if_t<std::is_pod_v<T>>>
        static void writeToStream(const T &t, std::ostream &os) {
            os.write(util::toChar(boost::endian::native_to_big(t)), sizeof(T));
        }
    };

    template<bool hasAlpha = false,
             typename Pixel = PixelType<hasAlpha>,
             typename SignedPixel = SignedPixelType<hasAlpha>>
    class Encoder 
    {
        using util = util<Pixel, SignedPixel, hasAlpha>;
    public:
        Encoder(const cv::Mat &mat) : mat(mat) {
            assert(((mat.channels() == 3 && !hasAlpha) || (mat.channels() == 4 && hasAlpha)) 
                    && "cv::Mat must have 3 or 4 channels.");
            assert(mat.depth() ==  CV_8U && "cv::Mat must have depth of 8 bits.");

            if (hasAlpha) {
                util::alpha(previousPixel) = 255;
                arr.fill({0, 0, 0, 0});
            }
            else {
                arr.fill({0, 0, 0});
            }
        }

        friend std::ostream& operator<<(std::ostream &os, const Encoder &e) {
            e.header(os);
            e.encodeImage(os);
            e.markEnd(os);
            return os;
        }

    private:
        void header(std::ostream &os) const {
            if (mat.rows > std::numeric_limits<uint32_t>::max() || mat.cols > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("One of the image dimensions is larger than the supported maximum size(32-bit)");
            }
            uint32_t width = mat.cols;
            uint32_t height = mat.rows;
            uint8_t channels = mat.channels();
            uint8_t colorspace = 1;
            util::writeArrayToStream(qoi::MAGIC, os);
            util::writeToStream(width, os);
            util::writeToStream(height, os);
            util::writeToStream(channels, os);
            util::writeToStream(colorspace, os);
        }

        void encodeImage(std::ostream &os) const {
            for (int r = 0; r < mat.rows; ++r) {
                auto *currentRow = mat.ptr<Pixel>(r);
                for (int c = 0; c < mat.cols; ++c) {
                    auto &currentPixel = currentRow[c];
                    #ifdef CVQOI_ASSERT_NO_CONSECUTIVE_INDEX
                    boost::optional<std::pair<uint8_t, uint8_t>> currentTag{boost::none};
                    //std::cout << "currentPixel: " << currentPixel << ", ";
                    //std::cout << "previousPixel: " << previousPixel << std::endl;
                    #endif

                    if (currentPixel == previousPixel) {
                        ++runningPixCnt;
                        if (runningPixCnt == run::UPPER_LIMIT || isLastPixel({r, c})) {
                            runningPixCnt -= run::BIAS;
                            util::writeToStream(runChunk(), os);
                            runningPixCnt = 0;
                            #ifdef CVQOI_ASSERT_NO_CONSECUTIVE_INDEX
                            currentTag.emplace(std::make_pair(run::TAG, 255));
                            #endif
                        }
                    }
                    else {
                        if (runningPixCnt > run::LOWER_RANGE) {
                            runningPixCnt -= run::BIAS;
                            util::writeToStream(runChunk(), os);
                            runningPixCnt = 0;
                        }

                        auto pairIsSeenBefore = isSeenBefore(currentPixel);
                        auto &seenBefore = pairIsSeenBefore.first;
                        auto &arrayIdx = pairIsSeenBefore.second;
                        if (seenBefore) {
                            util::writeToStream(indexChunk(arrayIdx), os);
                            #ifdef CVQOI_ASSERT_NO_CONSECUTIVE_INDEX
                            currentTag.emplace(std::make_pair(index::TAG, arrayIdx));
                            #endif
                        }
                        else {
                            arr[arrayIdx] = currentPixel;

                            SignedPixel dp, lumaDp;
                            util::blue(dp) = util::blue(currentPixel) - util::blue(previousPixel);
                            util::green(dp) = util::green(currentPixel) - util::green(previousPixel);
                            util::red(dp) = util::red(currentPixel) - util::red(previousPixel);
                            if (hasAlpha) {
                                util::alpha(dp) = util::alpha(currentPixel) - util::alpha(previousPixel);
                            }
                            if (hasAlpha && util::alpha(dp) != 0) {
                                util::writeArrayToStream(rgbaChunk(currentPixel), os);
                                #ifdef CVQOI_ASSERT_NO_CONSECUTIVE_INDEX
                                currentTag.emplace(std::make_pair(rgba::TAG, 255));
                                #endif
                            }
                            else {
                                if (util::isInDiffRange(dp)) {
                                    util::blue(dp) += diff::BIAS;
                                    util::green(dp) += diff::BIAS;
                                    util::red(dp) += diff::BIAS;
                                    util::writeToStream(diffChunk(dp), os);
                                    #ifdef CVQOI_ASSERT_NO_CONSECUTIVE_INDEX
                                    currentTag.emplace(std::make_pair(diff::TAG, 255));
                                    #endif
                                }
                                else if (util::isInLumaRange(dp, lumaDp)) {
                                    util::blue(lumaDp) += luma::red_blue::BIAS;
                                    util::green(lumaDp) += luma::green::BIAS;
                                    util::red(lumaDp) += luma::red_blue::BIAS;
                                    util::writeArrayToStream(lumaChunk(lumaDp), os);
                                    #ifdef CVQOI_ASSERT_NO_CONSECUTIVE_INDEX
                                    currentTag.emplace(std::make_pair(luma::TAG, 255));
                                    #endif
                                }
                                else {
                                    util::writeArrayToStream(rgbChunk(currentPixel), os);
                                    #ifdef CVQOI_ASSERT_NO_CONSECUTIVE_INDEX
                                    currentTag.emplace(std::make_pair(rgb::TAG, 255));
                                    #endif
                                }
                            }
                        }
                        previousPixel = currentPixel;
                    }
                    #ifdef CVQOI_ASSERT_NO_CONSECUTIVE_INDEX
                    //std::cout << "currentTag: " << (currentTag.has_value() ? +currentTag.value().first : -1) << ", ";
                    //std::cout << "previousTag: " << (previousTag.has_value() ? +previousTag.value().first : -1) << std::endl;
                    assert(!(currentTag.has_value() && previousTag.has_value()
                             && previousTag.value().first == index::TAG && currentTag.value().first == index::TAG
                             && previousTag.value().second == currentTag.value().second) 
                             && "Cannot emit two index tags in a row for the same index!");
                    previousTag = currentTag;
                    #endif
                }
            }
        }

        void markEnd(std::ostream &os) const {
            util::writeArrayToStream(qoi::EOS, os);
        }

        bool isLastPixel(const cv::Point2i &p) const {
            return p.x == mat.rows - 1 && p.y == mat.cols - 1;
        }

        std::pair<bool, int> isSeenBefore(const Pixel &p) const {
            auto currentPixelHash = util::hash(p);
            return std::make_pair(arr[currentPixelHash] == p, currentPixelHash);
        }

        index::chunk indexChunk(uint8_t idx) const {
            assert(idx < 64);
            return index::TAG | idx;
        }

        diff::chunk diffChunk(const Pixel &dp) const {
            assert(util::red(dp) < 4);
            assert(util::blue(dp) < 4);
            assert(util::green(dp) < 4);
            if (hasAlpha) {
                assert(util::alpha(dp) == 0);
            }
            return diff::TAG | ((util::red(dp) << 4) | (util::green(dp) << 2) | util::blue(dp));
        }

        luma::chunk lumaChunk(const Pixel &dp) const {
            assert(util::red(dp) < 16);
            assert(util::blue(dp) < 16);
            assert(util::green(dp) < 64);
            if (hasAlpha) {
                assert(util::alpha(dp) == 0);
            }
            luma::chunk ch;
            ch[0] = luma::TAG | util::green(dp);
            ch[1] = (util::red(dp) << 4) | util::blue(dp);
            return ch;
        }

        run::chunk runChunk() const {
            assert(runningPixCnt < 63);
            return run::TAG | runningPixCnt;
        }

        rgba::chunk rgbaChunk(const Pixel &currentPixel) const {
            rgba::chunk ch;
            ch[0] = rgba::TAG;
            ch[1] = util::red(currentPixel);
            ch[2] = util::green(currentPixel);
            ch[3] = util::blue(currentPixel);
            ch[4] = util::alpha(currentPixel);
            return ch;
        }

        rgb::chunk rgbChunk(const Pixel &currentPixel) const {
            rgb::chunk ch;
            ch[0] = rgb::TAG;
            ch[1] = util::red(currentPixel);
            ch[2] = util::green(currentPixel);
            ch[3] = util::blue(currentPixel);
            return ch;
        }

    private:
        const cv::Mat mat;
        mutable std::array<Pixel, 64> arr{};
        mutable Pixel previousPixel{};
        mutable uint8_t runningPixCnt{};
        #ifdef CVQOI_ASSERT_NO_CONSECUTIVE_INDEX
        mutable boost::optional<std::pair<uint8_t, uint8_t>> previousTag{boost::none};
        #endif
    };
};