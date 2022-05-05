cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_TOOLCHAIN_FILE C:/Users/the_s/Coding/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cvqoidev VERSION 0.1.0)

set(OpenCV_DIR)
set(OpenCV_DIR "C:/Users/the_s/Coding/vcpkg/installed/x86-windows/share/opencv")

find_package(OpenCV CONFIG REQUIRED)
find_package(Boost COMPONENTS system iostreams filesystem REQUIRED)

add_executable (cvqoitestmain src/main.cpp)
set_target_properties(cvqoitestmain PROPERTIES COMPILE_FLAGS "-m32" LINK_FLAGS "-m32")
target_include_directories(cvqoitestmain PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cvqoitestmain ${OpenCV_LIBS})

target_include_directories(cvqoitestmain PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(cvqoitestmain ${Boost_LIBRARIES})

target_include_directories(cvqoitestmain PRIVATE ../include)