cmake_minimum_required(VERSION 3.10)

project(Tracking VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(nlohmann_json 3.7.0 REQUIRED)

set(CMAKE_BUILD_TYPE Debug)

set(SOURCES
    tracking_module.cpp
)

add_executable(Tracker ${SOURCES})

target_link_libraries(Tracker ${PCL_LIBRARIES})
target_link_libraries(Tracker ${OpenCV_LIBS})
target_link_libraries(Tracker fitpackpp)

target_link_libraries(Tracker nlohmann_json::nlohmann_json)

target_include_directories(Tracker PRIVATE 
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${Eigen3_spline_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
    ${fitpackpp_INCLUDE_DIRS}
)
