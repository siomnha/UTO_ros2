cmake_minimum_required(VERSION 3.16)
project(ifds_gz_plugins)

find_package(ament_cmake REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-sim8 REQUIRED)

add_library(ifds_obstacle_oscillator SHARED src/obstacle_oscillator.cc)
target_compile_features(ifds_obstacle_oscillator PUBLIC cxx_std_17)
target_link_libraries(ifds_obstacle_oscillator
  PRIVATE
    gz-plugin2::gz-plugin2
    gz-sim8::gz-sim8
)

add_library(ifds_obstacle_path SHARED src/obstacle_path.cc)
target_compile_features(ifds_obstacle_path PUBLIC cxx_std_17)
target_link_libraries(ifds_obstacle_path
  PRIVATE
    gz-plugin2::gz-plugin2
    gz-sim8::gz-sim8
)

install(TARGETS ifds_obstacle_oscillator ifds_obstacle_path
  LIBRARY DESTINATION lib
)

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/ifds_gz_plugins.dsv.in")

ament_package()
