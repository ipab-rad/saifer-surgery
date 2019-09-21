cmake_minimum_required (VERSION 2.6)

link_libraries(apriltags)

add_executable(apriltags2 apriltags2.cpp)
pods_install_executables(apriltags2)

add_executable(apriltags_demo apriltags_demo.cpp Serial.cpp)
pods_install_executables(apriltags_demo)

add_executable(imu imu.cpp Serial.cpp)
pods_install_executables(imu)
