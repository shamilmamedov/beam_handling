cmake_minimum_required(VERSION 3.4)

# Set the project name and version
project(myproject)

# Path variables
set(CASADI_DIR /home/shamil/Desktop/phd/code/casadi)
set(LIBFRANKA_COMMON_DIR ${LIBFRANKA_DIR}/common)

# Append path of the Franka library
list(APPEND CMAKE_PREFIX_PATH ${CASADI_DIR})
list(APPEND CMAKE_MODULE_PATH ${CASADI_DIR}/cmake)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Dependencies
find_package(pinocchio REQUIRED)
find_package(casadi REQUIRED)

# Adding libraries
include_directories(${CASADI_DIR}/include)

# Add executables
add_executable(generate_kinematics_fcns src/generate_kinematics_fcns.cpp)

# Link libraries
target_link_libraries(generate_kinematics_fcns pinocchio::pinocchio casadi)