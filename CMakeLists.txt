cmake_minimum_required(VERSION 3.16)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS  ON)


# Debug
# set(smoke_testcase render)
# set(ASCEND_PRODUCT_TYPE Ascend910A)
# set(ASCEND_CORE_TYPE AiCore)    
# set(ASCEND_RUN_MODE cpu)
# set(ASCEND_INSTALL_PATH /usr/local/Ascend/ascend-toolkit/latest)
# Debug

set(CCE_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})
project(kernel_samples LANGUAGES CCE CXX)

add_subdirectory(cmake/cpu)
add_subdirectory(cmake/npu)