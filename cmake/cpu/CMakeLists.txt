# cpu
if (NOT DEFINED ENV{CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH ${ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/cmake)
endif()

find_package(tikicpulib REQUIRED)

file(GLOB SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

add_executable(${smoke_testcase}_cpu
    ${SRC_FILES}
)

target_include_directories(${smoke_testcase}_cpu PRIVATE
    ${ASCEND_INSTALL_PATH}/acllib/include
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(${smoke_testcase}_cpu PRIVATE
    tikicpulib::${ASCEND_PRODUCT_TYPE}
    ascendcl
)

target_compile_options(${smoke_testcase}_cpu PRIVATE
    -g
)

target_compile_definitions(${smoke_testcase}_cpu PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

set_target_properties(${smoke_testcase}_cpu PROPERTIES
    OUTPUT_NAME ${smoke_testcase}_${ASCEND_RUN_MODE}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
