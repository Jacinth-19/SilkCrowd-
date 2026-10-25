# =============================================================================
#  SilkCrowd - real-time hair rendering with hair meshes (SIGGRAPH 2024)
#
#    git clone --recursive <repo> && cd SilkCrowd
#    cmake -B build -DCMAKE_BUILD_TYPE=Release
#    cmake --build build -j
#    ./build/SilkCrowd                       # Qt6 editor (falls back to the CLI)
#    ./build/SilkCrowd --screenshot out.png  # headless render (Vulkan or software)
#
#  The core library, the validation suite and the software reference renderer need
#  nothing but a C++17 compiler. Vulkan and Qt6 are picked up when present; the
#  binary always builds so that CI can run on hosts without a GPU.
# =============================================================================
cmake_minimum_required(VERSION 3.20)
project(SilkCrowd VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(SILKCROWD_BUILD_TESTS "Build the unit-test and validation binaries" ON)
option(SILKCROWD_WITH_QT "Build the Qt6 Vulkan editor window" ON)
option(SILKCROWD_STRICT "Treat warnings as errors" OFF)

add_library(silkcrowd_core STATIC
  src/core/assets.cpp
  src/core/catmull_rom.cpp
  src/core/hair_mesh.cpp
  src/core/hair_mesh_texture.cpp
  src/core/image.cpp
  src/core/lod.cpp
  src/core/root_placement.cpp
  src/core/sim.cpp
  src/core/strand_generator.cpp
  src/core/styling.cpp
  src/render/software_renderer.cpp
)
target_include_directories(silkcrowd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(silkcrowd_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
if(SILKCROWD_STRICT)
  target_compile_options(silkcrowd_core PRIVATE -Werror)
endif()
# Threads make the crowd passes scale with the core count.
find_package(Threads QUIET)
if(Threads_FOUND)
  target_link_libraries(silkcrowd_core PUBLIC Threads::Threads)
  target_compile_definitions(silkcrowd_core PUBLIC SILKCROWD_HAS_THREADS=1)
endif()

# -----------------------------------------------------------------------------
#  Shaders: GLSL -> SPIR-V. hair_common.glsl is an #include dependency.
# -----------------------------------------------------------------------------
set(SILKCROWD_SHADERS
  hair.task hair.mesh hair.frag hair_compute.comp hair_draw.vert
  head.vert head.frag fullscreen.vert composite.frag bloom.frag shadow.frag)

set(SILKCROWD_SHADER_OUT ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SILKCROWD_SHADER_OUT})

find_program(SILKCROWD_GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin)
find_program(SILKCROWD_GLSLANG glslangValidator HINTS $ENV{VULKAN_SDK}/bin)

set(SILKCROWD_SPV_FILES)
if(SILKCROWD_GLSLC OR SILKCROWD_GLSLANG)
  foreach(shader ${SILKCROWD_SHADERS})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader})
    set(spv ${SILKCROWD_SHADER_OUT}/${shader}.spv)
    # Task/mesh shaders need SPIR-V 1.4 (VK_EXT_mesh_shader); the others are fine at 1.0.
    if(shader MATCHES "\\.(task|mesh)$")
      set(extra --target-env=vulkan1.2)
    else()
      set(extra --target-env=vulkan1.1)
    endif()
    if(SILKCROWD_GLSLC)
      add_custom_command(OUTPUT ${spv}
        COMMAND ${SILKCROWD_GLSLC} ${extra} -O
                -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${src} -o ${spv}
        DEPENDS ${src} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/hair_common.glsl
        COMMENT "glslc ${shader}"
        VERBATIM)
    else()
      add_custom_command(OUTPUT ${spv}
        COMMAND ${SILKCROWD_GLSLANG} -V --vn silk_${shader} -o ${spv} ${src}
                -I${CMAKE_CURRENT_SOURCE_DIR}/shaders
        DEPENDS ${src} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/hair_common.glsl
        COMMENT "glslangValidator ${shader}"
        VERBATIM)
    endif()
    list(APPEND SILKCROWD_SPV_FILES ${spv})
  endforeach()
  add_custom_target(silkcrowd_shaders ALL DEPENDS ${SILKCROWD_SPV_FILES})
else()
  message(WARNING
    "No glslc/glslangValidator found: the shaders will not be compiled and the Vulkan "
    "renderer will refuse to start. Install the Vulkan SDK, or run the software "
    "renderer with --software.")
endif()

# -----------------------------------------------------------------------------
#  Vulkan (optional)
# -----------------------------------------------------------------------------
find_package(Vulkan QUIET)
set(SILKCROWD_GPU_SOURCES)
if(Vulkan_FOUND)
  message(STATUS "Vulkan ${Vulkan_VERSION} found: building the GPU renderer")
  add_library(silkcrowd_gpu STATIC
    src/gpu/vulkan_context.cpp
    src/gpu/hair_gpu_renderer.cpp)
  target_link_libraries(silkcrowd_gpu PUBLIC silkcrowd_core Vulkan::Vulkan)
  target_include_directories(silkcrowd_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(silkcrowd_gpu PUBLIC
    SILKCROWD_WITH_VULKAN=1
    SILKCROWD_SHADER_DIR="${SILKCROWD_SHADER_OUT}")
  if(TARGET silkcrowd_shaders)
    add_dependencies(silkcrowd_gpu silkrowd_shaders)
  endif()
  set(SILKCROWD_GPU_SOURCES silkcrowd_gpu)
endif()

# -----------------------------------------------------------------------------
#  Qt6 editor (optional)
# -----------------------------------------------------------------------------
set(SILKCROWD_QT_SOURCES)
if(SILKCROWD_WITH_QT)
  find_package(Qt6 6.2 QUIET COMPONENTS Widgets Gui)
  if(Qt6_FOUND AND Vulkan_FOUND)
    message(STATUS "Qt6 ${Qt6_VERSION} found: building the editor window")
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(SILKCROWD_QT_SOURCES src/app/qt_app.cpp)
    set(SILKCROWD_QT_LIBS Qt6::Widgets Qt6::Gui)
  else()
    message(STATUS "Qt6/Vulkan not found: the editor window is disabled (CLI still works)")
  endif()
endif()

# -----------------------------------------------------------------------------
#  Executables
# -----------------------------------------------------------------------------
add_executable(SilkCrowd src/app/main.cpp src/app/assets_export.cpp ${SILKCROWD_QT_SOURCES})
target_link_libraries(SilkCrowd PRIVATE silkrowd_core)
if(SILKCROWD_GPU_SOURCES)
  set_target_properties(SilkCrowd PROPERTIES CXX_STANDARD 17)
  target_link_libraries(SilkCrowd PRIVATE ${SILKCROWD_GPU_SOURCES})
endif()
if(SILKCROWD_QT_SOURCES)
  target_link_libraries(SilkCrowd PRIVATE ${SILKCROWD_QT_LIBS})
  target_compile_definitions(SilkCrowd PRIVATE SILKCROWD_WITH_QT=1)
endif()
# The shaders are looked up next to the binary as well as in the source tree.
add_custom_command(TARGET SilkCrowd POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${SILKCROWD_SHADER_OUT} $<TARGET_FILE_DIR:SilkCrowd>/shaders
  COMMENT "Copying SPIR-V shaders next to the executable")
add_custom_command(TARGET SilkCrowd POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/assets $<TARGET_FILE_DIR:SilkCrowd>/assets
  COMMENT "Copying hair mesh presets next to the executable")

if(SILKCROWD_BUILD_TESTS)
  add_executable(silkcrowd_tests tests/tests.cpp)
  target_link_libraries(silkcrowd_tests PRIVATE silkrowd_core)
  add_test(NAME unit_tests COMMAND silkrowd_tests)
  add_test(NAME validation COMMAND SilkCrowd --validate)
  enable_testing()
endif()

install(TARGETS SilkCrowd RUNTIME DESTINATION bin)
install(DIRECTORY ${SILKCROWD_SHADER_OUT}/ DESTINATION bin/shaders)
install(DIRECTORY assets/ DESTINATION bin/assets)
