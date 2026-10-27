# -----------------------------------------------------------------------------
#  Android build of the SilkCrowd renderer (NativeActivity + Vulkan 1.1/1.3)
#
#  The same core library, shaders and HairGpuRenderer are compiled here; only the window
#  system glue (android_native_app_glue, ANativeWindow -> VkSurfaceKHR, swapchain) is
#  Android specific. The mesh-shader path is used when the driver exposes
#  VK_EXT_mesh_shader, otherwise HairGpuRenderer falls back to the compute shader of
#  Section 3.2 (Option A), which is the Adreno path the paper targets.
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.22)
# C as well as CXX: native_app_glue is a .c file, and CMake refuses to configure a target
# whose sources are in a language the project has not enabled
# ("Cannot determine link language for target glue") - that killed the first APK build.
project(silkcrowd_android LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# <repo>/android/app/src/main/cpp, so the repository root is five levels up (this used to
# be four, which resolved to <repo>/android and made every shared source a missing file).
set(SILKCROWD_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

add_library(native_app_glue STATIC
  ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)
target_include_directories(native_app_glue PUBLIC
  ${ANDROID_NDK}/sources/android/native_app_glue)

add_library(silkcrowd SHARED
  android_main.cpp
  ${SILKCROWD_ROOT}/src/core/assets.cpp
  ${SILKCROWD_ROOT}/src/core/catmull_rom.cpp
  ${SILKCROWD_ROOT}/src/core/gltf.cpp
  ${SILKCROWD_ROOT}/src/core/hair_mesh.cpp
  ${SILKCROWD_ROOT}/src/core/hair_mesh_texture.cpp
  ${SILKCROWD_ROOT}/src/core/image.cpp
  ${SILKCROWD_ROOT}/src/core/image_decode.cpp
  ${SILKCROWD_ROOT}/src/core/lod.cpp
  ${SILKCROWD_ROOT}/src/core/root_placement.cpp
  ${SILKCROWD_ROOT}/src/core/sim.cpp
  ${SILKCROWD_ROOT}/src/core/strand_generator.cpp
  ${SILKCROWD_ROOT}/src/core/styling.cpp
  ${SILKCROWD_ROOT}/src/render/software_renderer.cpp
  ${SILKCROWD_ROOT}/src/app/scenes.cpp
  ${SILKCROWD_ROOT}/src/gpu/vulkan_context.cpp
  ${SILKCROWD_ROOT}/src/gpu/hair_gpu_renderer.cpp)

# Same include layout as the desktop target: silkcrowd/*.h from include/, "gpu/..." and
# "scenes.h" from src/, stb_image from third_party/stb (image_decode.cpp needs it).
target_include_directories(silkcrowd PRIVATE
  ${SILKCROWD_ROOT}/include
  ${SILKCROWD_ROOT}/src
  ${SILKCROWD_ROOT}/src/app
  ${SILKCROWD_ROOT}/third_party/stb
  ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(silkcrowd PRIVATE
  SILKCROWD_WITH_VULKAN=1
  VK_USE_PLATFORM_ANDROID_KHR=1
  SILKCROWD_SHADER_DIR="shaders")
target_link_libraries(silkcrowd
  native_app_glue vulkan android log EGL GLESv2)

# The SPIR-V modules are placed in assets/shaders by the CI step; at runtime they are
# copied to the app's private directory (assets are not fopen()-able) and the process
# working directory is set there, which is where loadShader() looks.
set_target_properties(silkcrowd PROPERTIES OUTPUT_NAME "silkcrowd")

# native_app_glue provides ANativeActivity_onCreate from a static library, so the entry
# point has to be marked as used or the linker drops it and the Activity fails to start.
target_link_options(silkcrowd PRIVATE "-u ANativeActivity_onCreate")
