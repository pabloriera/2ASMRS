cmake_minimum_required(VERSION 3.15)
set(PROJECT_NAME "2ASRMS")

set(FORMATS Standalone)

file(STRINGS VERSION CURRENT_VERSION)
project(${PROJECT_NAME} VERSION ${CURRENT_VERSION})

set(CMAKE_XCODE_GENERATE_SCHEME OFF)

# Enable to build universal binaries on macOS, increasing build time
# This only affects local builds, GitHub actions always builds Universals
# set(CMAKE_OSX_ARCHITECTURES arm64 x86_64)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Support macOS down to High Sierra")

# Adds all the module sources so they appear correctly in the IDE
# Must be set before JUCE is added as a sub-dir (or any targets are made)
# https://github.com/juce-framework/JUCE/commit/6b1b4cf7f6b1008db44411f2c8887d71a3348889
set_property(GLOBAL PROPERTY USE_FOLDERS YES)

# Create a /Modules directory in the IDE with the JUCE Module code
option(JUCE_ENABLE_MODULE_SOURCE_GROUPS "Show all module sources in IDE projects" ON)

# JUCE is setup as a submodule in the /JUCE folder
# Locally, you'll need to run `git submodule update --init --recursive` once
# and `git submodule update --remote --merge` to keep it up to date
# On Github Actions, it's managed by actions/checkout

add_subdirectory(external/JUCE)
include_directories(external/json/include)

set(CMAKE_PREFIX_PATH external/libtorch/)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


juce_add_gui_app("${PROJECT_NAME}"
    COMPANY_NAME "Pablo Riera"                  # Specify the name of the plugin's author
    IS_SYNTH TRUE                               # Is this a synth or an effect?
    NEEDS_MIDI_INPUT TRUE                       # Does the plugin need midi input?
    NEEDS_MIDI_OUTPUT FALSE                     # Does the plugin need midi output?
    # IS_MIDI_EFFECT FALSE                      # Is this plugin a MIDI effect?
    # EDITOR_WANTS_KEYBOARD_FOCUS FALSE         # Does the editor need keyboard focus?
    # PLUGIN_MANUFACTURER_CODE CJSZ             # A four-character manufacturer id with at least one upper-case character
    PLUGIN_CODE P001                            # A unique four-character plugin id with exactly one upper-case character
                                                # GarageBand 10.3 requires the first letter to be upper-case, and the remaining letters to be lower-case
    FORMATS "${FORMATS}"                        # The formats to build. Other valid formats are: AAX Unity VST AU AUv3
    PRODUCT_NAME "${PROJECT_NAME}")                        # The name of the final executable, which can differ from the target name

juce_generate_juce_header("${PROJECT_NAME}")

target_compile_features("${PROJECT_NAME}" PRIVATE cxx_std_20)

set(SourceFiles
    Source/Autoencoder.h
    Source/Main.cpp
    Source/MainComponent.cpp
    Source/MainComponent.h)
target_sources("${PROJECT_NAME}" PRIVATE ${SourceFiles})

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release)
endif()

set(DONT_SET_USING_JUCE_NAMESPACE 0)

set_target_properties("${PROJECT_NAME}" PROPERTIES FOLDER "")

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/Source PREFIX "" FILES ${SourceFiles})

# juce_add_binary_data(Assets SOURCES pamplejuce.png)
# set_target_properties(Assets PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

foreach(target ${FORMATS} "All")
    if(TARGET ${PROJECT_NAME}_${target})
        set_target_properties(${PROJECT_NAME}_${target} PROPERTIES
            # Tuck the actual plugin targets into a folder where they won't bother us
            FOLDER "Targets"

            # MacOS only: Sets the default executable that Xcode will open on build
            # For this exact path to to work, manually build the AudioPluginHost.xcodeproj in the JUCE subdir
            XCODE_SCHEME_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/JUCE/extras/AudioPluginHost/Builds/MacOSX/build/Debug/AudioPluginHost.app"

            # Let us build the target in Xcode
            XCODE_GENERATE_SCHEME ON)
    endif()
endforeach()
# set_target_properties(Assets PROPERTIES FOLDER "Targets")

set(JUCE_DEPENDENCIES
    juce::juce_audio_utils
    juce::juce_dsp)

target_compile_definitions("${PROJECT_NAME}"
    PUBLIC
    # JUCE_WEB_BROWSER and JUCE_USE_CURL would be on by default, but you might not need them.
    JUCE_WEB_BROWSER=0  # If you remove this, add `NEEDS_WEB_BROWSER TRUE` to the `juce_add_plugin` call
    JUCE_USE_CURL=0     # If you remove this, add `NEEDS_CURL TRUE` to the `juce_add_plugin` call
    # JUCE_VST3_CAN_REPLACE_VST2=0
    )


target_link_libraries("${PROJECT_NAME}"
    PRIVATE
    #    Assets
        ${JUCE_DEPENDENCIES}
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
        "${TORCH_LIBRARIES}")

##### TESTING #####
# # Required for ctest (which is just easier for cross-platform CI)
# # include(CTest) does this too, but adds tons of targets we don't want
# # See: https://github.com/catchorg/Catch2/issues/2026
# # You could forgo ctest and call ./Tests directly from the build dir
# enable_testing()


# # "GLOBS ARE BAD" is brittle and silly UX, sorry CMake!
# # CONFIGURE_DEPENDS / Clion's CMake integration makes globbing absolutely fine
# file(GLOB_RECURSE TestFiles CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Tests/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/Tests/*.h")
# source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/Tests PREFIX "" FILES ${TestFiles})

# # Use Catch2 v3 on the devel branch
# Include(FetchContent)
# FetchContent_Declare(
#     Catch2
#     GIT_REPOSITORY https://github.com/catchorg/Catch2.git
#     GIT_PROGRESS TRUE
#     GIT_SHALLOW TRUE
#     GIT_TAG v3.1.0)
# FetchContent_MakeAvailable(Catch2) # find_package equivalent

# # Setup the test executable, again C++ 20 please
# add_executable(Tests ${TestFiles})
# target_compile_features(Tests PRIVATE cxx_std_20)

# # Our test executable also wants to know about our plugin code...
# target_include_directories(Tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Source)
# target_link_libraries(Tests PRIVATE Catch2::Catch2WithMain "${PROJECT_NAME}" ${JUCE_DEPENDENCIES})

# # Make an Xcode Scheme for the test executable so we can run tests in the IDE
# set_target_properties(Tests PROPERTIES XCODE_GENERATE_SCHEME ON)

# # Organize the test source in the Tests/ folder in the IDE
# source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/Tests PREFIX "" FILES ${TestFiles})

# # Load and use the .cmake file provided by Catch2
# # https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md
# # We have to manually provide the source directory here for now
# include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
# catch_discover_tests(Tests)

# Color our warnings and errors
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   add_compile_options (-fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
   add_compile_options (-fcolor-diagnostics)
endif ()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET aev1
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:example-app>)
# endif (MSVC)