# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/matej/opencv_withoutcam

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/matej/opencv_withoutcam

# Include any dependencies generated for this target.
include CMakeFiles/Pick_and_place.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Pick_and_place.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Pick_and_place.dir/flags.make

CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o: CMakeFiles/Pick_and_place.dir/flags.make
CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o: pickandplace.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/matej/opencv_withoutcam/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o -c /home/matej/opencv_withoutcam/pickandplace.cpp

CMakeFiles/Pick_and_place.dir/pickandplace.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Pick_and_place.dir/pickandplace.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matej/opencv_withoutcam/pickandplace.cpp > CMakeFiles/Pick_and_place.dir/pickandplace.cpp.i

CMakeFiles/Pick_and_place.dir/pickandplace.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Pick_and_place.dir/pickandplace.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matej/opencv_withoutcam/pickandplace.cpp -o CMakeFiles/Pick_and_place.dir/pickandplace.cpp.s

CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o.requires:

.PHONY : CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o.requires

CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o.provides: CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o.requires
	$(MAKE) -f CMakeFiles/Pick_and_place.dir/build.make CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o.provides.build
.PHONY : CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o.provides

CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o.provides.build: CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o


# Object files for target Pick_and_place
Pick_and_place_OBJECTS = \
"CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o"

# External object files for target Pick_and_place
Pick_and_place_EXTERNAL_OBJECTS =

Pick_and_place: CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o
Pick_and_place: CMakeFiles/Pick_and_place.dir/build.make
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stitching3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_superres3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videostab3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_aruco3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bgsegm3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bioinspired3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ccalib3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_cvv3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dpm3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_face3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_fuzzy3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_hdf3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_img_hash3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_line_descriptor3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_optflow3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_reg3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_rgbd3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_saliency3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stereo3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_structured_light3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_surface_matching3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_tracking3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xfeatures2d3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ximgproc3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xobjdetect3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xphoto3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_shape3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_photo3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_datasets3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_plot3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_text3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dnn3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ml3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_video3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_calib3d3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_features2d3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_highgui3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videoio3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_viz3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_phase_unwrapping3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_flann3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgcodecs3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_objdetect3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgproc3.so.3.3.1
Pick_and_place: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_core3.so.3.3.1
Pick_and_place: CMakeFiles/Pick_and_place.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/matej/opencv_withoutcam/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Pick_and_place"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Pick_and_place.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Pick_and_place.dir/build: Pick_and_place

.PHONY : CMakeFiles/Pick_and_place.dir/build

CMakeFiles/Pick_and_place.dir/requires: CMakeFiles/Pick_and_place.dir/pickandplace.cpp.o.requires

.PHONY : CMakeFiles/Pick_and_place.dir/requires

CMakeFiles/Pick_and_place.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Pick_and_place.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Pick_and_place.dir/clean

CMakeFiles/Pick_and_place.dir/depend:
	cd /home/matej/opencv_withoutcam && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/matej/opencv_withoutcam /home/matej/opencv_withoutcam /home/matej/opencv_withoutcam /home/matej/opencv_withoutcam /home/matej/opencv_withoutcam/CMakeFiles/Pick_and_place.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Pick_and_place.dir/depend

