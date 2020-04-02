cmake_minimum_required(VERSION 3.9)

# cmake config
project(YALL VERSION 1.0.1 DESCRIPTION "Yet Another Learning Library")
include(GNUInstallDirs)

## create shared library (.so file) which is the source code for the project
add_library(YALL SHARED
	src/YALL/Models/NeuralNet.cc
	src/YALL/Models/DecisionTree.cc
)

## create list of includes (public headers)
## THese will be included in the programs using our program
set(PHEADERS 
	"include/YALL/Models.hpp"
	"include/YALL/Algorithms.hpp"
)

# some CMAKE config
# PHEADERS and PUBLIC_HEADER allow us to include the headers in our project, 
# i.e. they create the /usr/local/YALL/*.hpp files
set_target_properties(YALL PROPERTIES 
	VERSION ${PROJECT_VERSION}
	SOVERSION 1
	PUBLIC_HEADER "${PHEADERS}")

## Create the pkg-config file
configure_file(yall.pc.in yall.pc @ONLY)

## not 100% sure here
target_include_directories(YALL PRIVATE include)
target_include_directories(YALL PRIVATE src)

## Install the project shared library files and the PUBLIC HEADERS (files to be used by apps).
install(TARGETS YALL 
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/YALL
)

## builc the .pc pkg-config file and move it
install(FILES ${CMAKE_BINARY_DIR}/yall.pc
	DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

## Need to copy these files to the include directories to be includeed by the public headers.
install(DIRECTORY "include/YALL/Models/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/YALL/Models)
install(DIRECTORY "include/YALL/Algorithms/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/YALL/Algorithms)