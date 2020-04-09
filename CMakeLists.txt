cmake_minimum_required(VERSION 3.9)

# cmake config
project(YALL VERSION 1.0.1 DESCRIPTION "Yet Another Learning Library")
include(GNUInstallDirs)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

## create shared library (.so file) which is the source code for the project
add_library(YALL SHARED
	src/YALL/Models/NeuralNet.cc
	src/YALL/Models/DecisionTree.cc
	src/YALL/Models/LinearRegression.cc
	src/YALL/Utils/Statistics.cc
	src/YALL/Utils/DataTable.cc
)

## create list of includes (public headers)
## THese will be included in the programs using our program
set(PHEADERS 
	"include/YALL/Models.hpp"
	"include/YALL/Algorithms.hpp"
	"include/YALL/Utils.hpp"
	"include/YALL/Plot.hpp"
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
## A good way to thinkg of this is that we're adding the second part (after the PRIVATE/PUBLIC/etc.)
## as compiler flags in the Makefile with the first value as the target (YALL) which are just Makefile
## targets.
target_include_directories(YALL PRIVATE include)
target_include_directories(YALL PRIVATE src)
target_link_libraries(YALL PRIVATE ${ARMADILLO_LIBRARIES})
## e.g. add the flag -Wno-psabi to the Makefile target YALL
target_compile_options(YALL PRIVATE -Wno-psabi)

## Install the project shared library files and the PUBLIC HEADERS (files to be used by apps:).
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
install(DIRECTORY "include/YALL/Utils/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/YALL/Utils)
install(DIRECTORY "include/YALL/Plot/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/YALL/Plot)
