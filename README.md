# YALL (Yet Another Learning Library)

YALL is a C++ library that implements many machine learning algorithms and attempts not to use company or platform specific libraries. That is, 
the goal of YALL is to usable on Windows, Linux, Mac, Raspberry Pi, etc. without a lot of installation and headaches. 

There are many great machine learning libraries out there that do many things, probably more efficiently than YALL. The primary purpose of this
library is for my own personal experience with a secondary purpose of being portable, useful, and efficient

An additional FYI, this is the first project I'm building using CMAKE. It's also acting as an introduction to this build system and enhancing my knowledge of linux linking/compiling. Excuse the note files all over the place...

## Dependencies

+ Armadillo (built with 9.860.1)
	+ Depends on (Ubuntu versions): libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev
	+ Install Instructions for all Platforms: http://arma.sourceforge.net/download.html
+ CMAKE
+ Gnuplot
	+ http://www.gnuplot.info/
	+ Installing on Linux: should be available in whatever package manager is used for your distribution as 'gnuplot'
	+ Installing on Windows
		+ https://stackoverflow.com/questions/47453376/how-to-install-gnuplot-on-windows
+ C++11
    + The library will might work with older versions of C++ but I'm compiling my tests with
        --std=c++11


## Save/Load Neural Network Weights

### File Structure
Line 1: Input size
Line 2: Output size
Line 3: Number of hidden layers (*h*)
Line 4: *h* values separater by a space to define the layer widths
Lines 5-(5+*h*): one line for each weight matrix associated with the *h* hidden layers



## Custom Neural Network Activation Functions

## Custom Neural Network Back Propagation Techniques
