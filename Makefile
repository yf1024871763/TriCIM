install:
	git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure.git
	cd accelergy-timeloop-infrastructure

	# Install Accelergy
	make install_accelergy

	# Install the Timeloop Python Front-End
	pip3 install ./src/timeloopfe

	# Install Timeloop
	make install_timeloop
	
	cd booksim2/src && make
