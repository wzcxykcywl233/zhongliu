# Get the GCC preinstalled image from Docker Hub
FROM gcc:11.4.0

# Fix issue with GLIBCXX
RUN export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

# Copy the current folder which contains C++ source code to the Docker image under /workspace
# COPY . /workspace

# Specify the working directory
# WORKDIR /workspace

# Use GCC to compile the cpp source file
# RUN g++ -o read-roi read-roi.cpp