FROM python:3.7

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy

WORKDIR /
ENV OPENCV_VERSION="4.1.1"
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
    && unzip opencv.zip \
    && unzip opencv_contrib.zip \
    && mv opencv-${OPENCV_VERSION} opencv \ 
    && mv opencv_contrib-${OPENCV_VERSION} opencv_contrib \ 
    && cd opencv \
    && mkdir build \ 
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$(python3.7 -c "import sys; print(sys.prefix)") \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DINSTALL_C_EXAMPLES=OFF \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
    -DPYTHON_EXECUTABLE=$(which python3.7) \
    -DPYTHON_INCLUDE_DIR=$(python3.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_PACKAGES_PATH=$(python3.7 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    .. \
    # && cmake -DBUILD_TIFF=ON \
    #   -DBUILD_opencv_java=OFF \
    #   -DWITH_CUDA=OFF \
    #   -DWITH_OPENGL=ON \
    #   -DWITH_OPENCL=ON \
    #   -DWITH_IPP=ON \
    #   -DWITH_TBB=ON \
    #   -DWITH_EIGEN=ON \
    #   -DWITH_V4L=ON \
    #   -DBUILD_TESTS=OFF \
    #   -DBUILD_PERF_TESTS=OFF \
    #   -DCMAKE_BUILD_TYPE=RELEASE \
    #   -DCMAKE_INSTALL_PREFIX=$(python3.7 -c "import sys; print(sys.prefix)") \
    #   -DPYTHON_EXECUTABLE=$(which python3.7) \
    #   -DPYTHON_INCLUDE_DIR=$(python3.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    #   -DPYTHON_PACKAGES_PATH=$(python3.7 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    #   .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm /opencv.zip \
    && rm /opencv_contrib.zip \
    && rm -r /opencv \
    && rm -r /opencv_contrib

RUN ln -s \
    /usr/local/python/cv2/python-3.7/cv2.cpython-37m-x86_64-linux-gnu.so \
    /usr/local/lib/python3.7/site-packages/cv2.so


RUN wget -O app.zip https://github.com/maxp36/testopencv/archive/master.zip \
    && unzip app.zip \
    && rm app.zip

WORKDIR /app

ENTRYPOINT [ "python", "app.py" ]
