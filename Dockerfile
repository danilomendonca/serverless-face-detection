FROM python:2-alpine3.6

# This dockerfile is based on https://github.com/julianbei/alpine-opencv-microimage
# and https://github.com/daveresbk/docker-thumbor/blob/master/alpine-opencv-microimage/Dockerfile.
# This dockerfile can build OpenCv with optimisations and adds support for Pillow.

RUN echo -e '@edgunity http://nl.alpinelinux.org/alpine/edge/community\n\
@edge http://nl.alpinelinux.org/alpine/edge/main\n\
@testing http://nl.alpinelinux.org/alpine/edge/testing\n\
@community http://dl-cdn.alpinelinux.org/alpine/edge/community'\
  >> /etc/apk/repositories

RUN apk update && apk upgrade

RUN apk add --update --no-cache \
  # --virtual .build-deps \
      build-base \
      openblas-dev \
      unzip \
      wget \
      cmake \

      #Intel® TBB, a widely used C++ template library for task parallelism'
      libtbb@testing  \
      libtbb-dev@testing   \

      # Wrapper for libjpeg-turbo
      libjpeg  \

      # accelerated baseline JPEG compression and decompression library
      libjpeg-turbo-dev \

      # Portable Network Graphics library
      libpng-dev \

      # A software-based implementation of the codec specified in the emerging JPEG-2000 Part-1 standard (development files)
      jasper-dev \

      # Provides support for the Tag Image File Format or TIFF (development files)
      tiff-dev \

      # Libraries for working with WebP images (development files)
      libwebp-dev \

      # A C language family front-end for LLVM (development files)
      clang-dev \

      linux-headers && \

      pip install numpy

ENV CC /usr/bin/clang
ENV CXX /usr/bin/clang++
ENV OPENCV_VERSION 3.4.2

RUN mkdir /opt && cd /opt && \
  wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
  unzip ${OPENCV_VERSION}.zip && \
  rm -rf ${OPENCV_VERSION}.zip && \
  cd /opt/opencv-${OPENCV_VERSION} && \
  mkdir build && \
  cd build && \
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D BUILD_PYTHON_SUPPORT=ON \
	-D BUILD_EXAMPLES=OFF \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D WITH_FFMPEG=NO \
	-D ENABLE_FAST_MATH=ON \
        -D WITH_TBB=ON \
        -D WITH_IPP=ON	\
        -D WITH_V4L=OFF  \
        -D ENABLE_AVX=ON \
        -D ENABLE_SSSE3=ON \
        -D ENABLE_SSE41=ON \
        -D ENABLE_SSE42=ON \
        -D ENABLE_POPCNT=ON \
        -D BUILD_TESTS=OFF \
	-D BUILD_PERF_TESTS=OFF \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D WITH_OPENEXR=NO .. && \
  make VERBOSE=1 && \
  make && \
  make install && \
  rm -rf /opt/opencv-${OPENCV_VERSION}

RUN rm -rf /var/cache/apk/*
RUN rm -rf /tmp/*

# Upgrade and install basic Python dependencies
RUN apk add --no-cache \
        bash \
        bzip2-dev \
        gcc \
        libc-dev \
        libxslt-dev \
        libxml2-dev \
        libffi-dev \
        linux-headers \
        openssl-dev \
        python-dev

# Install common modules for python
RUN pip install --no-cache-dir --upgrade pip setuptools six \
 && pip install --no-cache-dir \
        gevent==1.3.6 \
        flask==1.0.2 \
        beautifulsoup4==4.6.3 \
        httplib2==0.11.3 \
        kafka_python==1.4.3 \
        lxml==4.2.5 \
        python-dateutil==2.7.3 \
        requests==2.19.1 \
        scrapy==1.5.1 \
        simplejson==3.16.0 \
        virtualenv==16.0.0 \
        twisted==18.7.0 \
      	Pillow==5.3.0

ENV FLASK_PROXY_PORT 8080

# Add the action proxy
ADD https://raw.githubusercontent.com/apache/incubator-openwhisk-runtime-docker/master/core/actionProxy/actionproxy.py /actionProxy/actionproxy.py

# Add python runner
ADD https://raw.githubusercontent.com/apache/incubator-openwhisk-runtime-python/master/core/pythonAction/pythonrunner.py /pythonAction/pythonrunner.py

RUN mkdir -p /action

CMD ["/bin/bash", "-c", "cd pythonAction && python -u pythonrunner.py"]
