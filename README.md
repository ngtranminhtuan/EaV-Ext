# 1.Install dependencies
    sudo apt install libsrtp2-dev
    sudo apt-get install -y libvpx-dev
    sudo apt install libopus-dev
    sudo apt-get install python3-tk 
    pip3 install -r requirements.txt

# 2. Install tensorflow
    sudo apt-get update
    sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
    sudo apt-get install python3-pip
    sudo pip3 install -U pip testresources setuptools==49.6.0
    sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
    sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow==1.15.5+nv21.2
    
# 3.Generate HTTPS Cert
    cd Streaming
    openssl genrsa -out cert.key 2048
    openssl req -x509 -new -nodes -key cert.key -sha256 -days 1825 -out cert.pem

# 4. Config system
    - Open config.ini file and replace following items:
        + SELECT_COUNTING_ZONE (0/1)        : Init counting zones, specify 1 if you want replace with new counting zone, then specify 0 and re-run application.
        + CAM_MODE             (0/1)        : Specify 0 if you want use video mode, 1 if you want camera mode.
        + VIDEO_PATH           (PATH)       : Enter video path in video mode.
        + OUTPUT_DIR           (DIR)        : Enter video dir to save output video.
        + DOOR_POSITION        (TOP/BOTTOM) : Door position's top or bottom.


# 5. Build aiortc lib
    sudo apt install yasm libvpx. libx264. cmake libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libsrtp2-dev libpython3-dev python3-numpy
    wget https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/ffmpeg/7:4.2.4-1ubuntu0.1/ffmpeg_4.2.4.orig.tar.xz
    tar -xf ffmpeg_4.2.4.orig.tar.xz
    cd ffmpeg-4.2.4
    ./configure --disable-static --enable-shared --disable-doc
    make
    sudo make install
    sudo ldconfig
    sudo pip3 install aiortc

# 6. Build pyrealsense2 lib
    wget https://github.com/IntelRealSense/librealsense/archive/refs/tags/v2.48.0.zip
    unzip v2.48.0.zip
    cd librealsense-2.48.0
    mkdir -p build && cd build
    cmake ../ -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true -DBUILD_WITH_CUDA:bool=true
    make -j8
    sudo make install
    cd ..
    echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2' >> ~/.bashrc
    source ~/.bashrc
    ./scripts/setup_udev_rules.sh

# 7. Run application
    Method 1: Run seperate applications.
        - Run AI application:
            python3 app.py
        - Run streaming server:
            cd Streaming
            python3 app.py
    Method 2: Run all applications.
        ./run_app.sh
    #Note: cert.pem and cert.key are generated from step 3.
    
# 6.Visualize output
    - In client side, open web browser and enter xavier output log in xavier.
    Example:
        https://192.168.10.62:8080
        
    


