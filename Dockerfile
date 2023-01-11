FROM tensorflow/tensorflow:2.6.0-gpu

RUN apt-get update --fix-missing
RUN apt-get install -y --no-install-recommends libtcmalloc-minimal4
RUN apt-get update && apt-get install -y libopenexr-dev libgl1-mesa-glx

# Install python packages
RUN pip install OpenEXR
RUN pip install opencv-python opencv-contrib-python
RUN pip install psutil
RUN pip install pytz
RUN pip install parmap
RUN pip install scipy

# Run codes
VOLUME /data
VOLUME /codes
WORKDIR /codes
