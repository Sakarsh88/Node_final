FROM daisukekobayashi/darknet:darknet_yolo_v4_pre-gpu-cv-cc53
# Edit the base image here, e.g., to use 
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/) 
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
#RUN apt-get update && apt-get install -y python3.7 python3-distutils python3-pip python3-apt wget git
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output




ENV PATH="/home/algorithm/.local/bin:${PATH}"
RUN apt-get update && apt-get install -y python3.7 python3-distutils python3-pip python3-apt wget git python-opencv
USER algorithm
WORKDIR /opt/algorithm
RUN git clone https://github.com/AlexeyAB/darknet.git
RUN python3 -m pip install --user -U pip
RUN  pip install --upgrade pip

# Copy all required files so that they are available within the docker image 
# All the codes, weights, anything you need to run the algorithm!
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm entrypoint.sh /opt/algorithm/
COPY --chown=algorithm:algorithm getting_results.py /opt/algorithm/
COPY --chown=algorithm:algorithm preprocessing.py /opt/algorithm/
COPY --chown=algorithm:algorithm create_dirs.py /opt/algorithm/
COPY --chown=algorithm:algorithm create_files.py /opt/algorithm/
#COPY --chown=algorithm:algorithm input /input
COPY --chown=algorithm:algorithm utils.py /opt/algorithm/utils.py
#ADD darknet .
# Install required python packages via pip - please see the requirements.txt and adapt it to your needs
#RUN python3 -m pip install --user cmake
#RUN python3 -m pip install --user dlib
RUN pip3 install --user -r requirements.txt
RUN pip3 install --user opencv-python
#RUN python3 -m pip install --user SimpleITK
COPY --chown=algorithm:algorithm process.py postprocessing.py /opt/algorithm/

# Entrypoint to run, entypoint.sh files executes process.py as a script
ENV PYTHONPATH = "$PYTHONPATH:/home/algorithm/.local/lib/python3.6/site-packages"
ENTRYPOINT ["bash", "entrypoint.sh"]

## ALGORITHM LABELS: these labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=noduledetection
# These labels are required and describe what kind of hardware your algorithm requires to run for grand-challenge.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=2
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=12G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=60G


