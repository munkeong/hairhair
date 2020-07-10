FROM nvidia/cuda:10.0-base-ubuntu16.04
#MAINTAINER  jeon

RUN sed -i 's/archive.ubuntu.com/mirror.us.leaseweb.net/' /etc/apt/sources.list \
    && sed -i 's/deb-src/#deb-src/' /etc/apt/sources.list \
    #&& apt-get update \
    #&& apt-get upgrade -y \
    && apt-get install -y \
    build-essential \
    ca-certificates \
    gcc \
    git \
    libpq-dev \
    make \
    pkg-config \
    python3.5 \
    python3.5-dev \
    python3.5-pip \
    aria2 \
    && apt-get autoremove -y \
    && apt-get clean

RUN pip3 install tensorflow-1.15.3-cp35-cp35m-linux_x86_64.whl

RUN mkdir -p /root/detection
WORKDIR /root/detection
ADD . /root/detection/

RUN mkdir -p /root/cert.bak
RUN rpm -Vv ca-certificates | awk '$1!="........."&& $2!="d" {system("mv -v " $NF " /root/cert.bak")}'

RUN apt-get update
RUN apt-get install wget

#RUN pip3 install -U virtualenv
#RUN pip3 install zipp==1.0.0

#RUN pip3 install --upgrade pip
#RUN pip3 install -r requirements.txt
#RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y libxrender-dev
#RUN pip3 install opencv-python
RUN apt-get install -y python3.5-tk
RUN apt-get install -y libxrender1


#RUN pip3 install pillow
#RUN pip3 install flask


#RUN pip3 install tensorflow-1.15.3-cp35-cp35m-linux_x86_64.whl

EXPOSE 5000

#CMD []
#ENTRYPOINT ["/bin/bash"]
CMD ["python3.5", "app.py"]
