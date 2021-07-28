from nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ENV TZ=Europe/Stockholm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update && apt-get install -y cython python3-tk python3-pip build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget

RUN python3 -m pip install --upgrade pip

RUN pip3 install pymongo torch torchvision tk

RUN pip3 install opencv-python-headless



COPY ./ /app

RUN pip3 install -r /app/sort/requirements.txt


WORKDIR /app


CMD [ "python3", "-u", "main.py" ]

