########## Pull ##########
# FROM ubuntu:20.04
FROM nvidia/cuda:10.2-base-ubuntu18.04
########## Non-interactive ##########
ENV DEBIAN_FRONTEND=noninteractive
########## Common tools ##########
RUN apt-get update && \
    apt-get install -y \
	    vim \
    	wget \
    	unzip \
    	git \
		python3-tk
########## PyTorch ##########
RUN apt-get update && \
    apt-get install -y \
	    python3-pip && \
	pip3 install --upgrade pip && \
	pip3 install \
		torch==1.10.1+cu102 \
		torchvision==0.11.2+cu102 \
		torchaudio==0.10.1 \
		-f https://download.pytorch.org/whl/torch_stable.html
########## Cache busting ##########
# ARG CACHEBUST=1
########## rollover_detection ##########
RUN pip3 install \
		matplotlib && \
	cd ~
	# git clone https://github.com/ozakiryota/rollover_detection.git
########## Initial position ##########
WORKDIR /root/rollover_detection