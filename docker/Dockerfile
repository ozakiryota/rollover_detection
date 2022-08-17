########## Pull ##########
FROM nvidia/cuda:11.1.1-base-ubuntu18.04
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
		torch==1.10.1+cu111 \
		torchvision==0.11.2+cu111 \
		torchaudio==0.10.1 \
		-f https://download.pytorch.org/whl/torch_stable.html \
		tensorboardX \
		tensorboard
########## Cache busting ##########
ARG CACHEBUST=1
########## rollover_detection ##########
RUN pip3 install \
		matplotlib \
		tqdm && \
	cd ~ && \
	git clone https://github.com/ozakiryota/rollover_detection.git
########## Initial position ##########
WORKDIR /root/rollover_detection