# FROM nvcr.io/nvidia/pytorch:21.09-py3
# FROM nvcr.io/nvidia/pytorch:23.09-py3 too latest
FROM nvcr.io/nvidia/pytorch:23.01-py3
ENV DEBIAN_FRONTEND=noninteractive 

# dependencies for gym
#
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 gcc-8 \
 g++-8 \
 vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 git-lfs 

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# WORKDIR /opt/isaacgym
WORKDIR /isaacgym/isaacgym

COPY . .

# install gym modules
ENV PATH="/home/gymuser/.local/bin:$PATH"
RUN cd python && pip install -e .
RUN ls && cd IsaacGymEnvs && pip install -e .

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# install ros packages
# RUN apt-get update
# RUN apt-get install -y lsb-release
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# RUN apt-get update 
# RUN apt-get install -y ros-noetic-desktop-full
# RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
# RUN source ~/.bashrc
 
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" --\
    -p git -p 'history-substring-search' \
    -a 'bindkey "\$terminfo[kcuu1]" history-substring-search-up' \
    -a 'bindkey "\$terminfo[kcud1]" history-substring-search-down' \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting

RUN echo "source /opt/ros/noetic/setup.zsh" >> ~/.zshrc