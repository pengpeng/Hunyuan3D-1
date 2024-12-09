################################################################################
# 用于构建镜像 'yanwk/comfyui-boot:cu124-cn' 的 Dockerfile
# 为适应国内网络环境，运行时的下载地址均改用国内源。
# 但需注意，构建本镜像时仍需要流畅的网络环境（完全走国内源请参考另一 Dockerfile）。
#
# 使用 CUDA 12.4, Python 3.12, GCC 13
# 容器内将以 root 用户运行，以便于 rootless 部署
################################################################################

FROM docker.io/opensuse/tumbleweed:latest

LABEL maintainer="YAN Wenkun <code@yanwk.fun>"

RUN set -eu

COPY . .

################################################################################
# Python 及工具
# 利用 openSUSE 软件仓库的 PIP 包，以确保兼容性以及更多的系统级支持。后续仍可使用 PIP 更新。

RUN --mount=type=cache,target=/var/cache/zypp \
    zypper addrepo --check --refresh --priority 90 \
        'https://ftp.gwdg.de/pub/linux/misc/packman/suse/openSUSE_Tumbleweed/Essentials/' packman-essentials \
    && zypper --gpg-auto-import-keys \
        install --no-confirm --auto-agree-with-licenses \
python312-devel \
python312-pip \
python312-wheel \
python312-setuptools \
    && rm /usr/lib64/python3.12/EXTERNALLY-MANAGED \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100

################################################################################
# GCC 13 
# 与 CUDA 12.4 兼容

# RUN --mount=type=cache,target=/var/cache/zypp \
#     zypper --gpg-auto-import-keys \
#         install --no-confirm --auto-agree-with-licenses \
# gcc13 \
# gcc13-c++ \
# cpp13 \
#     && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-13 90 \
#     && update-alternatives --install /usr/bin/cc  cc  /usr/bin/gcc-13 90 \
#     && update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-13 90 \
#     && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 90 \
#     && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 90 \
#     && update-alternatives --install /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-13 90 \
#     && update-alternatives --install /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-13 90 \
#     && update-alternatives --install /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-13 90 \
#     && update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-13 90 \
#     && update-alternatives --install /usr/bin/gcov-dump gcov-dump /usr/bin/gcov-dump-13 90 \
#     && update-alternatives --install /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-13 90 

################################################################################
# Python 包

# PyTorch, xFormers
RUN --mount=type=cache,target=/root/.cache/pip \
    pip list \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN bash env_install.sh

RUN cd third_party && git clone --recursive https://github.com/naver/dust3r.git && cd ../third_party/weights && wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

         

# 绑定环境变量 (依赖库 .so 文件)
#ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}\
#:/usr/local/lib64/python3.12/site-packages/torch/lib\
# :/usr/local/lib/python3.12/site-packages/nvidia/cuda_cupti/lib\
# :/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib\
# :/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib\
# :/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib\
# :/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib\
# :/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib\
# :/usr/local/lib/python3.12/site-packages/nvidia/curand/lib\
# :/usr/local/lib/python3.12/site-packages/nvidia/cusolver/lib\
# :/usr/local/lib/python3.12/site-packages/nvidia/cusparse/lib\
# :/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib\
# :/usr/local/lib/python3.12/site-packages/nvidia/nvjitlink/lib\
# :/usr/local/lib/python3.12/site-packages/nvidia/nvtx/lib"


# 1. 安装 ComfyUI 及扩展的依赖项
# 2. 处理 ONNX Runtime 报错 "missing CUDA provider"，并添加 CUDA 12 支持，参考： https://onnxruntime.ai/docs/install/
# 3. 接上，处理 MediaPipe's 的依赖项错误（需要 protobuf<4）


################################################################################

# 使用国内下载源

# RUN --mount=type=cache,target=/var/cache/zypp \
#     zypper modifyrepo --disable --all \
#     && zypper addrepo --check --refresh --gpgcheck \
#         'https://mirrors.tuna.tsinghua.edu.cn/opensuse/tumbleweed/repo/oss/' mirror-oss \
#     && zypper addrepo --check --refresh --gpgcheck \
#         'https://mirrors.tuna.tsinghua.edu.cn/opensuse/tumbleweed/repo/non-oss/' mirror-non-oss \
#     && zypper addrepo --check --refresh --priority 90 \
#         'https://mirrors.tuna.tsinghua.edu.cn/packman/suse/openSUSE_Tumbleweed/Essentials/' mirror-packman-essentials

# ENV PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
        
# ENV HF_ENDPOINT="https://hf-mirror.com"

################################################################################
 

USER root
VOLUME /root
WORKDIR /root
EXPOSE 8080
ENV CLI_ARGS=""
CMD ["python3"," app.py.py --save_memory"]
