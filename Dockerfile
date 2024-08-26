# 使用phusion/baseimage作为基础镜像,去构建你自己的镜像,需要下载一个明确的版本,千万不要使用`latest`.
# 查看https://github.com/phusion/baseimage-docker/blob/master/Changelog.md,可用看到版本的列表.
FROM phusion/baseimage:jammy-1.0.4

# 设置正确的环境变量.
ENV HOME /root

# 生成SSH keys,baseimage-docker不包含任何的key,所以需要你自己生成.你也可以注释掉这句命令,系统在启动过程中,会生成一个.
RUN /etc/my_init.d/00_regen_ssh_host_keys.sh

# 初始化baseimage-docker系统
CMD ["/sbin/my_init"]

# 这里可以放置你自己需要构建的命令

# 换源
RUN seq 1 8 | xargs -I{} sed -i 's/archive.ubuntu.com/mirrors.nju.edu.cn/g' /etc/apt/sources.list 
RUN seq 1 8 | xargs -I{} sed -i 's/security.ubuntu.com/mirrors.nju.edu.cn/g' /etc/apt/sources.list

RUN add-apt-repository ppa:xmake-io/xmake && apt update && apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 python3-pip git xmake



RUN pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions -i https://mirrors.nju.edu.cn/pypi/web/simple/


# 将当前文件夹内数据拷贝到镜像
ADD ./env /root/workspace/

# 进入到/root/目录
WORKDIR /root/workspace/

# 安装Env下的run.sh

RUN chmod +x ./*.run && \
    ./*.run --quiet --install 
# 更新环境变量 
RUN echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /etc/profile

RUN echo "export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest" >> /etc/profile

RUN git clone https://github.com/KVM-Explorer/AscendPathTracing.git
# 当完成后,清除APT.
# RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*