FROM python:3.10-slim-bullseye

WORKDIR /app

RUN rm -f /etc/apt/sources.list.d/*.sources \
    && echo "deb http://archive.debian.org/debian bullseye main" > /etc/apt/sources.list \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get -o Acquire::Check-Valid-Until=false install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install \
    tensorflow==2.13.0 \
    opencv-python \
    numpy \
    psutil \
    requests \
    pillow

COPY . /app

CMD ["bash"]