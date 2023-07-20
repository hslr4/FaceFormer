FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends ffmpeg curl rustc cargo && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

CMD python3 server.py