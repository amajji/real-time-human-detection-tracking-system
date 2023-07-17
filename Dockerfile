FROM ubuntu:18.04

 
RUN mkdir /code

WORKDIR /code
 

RUN apt-get update \

    && apt-get install -y python3.10 \

    python3-pip

 

RUN pip3 install --upgrade pip

 

ENV LANG C.UTF-8

ENV LC_ALL C.UTF-8

 

COPY ./requirements.txt /code/requirements.txt
COPY ./src  /code/src


 

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

 
 

CMD uvicorn src.app.app:app --host 0.0.0.0 --port 80  --forwarded-allow-ips '*' --reload --log-config ./src/app/log.ini
