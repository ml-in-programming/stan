FROM ubuntu:latest

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y  software-properties-common && \
    add-apt-repository ppa:webupd8team/java -y && \
    apt-get update && \
    echo oracle-java7-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections && \
    apt-get install -y oracle-java8-installer && \
    apt-get clean

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN apt-get --assume-yes install git

RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    apt-get --assume-yes install curl && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get --assume-yes install google-cloud-sdk

WORKDIR /app
RUN mkdir /app/backend && \
    mkdir /app/data && \
    mkdir /app/interface && \
    mkdir /app/workflow

COPY backend/project_predictor.py /app/backend/
RUN gsutil cp gs://codestyle-similarity/models/model_actual.dat /app/backend/model_actual.dat
COPY backend/run_coan /app/backend/
COPY backend/project_predictor.py /app/backend/
COPY data/encoder_actual.txt /app/data
COPY data/encoder_links.txt /app/data
COPY data/data_headers.csv /app/data
COPY workflow/data_preparation.py /app/workflow
COPY interface /app/
COPY requirements.txt /app/

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8000

RUN ls -la

CMD python3 -u /app/manage.py runserver 0.0.0.0:8000