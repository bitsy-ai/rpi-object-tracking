FROM debian

WORKDIR /workspace

RUN apt-get update
RUN apt-get -y install \
  curl \
  gnupg2

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list

RUN apt-get update
run apt-get install edgetpu

RUN apt-get clean

CMD command