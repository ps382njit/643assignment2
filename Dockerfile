FROM centos:7

RUN sudo yum -y update 
RUN sudo yum -y install python3 python3-dev python3-pip python3-virtualenv
RUN sudo yum -y install java-1.8.0-openjdk wget

ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

RUN pip3 install --upgrade pip
RUN pip3 install numpy panda

RUN cd /home/ec2-user && sudo wget https://apache.osuosl.org/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz
RUN tar -xzf spark-2.4.5-bin-hadoop2.7.tgz
RUN rm spark-2.4.5-bin-hadoop2.7.tgz


RUN sudo ln -s /home/ec2-user/spark-2.4.5-bin-hadoop2.7 /opt/spark
RUN (echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc && echo 'export PATH=$SPARK_HOME/bin:$PATH' >> ~/.bashrc && echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc)

RUN mkdir /643assignment2
COPY predict.py /643assignment2/ 

RUN sudo rm /bin/sh && sudo ln -s /bin/bash /bin/sh

RUN sudo /bin/bash -c "source ~/.bashrc"
RUN sudo /bin/sh -c "source ~/.bashrc"

WORKDIR /643assignment2

ENTRYPOINT ["/opt/spark/bin/spark-submit", "--packages", "org.apache.hadoop:hadoop-aws:2.7.7", "predict.py"]


