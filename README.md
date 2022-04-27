# 643assignment2
643 Assignment 2

Running training on EMR instances
1. Create a cluster with the following configs:<br />
    `Step type` = `Spark application`, then click `Configure`<br />
    `Deploy mode` = `Client`<br />
    `Spark-submit options`=`--packages org.apache.hadoop:hadoop-aws:2.7.7`<br />
    `Application location` = <your S3 path> (mine was `s3://parth643assignment2/train.py`<br />
    `Instance Type` = `m5.large`<br />
    `Number of instances` = `4`<br />

2. Click `Create Cluster`
3. After 20 minutes or so, the steps will be completed and the models will be uploaded to your s3 folder.


Running prediction on Ec2 instance
1. Create Ec2 instance with `Amazon-Linux` AMI
2. Set-up Java and `JAVA_HOME` variables: <br />
    https://bhargavamin.com/how-to-do/setting-up-java-environment-variable-on-ec2/
3. Set-up Spark, Pyspark, and Python: <br />
  https://computingforgeeks.com/how-to-install-python-on-amazon-linux/<br />
  https://analyticsindiamag.com/beginners-guide-to-pyspark-how-to-set-up-apache-spark-on-aws/<br />
  https://spark.apache.org/docs/1.6.2/ec2-scripts.html
4. Make sure environment variables are correct in `.bashrc`
5. Upload `predict.py` to `/home/ec2-user` as well as `ValidationDataset.csv` or `TestDataset.csv` for local deployment
6. To run locally on ec2 instance, in `/home/ec2-user` where both py and csv file should be- run:<br />
    `spark-submit --packages org.apache.hadoop:hadoop-aws:2.7.7 predict.py --testCsv ValidationDataset.csv --classifier linear`
7. To deploy from S3, change `--testCsv` argument to `s3a://parth643assignment2/ValidationDataset.csv`
8. To run Random Forrest Regression instead of Linear Regression, change `--classifier` argument to `rfc`

Running using Docker
1. Create Ec2 instance with `Amazon-Linux` AMI
2. Upload `643assignment2` with all csv and py files
3. `cd` into `643assignment2` directory
4. Run `docker build -t 643assignment2 .` to build the image
5. Run `docker images --filter reference=643assignment2` to verify image has been created
6. Run `docker run -it 643assignment2:latest --testCsv ValidationDataset.csv --classifier linear` or `docker run -it 643assignment2:latest --testCsv s3a://parth643assignment2/ValidationDataset.csv --classifier linear` for s3



