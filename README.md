# 643assignment2
643 Assignment 2

Running training on EMR instances
1. Create a cluster with the following configs:
    `Step type` = `Spark application`, then click `Configure`
    `Deploy mode` = `Client`
    `Spark-submit options`=`--packages org.apache.hadoop:hadoop-aws:2.7.7`
    `Application location` = <your S3 path> (mine was `s3://parth643assignment2/train.py`
    `Instance Type` = `m5.large`
    `Number of instances` = `4`

2. Click `Create Cluster`
3. After 20 minutes or so, the steps will be completed and the models will be uploaded to your s3 folder.


Running prediction on Ec2 instance
1. Create Ec2 instance with `Amazon-Linux` AMI
2. Set-up Java and `JAVA_HOME` variables: https://bhargavamin.com/how-to-do/setting-up-java-environment-variable-on-ec2/
3. Set-up Spark, Pyspark, and Python: 
  https://computingforgeeks.com/how-to-install-python-on-amazon-linux/
  https://analyticsindiamag.com/beginners-guide-to-pyspark-how-to-set-up-apache-spark-on-aws/
  https://spark.apache.org/docs/1.6.2/ec2-scripts.html
4. Make sure environment variables are correct in `.bashrc`
5. Upload `predict.py` to `/home/ec2-user` as well as `ValidationDataset.csv` or `TestDataset.csv` for local deployment
6. To run locally on ec2 instance, in `/home/ec2-user` where both py and csv file should be- run:
    `spark-submit --packages org.apache.hadoop:hadoop-aws:2.7.7 predict.py --testCsv ValidationDataset.csv --classifier linear`
7. To deploy from S3, change `--testCsv` argument to `s3a://parth643assignment2/ValidationDataset.csv`
8. To run Random Forrest Regression instead of Linear Regression, change `--classfier` argument to `rfc`



