<h3 align="center"><b>FAULTY WAFER COMPONENT DETECTION</b></h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> The component in focus is <b>Wafer</b> which is a thin slice of semiconductor, such as a crystalline silicon (c-Si), used for the fabrication of integrated circuits and, in photovoltaics, to manufacture solar cells. It serves as the substrate for microelectronic devices built in and upon the wafer.  The objective here boils down to determining whether a wafer at hand is faulty or not, with the help of <b>Machine Learning Pipeline</b> built and employed in the project, resulting in obliteration of the need and thus the cost of employing manual labor.<br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Built Using](#built_using)
- [Deployment](#deployment)


## 🧐 About <a name = "about"></a>

As introduced, <b>Wafer</b> is a thin slice of semiconductor, such as a crystalline silicon (c-Si), used for the fabrication of integrated circuits and, in photovoltaics, to manufacture solar cells; serving as the substrate (serves as foundation for construction of other components) for microelectronic devices built in and upon the wafer. They undergoes many microfabrication processes, such as doping, ion implantation, etching, thin-film deposition of various materials, and photolithographic patterning. Finally, the individual microcircuits are separated by wafer dicing and packaged as an integrated circuit.

### <b>Problem Statement</b>

It's to be determined that whether a wafer at hand is faulty or not. Up until now, this probem has been dealt with the manual efforts. Even if the employed labor have some sort of hunch about the wafers' status, they have to open them from scratch which ivolves a lot of time and not to mention the period of inactivity for other wafers in the vicinity, cost incurred, manpower and what not coming along as a package with this modus operandi. Apparently, the motivation behind this project becomes to obliterate the need of employing aforementioned manual labor and saving them a lot of trouble.

### <b>Solution Proposed</b>

Data sent on by wafers via MQTT (or any other IoT communication protocol) to messaging servers (such as Apache Kafka), from where the data is fetched and dumped into the desired database (here, MongoDB), and then accessed; is to pass through the machine learning pipeline to conclude the status of the wafers and in doing so apparently, obliterating the need and thus cost of hiring manual labor.

### <b>About Data</b>

The client sends data in multiple sets of files as batches. Data contain wafer names and 590 columns each representing a constituent sensor containing respective readings for its wafer. The last column tells about the status "Good/Bad" value of wafers.

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### <b>Prerequisites</b>

This project requires the [Python](https://www.python.org/downloads/), [Pip-PyPI](https://pip.pypa.io/en/stable/installation/) and [Docker](https://www.docker.com/) installed. Apart from these, you do need to have an [AWS](https://aws.amazon.com/?nc2=h_lg) account to access the services like [Amazon ECR](https://aws.amazon.com/ecr/), [Amazon EC2](https://aws.amazon.com/ec2/?nc2=type_a), and [Amazon S3](https://aws.amazon.com/s3/).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included.

**NOTE:** All the other dependencies will be installed when you'd get your virtual env ready and install the requirements present in the **requirements.txt** file.

### <b>Installing</b>

A step by step series of examples that tell you how to get the development env running.

**i.** First and foremost, create a virtual environment,

```
conda create -n hey_wafer python==3.7
```

..accompanied by the activation of the created environment.

```
conda activate hey_wafer
```

**ii.** Now, install the requirements for this project.

```
pip install -r requirements.txt
```

**iii.** Setup a database, now, in the [MongoDB Atlas](https://www.mongodb.com/atlas/database) and copy the connection string from there, followed by creating a `.env` file in the development env. Then, create an env variable `MONGO_DB_URL` into `.env` file assign the connection string to it.

```
MONGO_DB_URL = <connection_string>
```

**iv.** Now to orchestrate and monitor the **machine learning pipelines/workflows**, run 

```
# The Standalone command will initialise the database, make a user,
# and start all components for you.

airflow standalone
```
..followed by visiting `localhost:8080` in the browser. Can now use the admin account details shown on the terminal to login. 

**v.** At the moment, you're good to run the training and prediction pipelines from the airflow UI as per the requirement.

**NOTE:** If you are running the training pipeline for the very first time or there's some additional new data and you wish to retrain the pipline, go to `wafer/pipelines/training.py` and set the `new_data`=True in the **TrainingPipeline** class.


## ⛏️ Built Using <a name = "built_using"></a>

- [MongoDB](https://www.mongodb.com/) - Database
- [Airflow](https://airflow.apache.org/) - Scheduler and Orchestrator of Pipelines and Workflows
- [Docker](https://www.docker.com/) - To Containerize the Application
- [Github Actions](https://github.com/features/actions) - For Continous Integration and Continous Delivery


## 🚀 Deployment <a name = "deployment"></a>

To deploy this application on a live system, we are gonna use the **AWS** cloud platform. The flow would go like this -- A docker image in regard to instructions specified in `Dockerfile` from GitHub is going to be pushed to [Amazon ECR (Elastic Container Registry)](https://aws.amazon.com/ecr/), from where it's gonna be pulled to [Amazon EC2 (Elastic Compute Cloud)](https://aws.amazon.com/ec2/?nc2=type_a) wherein it'll run and build the apparent docker container. **All these steps are automated via [GitHub Actions](https://github.com/features/actions).**

Whence the training pipeline concludes in the airflow UI, all the artifacts and models built in the process are gonna be synced to [Amazon S3](https://aws.amazon.com/s3/) bucket as instructed in the **training airflow dag**.

```
# training airflow dag

training_pipeline >> sync_data_to_s3
```

The batches on which predictions are to be made can be uploaded directly to the [Amazon S3](https://aws.amazon.com/s3/) bucket from where they'll be downloaded to the airflow and the prediction pipeline will run atop of them returning a single output file containing the status of all the wafers, which will be synced to the prediction dir in **Amazon S3**, in regard to the flow defined in the **prediction airflow dag**.

```
# prediction airflow dag

download_input_files >> generate_prediction_files >> upload_prediction_files
```


Now, follow the following series of steps to deploy this application on a live system:

**NOTE:** Following steps are to be followed only after you commit and push all the code into a maintainable GitHub repository.

<br>

**i.** First off all, login in to the AWS using your AWS account credentials. Then, go to the **IAM** section and create a user `<username>` there, followed by selection of `Access key - Programmactic Access` for the **AWS access type**. Going next you'll find the **Attach existing policies directly** tab, where you have to check `Admininstrator Access` option. 

Now, upon creation of the user, you'll see an option to download csv file, from which the three necessary credentials oughta be gathered -- **AWS_ACCESS_KEY_ID**, **AWS_SECRET_ACCESS_KEY** and **AWS_REGION**. As such, download the csv file and store these credentials somewhere.

<br>

**ii.** Now, create a repository in [Amazon ECR](https://aws.amazon.com/ecr/) where the docker image is to be stored and can be pulled into [Amazon EC2](https://aws.amazon.com/ec2/?nc2=type_a) and be put to use, as per the need/desire.

After this, copy the URI of the created repository from the ECR repositories section and assign it to the **AWS_ECR_LOGIN_URI** and **ECR_REPOSITORY_NAME** accordingly and store these variables somewhere secure.

```
# Say, for example
URI = 752513066493.dkr.ecr.ap-south-1.amazonaws.com/wafer-fault-detection

# Assign accordingly
AWS_ECR_LOGIN_URI = 752513066493.dkr.ecr.ap-south-1.amazonaws.com
ECR_REPOSITORY_NAME = wafer-fault-detection
```
<br>

**iii.** As of now, [Amazon S3](https://aws.amazon.com/s3/) bucket and [Amazon EC2](https://aws.amazon.com/ec2/?nc2=type_a) instance are yet to be created. Let's get on with the S3 bucket!

Go to the **Amazon S3** section, get started with creating a S3 bucket by choosing a name that is universally unique, followed by choosing the correct `AWS Region`. Upon creation of the said bucket, assign the bucket name to the variable **BUCKET_NAME** and store it somewhere secure.

<br>

**iv.** Similarly, let's get going with the EC2 instance. Go to the instances section of the EC2, and launch a new  EC2 instance escorted by choosing a name and a relevant `Application and OS Image`(say, **Ubuntu** for this project) for it. After that in the `Key pair (login)`, create a new key pair upon which a `.pem` file will be downloaded. Now, you can configure storage as per the requirement for this instance and finally get done with its launch.

At the moment, you have to wait for the `Status check` of launch of this instance to be passed.

<br>

**v.** Now, to access this instance with not only the SSH request but with the HTTP request too, we've gotta configure this setting. For this matter, go to the `Edit inbound rules` of this instance and add a new rule -- Make the **Type** as `All traffic` and the **Source** as `Anywhere-IPv4` -- and save this rule.

<br>

**vi.** Click on the `connect` button of the launched instance and the terminal will get open.

Now, in the terminal, execute the following commands, one at a time, to get the **Docker** up and running.

```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

<br>

**vii.** From the **runners** section in the **Actions settings** of your GitHub repo, create a new **linux** `self-hoster runner` and copy the commands from therein and execute them one by one in the EC2 instance terminal.

**Download**

```
# Create a folder
$ mkdir actions-runner && cd actions-runner# Download the latest runner package
$ curl -o actions-runner-linux-x64-2.301.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.301.1/actions-runner-linux-x64-2.301.1.tar.gz# Optional: Validate the hash
$ echo "3ee9c3b83de642f919912e0594ee2601835518827da785d034c1163f8efdf907  actions-runner-linux-x64-2.301.1.tar.gz" | shasum -a 256 -c# Extract the installer
$ tar xzf ./actions-runner-linux-x64-2.301.1.tar.gz
```

**Configure**

```
# Create the runner and start the configuration experience
$ ./config.sh --url https://github.com/suryanshyaknow/faulty-wafer-component-detection --token APNAENNPAOWALIWWN55VAELD2BO4A
# Last step, run it!
$ ./run.sh
```
**NOTE:** Upon the successful addition of the runner, when it's being asked on the terminal the name of the runner, enter `self-hosted`.

**Using your self-hosted runner**

```
# Use this YAML in your workflow file for each job
runs-on: self-hosted
```

<br>

**viii.** Now, at last, gather all those variabes that you were earlier asked to store securely -- **AWS_ACCESS_KEY_ID**, **AWS_SECRET_ACCESS_KEY**, **AWS_REGION**, **AWS_ECR_LOGIN_URI**, **ECR_REPOSITORY_NAME**, **BUCKET_NAME**, and at last but not the least, **MONGO_DB_URL** -- and add them as the **Secrets** in the `Actions Secrets and Variables` section of your GitHub repo. 

With this, the deployment setup has ultimately been configured. You can now access the deployment link of this application which is the `Public IPv4 DNS` address of the said EC2 instance.

---