---
layout: post
title: Docker - Basic steps
subtitle: 'A software platform that simplifies the process of building, running, managing and distributing applications.'
description: >- 
    'A container management software'

image: >-
  https://blog.adacore.com/uploads/_2400x1200_crop_center-center_none/iStock-1144628524.jpg
optimized_image: >-
  https://blog.adacore.com/uploads/_2400x1200_crop_center-center_none/iStock-1144628524.jpg
category: blog
tags:
  - docker
  - deployment
author: mowlanica
paginate: False
---

### Docker

![](https://miro.medium.com/max/1356/1*EM4xjGX5K1BLXa7919pcEA.png)

It's a container management software.

Think of Docker simply as service running on your machine.Think of Docker containers as processes or threads of this service. Using Docker you run Docker containers. Docker containers run using an image called **Docker Image** (which you can download from Docker's registry or modify those images to create new images).

When you run a Docker container (using Docker run), it runs a container inside which an isolated OS (for example Ubuntu) runs and you can just attach (login to the container) to the container and perform various operations, run commands, applications just like you would do on your local machine thereby providing you an isolated environment.

You can later map the application to the host machine making it accessible outside the container.

You can create an image of this container, push it to Docker Hub account and then Pull it on a different machine where Docker service is present, and just run a container from this image and you will have the same container as before running on this host machine.

**Images in Docker** : Blue-print for a container/ Instructions for building containers/ snapshot of containers made up of layers like below
![](https://cdn.nanalyze.com/uploads/2018/12/Containerized-Applications.jpg)

**Containers** : Ready to roll applications from Docker images/ running instance of a docker image/when the blue-print is run, it's a container.

**Benefits** : 
1. Easy to install
2. Collaboration
3. Flexibility
4. Totality

**Dockers vs VM's**
![VM's vs Containers](https://qph.fs.quoracdn.net/main-qimg-98b2f9f32e08f3748347c3293b425366)

**Docker** is container based technology and containers are just user space of the operating system. At the low level, a container is just a set of processes that are isolated from the rest of the system, running from a distinct image that provides all files necessary to support the processes. It is built for running applications. In Docker, the containers running share the host OS kernel.

A **Virtual Machine**, on the other hand, is not based on container technology. They are made up of user space plus kernel space of an operating system. Under VMs, server hardware is virtualized. Each VM has Operating system (OS) & apps. It shares hardware resource from the host.

**VMs & Docker** – each comes with benefits and demerits. Under a VM environment, each workload needs a complete OS. But with a container environment, multiple workloads can run with 1 OS. The bigger the OS footprint, the more environment benefits from containers. With this, it brings further benefits like Reduced IT management resources, reduced size of snapshots, quicker spinning up apps, reduced & simplified security updates, less code to transfer, migrate and upload workloads.

![](https://eadn-wc03-4064062.nxedge.io/cdn/wp-content/uploads/2020/05/2020_05_13_12_19_07_PowerPoint_Slide_Show_Azure_AZ104_M01_Compute_ed1_.png)



## STEPS :

1. FROM *python:3.8.0* 
(`FROM` initializes a new build stage and sets the *Base Image* for subsequent instructions. As such, a valid Dockerfile must start with a FROM instruction. The image can be any valid image – it is especially easy to start by pulling an image from the Public Repositories.)
2. WORKDIR */path/to/workdir*
(`WORKDIR` set the working directory) 
3. COPY /path/to/workdir
(`COPY` instruction copies new files or directories from *<src>* and adds them to the filesystem of the container at the path *<dest>*.Multiple *<src>* resources may be specified but the paths of files and directories will be interpreted as relative to the source of the context of the build.)
4. RUN *pip install -r requirements.txt*
(`RUN` instruction will execute any commands in a new layer on top of the current image and commit the results. The resulting committed image will be used for the next step in the `Dockerfile`.Layering `RUN` instructions and generating commits conforms to the core concepts of Docker where commits are cheap and containers can be created from any point in an image’s history, much like source control. The `exec` form makes it possible to avoid shell string munging, and to `RUN` commands using a base image that does not contain the specified shell executable.)
5. ENTRYPOINT["python"]
(An `ENTRYPOINT` allows you to configure a container that will run as an executable.)
6. CMD["flask app"]
(The main purpose of a `CMD` is to provide defaults for an executing container. These defaults can include an executable, or they can omit the executable, in which case you must specify an `ENTRYPOINT` instruction as well.)

### Basic commands:
`docker -version` : shows the docker version
`docker images`   : shows a list of docker images
`docker ps -a`    : checks all the docker mages running currently/active
`docker build -t tagname:1.0 .` : builds a docker image in the current folder where -t is the docker image tag (Example: Iris species classification - I'll name this image as iris_dataset:1.0 , 1.0 is the version number and ` .` checks for all the images with that tag and version in the folders)
`docker run -p 5000:3000 tagname:1.0` : 5000 is the port where we want docker to run and 3000 is the port mentioned in the flask app along with tagname+verison
`docker run -d - p 5000:3000 tagname:1.0` : `-d` runs the container in the background and prints the container ID(detached the container)
