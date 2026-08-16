# Reflection

When I started this assignment I honestly did not know much about MLOps. I had
some idea about training models in PyTorch, but Docker and Kubernetes were almost
completely new to me. So most of my learning in this assignment did not come from
the machine learning part, but from all the infrastructure around it. The actual
model (a ResNet-18 on CIFAR-10) was the easy part. Getting it to run inside Docker,
and then inside Kubernetes, was where I struggled and learned the most.

The most challenging part for me was getting the CIFAR-10 data into Kubernetes.
When I ran training on my own laptop, the dataset just downloaded automatically,
although even that failed at first because of an expired SSL certificate error, so
I had to download the dataset manually and place it in the data folder. But the
bigger confusion came with Kubernetes. I assumed the container would be able to see
my Windows folders like it did with Docker volume mounts, but it could not. I learned
that a PersistentVolumeClaim (PVC) starts empty, and you cannot copy files directly
into a PVC. I had to create a small helper pod that mounts the PVC, use "kubectl cp"
to copy the CIFAR data into that pod, and then delete the helper pod. Understanding
why this was needed (that only a pod can mount a volume, and a PVC is just storage)
was a real "aha" moment for me.

I also faced a lot of small environment problems because I am on Windows with only
8 GB of RAM. Installing "kind" was tricky because it got installed but was not added
to the PATH properly, so the command was not recognised until I fixed the PATH and
restarted the terminal. During one Docker build, my internet connection dropped in
the middle of downloading PyTorch and the file got corrupted, which I fixed by adding
retry and timeout options to pip. Running Docker, a kind cluster, and two serving
replicas together was very heavy for my laptop, so I had to stop my other project
containers and use smaller resource requests while still keeping the resource limits
that the assignment asked for.

Along the way, some concepts finally became clear to me. I understood the difference
between a Job, which runs once and finishes (used for training), and a Deployment,
which keeps pods running all the time (used for serving). I also learned what liveness
and readiness probes do, and why images have to be loaded into the kind cluster
separately using "kind load", because the cluster has its own image store.

Overall, the biggest lesson I took from this assignment is that MLOps is mostly about
making things work the same way in every environment. The model itself was simple,
but the real effort was in the "glue" around it, like networking, storage, and
configuration. It was frustrating at times, but I feel a lot more confident with
Docker and Kubernetes now than when I started.
