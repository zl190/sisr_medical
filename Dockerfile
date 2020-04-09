# Install tensorflow
FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /root

# prepare devs
RUN pip3 install tensorflow_datasets numpy pandas google-cloud-storage scikit-image pytest
RUN apt-get install -y git
RUN git clone https://github.com/jason-zl190/datasets.git

# mkdir project directory and copies the trainer code to the docker image.
RUN mkdir sisr_ct
WORKDIR sisr_ct
COPY trainer ./trainer
RUN mkdir trained_models

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "-m", "trainer.srresnet_task"]