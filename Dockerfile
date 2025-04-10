FROM gcr.io/kaggle-gpu-images/python
WORKDIR /workspace
COPY vm_setup.sh .
ENV HOME=/workspace

# Define default command
CMD ["bash"]
