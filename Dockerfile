# Use pytorch image as base
FROM pytorch/pytorch:latest

RUN apt update
RUN apt install git -y

# Keep container running for bash shell
ENTRYPOINT ["/bin/bash", "-c", "echo Welcome"]
