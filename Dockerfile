# Use pytorch image as base
FROM pytorch/pytorch:latest

RUN apt update
RUN apt install vim git -y

# Keep container running for bash shell
ENTRYPOINT ["/bin/bash", "-c", "echo Welcome"]
