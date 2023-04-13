# Use pytorch image as base
FROM pytorch/pytorch:latest

# Keep container running for bash shell
ENTRYPOINT ["tail", "-f", "/dev/null"]
