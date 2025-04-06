.PHONY: deploy upload run clean prompt_vars

# Target file to upload and execute
LOCAL_SCRIPT := vm_setup.sh
REMOTE_SCRIPT := /tmp/setup_script.sh

# Ask for variables if not already set
define prompt_vars
	@read -p "Enter SSH username: " ssh_user; \
	read -p "Enter SSH host (e.g. IP or domain): " ssh_host; \
	read -p "Enter path to SSH private key (e.g. ~/.ssh/id_rsa): " ssh_key; \
	echo "SSH_USER=$$ssh_user" > .ssh_env; \
	echo "SSH_HOST=$$ssh_host" >> .ssh_env; \
	echo "SSH_KEY=$$ssh_key" >> .ssh_env;
endef

# Load saved SSH environment
-include .ssh_env

# Full deploy process
prompt_vars:
	@if [ ! -f .ssh_env ]; then $(prompt_vars); fi

upload:
	scp -i $(SSH_KEY) $(LOCAL_SCRIPT) $(SSH_USER)@$(SSH_HOST):$(REMOTE_SCRIPT)

run:
	ssh -i $(SSH_KEY) -t $(SSH_USER)@$(SSH_HOST) "bash $(REMOTE_SCRIPT)"

clean:
	rm -f .ssh_env

deploy: prompt_vars upload run clean
