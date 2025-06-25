# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

cd $HOME
git clone https://github.com/LongDangHoang/DiffusionPokemon DiffusionPokemonProject
cd DiffusionPokemonProject

# request secrets
read -p "Enter AWS_ACCESS_KEY key: " AWS_ACCESS_KEY
read -p "Enter AWS_SECRET_ACCESS_KEY key: " AWS_SECRET_ACCESS_KEY
read -p "Enter WANDB_API_KEY: " WANDB_API_KEY
read -p "Enter kaggle.json contents: " KAGGLE_JSON_CONTENT

cat > .env <<EOF
AWS_ACCESS_KEY="$AWS_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
WANDB_API_KEY="$WANDB_API_KEY"
EOF
mkdir $HOME/.kaggle
cat > $HOME/.kaggle/kaggle.json <<EOF
$KAGGLE_JSON_CONTENT
EOF

# populate dataset
cd $HOME/DiffiusionPokemonProject
mkdir $HOME/data/kaggle/pokemon-image-dataset -p
uv run kaggle datasets download -d danghoanglong/pokemon-images-and-sprites --path $HOME/data/kaggle/pokemon-image-dataset
cd $HOME/data/kaggle /pokemon-image-dataset
unzip pokemon-image-dataset.zip
rm -f pokemon-image-dataset.zip 
cd $HOME

# set a new virtual env to local
# uv venv
# export VIRTUAL_ENV=./.venv

# check if uv is using the right thing
# uv run python -c "import sys; print(sys.executable)"

# install everything
# uv pip install --requirement pyproject.toml

# install jupyter kernel
uv run python -m ipykernel install --user --name=uv-env --display-name "Python (uv-env)"

# run jupyter
# uv run jupyter notebook password
# uv run jupyter lab

# git stuff
# git config --global user.email "hoanglongdang2001@gmail.com"
# git config --global user.name "Long Dang"