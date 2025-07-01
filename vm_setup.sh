# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# for vast.ai
HOME_DIR=/workspace

cd $HOME_DIR
git clone https://github.com/LongDangHoang/DiffusionPokemon DiffusionPokemonProject
cd DiffusionPokemonProject

# request secrets, can be skipped if you already have them in .env
# read -p "Enter AWS_ACCESS_KEY key: " AWS_ACCESS_KEY
# read -p "Enter AWS_SECRET_ACCESS_KEY key: " AWS_SECRET_ACCESS_KEY
# read -p "Enter WANDB_API_KEY: " WANDB_API_KEY
# read -p "Enter kaggle.json contents: " KAGGLE_JSON_CONTENT

# cat > .env <<EOF
# AWS_ACCESS_KEY="$AWS_ACCESS_KEY"
# AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
# WANDB_API_KEY="$WANDB_API_KEY"
# EOF
# mkdir $HOME/.kaggle
# cat > $HOME/.kaggle/kaggle.json <<EOF
# $KAGGLE_JSON_CONTENT
# EOF

# populate dataset
cd $HOME_DIR/DiffusionPokemonProject
mkdir $HOME_DIR/data/kaggle -p
uv run kaggle datasets download -d danghoanglong/pokemon-images-and-sprites --path $HOME_DIR/data/kaggle
cd $HOME_DIR/data/kaggle
unzip pokemon-images-and-sprites.zip
rm -f pokemon-images-and-sprites.zip
cd $HOME_DIR

# check if uv is using the right thing
uv run python -c "import sys; print(sys.executable)"

# set a new virtual env to local
export VIRTUAL_ENV=.venv

# install jupyter kernel
uv run python -m ipykernel install --user --name=uv-env --display-name "Python (uv-env)"

# run jupyter
uv run jupyter notebook password
uv run jupyter notebook

# git stuff
# git config --global user.email "hoanglongdang2001@gmail.com"
# git config --global user.name "Long Dang"