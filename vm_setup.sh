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
uv run kaggle datasets download -d hlrhegemony/pokemon-image-dataset --path $HOME/data/kaggle/pokemon-image-dataset
cd $HOME/data/kaggle /pokemon-image-dataset
unzip pokemon-image-dataset.zip
rm -f pokemon-image-dataset.zip 
cd $HOME
