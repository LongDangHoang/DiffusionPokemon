# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# populate dataset
cd ..
mkdir ./kaggle/input/pokemon-image-dataset
cd ./kaggle/input/
kaggle datasets download -d hlrhegemony/pokemon-image-dataset
unzip pokemon-image-dataset.zip
rm -f pokemon-image-dataset.zip 
mv images ./pokemon-image-dataset
cd ../..
