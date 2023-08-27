pip install kaggle pytorch-lightning python-dotenv wandb==0.15.0 protobuf==3.20.3 boto3

cd ..

mkdir ./kaggle/input/pokemon-image-dataset
cd ./kaggle/input/
kaggle datasets download -d hlrhegemony/pokemon-image-dataset
unzip pokemon-image-dataset.zip
rm -f pokemon-image-dataset.zip 
mv images ./pokemon-image-dataset

cd ../..
