#! /bin/bash


BASE_URL="https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"



Train_path="GTSRB-Training_fixed.zip"
Train_path_md5="513f3c79a4c5141765e10e952eaa2478"

Test_images="GTSRB_Final_Test_Images.zip"
Test_images_md5="c7e4e6327067d32654124b0fe9e82185"

Test_GT="GTSRB_Final_Test_GT.zip"
Test_GT_md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5"

ROOT_PATH="./GTSRB"

mkdir -p $ROOT_PATH


function download_file() {
    local url=$1
    local filename=$2
    local md5=$3

    if [ ! -f "$filename" ]; then
        echo "Downloading $filename..."
        wget "$url" -O "$filename"
        echo "Download complete."
    else
        echo "File $filename already exists, skipping download."
    fi

    if [[ $(md5sum "$filename" | awk '{print $1}') != "$md5" ]]; then
        echo "MD5 mismatch for $filename. Try to redownload..."
        rm "$filename"
        download_file "$url" "$filename" "$md5"
    else
        echo "MD5 checksum matches for $filename."
    fi
}


cd $ROOT_PATH

download_file "$BASE_URL/$Train_path" "$Train_path" "$Train_path_md5"
download_file "$BASE_URL/$Test_images" "$Test_images" "$Test_images_md5"
download_file "$BASE_URL/$Test_GT" "$Test_GT" "$Test_GT_md5"


unzip -q "$Train_path"
unzip -q "$Test_images"
unzip -q "$Test_GT"


echo "Download complete."

