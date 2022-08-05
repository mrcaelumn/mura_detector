#! /bin/bash



### create folders
listFolder=("result" "samples" "saved_model")

for t in ${listFolder[@]}; do
    echo "Create folder $t if not exists"
    mkdir -p $t
done

### run experiments


listDataset=(
    "tile" "leather" 
)



echo "start training process for resunetGAN"
for t in ${listDataset[@]}; do
    version=1
    echo "Start Program $t of version $version"

    # run programming
    python3 resunetGan.py -dn $t

    sleep 5
    echo "Oops! I fell asleep for a couple seconds!"
done

echo "start training process for skipGANomaly"
for t in ${listDataset[@]}; do
    version=2
    echo "Start Program $t of version $version"

    # run programming
    python3 skip-ganomaly.py -dn $t

    sleep 5
    echo "Oops! I fell asleep for a couple seconds!"
done