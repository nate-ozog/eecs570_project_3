set -e

mkdir ./external/parasail-master/build
cd ./external/parasail-master/build
cmake .. -DBUILD_SHARED_LIBS=OFF

# Use the 'j' option to parallelize the build
# make -j16
make
