SRC=nbody.cu
EXE=nbody_gpu
nvcc -O3 -DSHMOO -o $EXE $SRC -lm

echo $EXE

K=1024
for i in {1..10}
do
    ./$EXE $K
    K=$(($K*2))
done
