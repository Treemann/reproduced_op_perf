# rccl-tests for AMD GPU: https://github.com/ROCm/rccl-tests.git
# mkdir build
# cd build
# export GPU_TARGETS="gfx942" # MI300 series
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/mpi;/path/to/rocm" -DUSE_MPI=ON ..
# make

comm_op=reduce_scatter # alltoall, alltoallv
platform=mi308 # h20 etc.

# single-node
for ngpu in 2 4 8
do
    echo "Running ${comm_op} bandwidth test for ${ngpu} gpus..."
    LOG=${platform}-${comm_op}-${ngpu}gpus.log
    mpirun -np ${ngpu} -allow-run-as-root ./build/${comm_op}_perf -b 128 -e 8G -f 2 -g 1 -d half | tee $LOG
done
