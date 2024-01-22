cd CudaLib
make clean
make release
cd ..
export LD_LIBRARY_PATH=./CudaLib/:$LD_LIBRARY_PATH
cargo clean
cargo build
./target/debug/Stainless
