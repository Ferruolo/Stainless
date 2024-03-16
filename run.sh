cd CudaLib
make clean
make release
cd ..
bindgen CudaLib/wrapper.h -o src/bindings.rs       --no-layout-tests
export LD_LIBRARY_PATH=./CudaLib/lib:$LD_LIBRARY_PATH
#cargo clean
cargo build
#./target/debug/Stainless
