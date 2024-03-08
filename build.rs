// use bindgen::Builder;
fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=native=./CudaLib/lib");
    // //
    // // //
    println!("cargo:rustc-link-lib=dylib=Cuda");
//

}
