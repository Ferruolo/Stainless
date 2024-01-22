// use bindgen::Builder;
fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=native=./CudaLib/");

    //
    println!("cargo:rustc-link-lib=dylib=CudaLib");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    // println!("cargo:rerun-if-changed=library.cuh");

}
