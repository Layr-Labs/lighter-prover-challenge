pub mod config;
pub mod hash;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
mod metal;

#[cfg(test)]
pub mod p3;
