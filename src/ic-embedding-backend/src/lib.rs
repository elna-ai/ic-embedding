mod onnx;

use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager},
    DefaultMemoryImpl,
};
use std::cell::RefCell;

const WASI_MEMORY_ID: MemoryId = MemoryId::new(0);

thread_local! {
    // The memory manager is used for simulating multiple memories.
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
}

#[ic_cdk::query]
fn get_embeddings(text: String) -> Vec<Vec<f32>> {
    let embeddings = onnx::inference(&text).unwrap();
    let vec_of_vecs: Vec<Vec<f32>> = embeddings.outer_iter().map(|row| row.to_vec()).collect();

    vec_of_vecs
}

#[ic_cdk::init]
fn init() {
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
    onnx::setup_model().unwrap();
    onnx::setup_vocab().unwrap();
}

#[ic_cdk::post_upgrade]
fn post_upgrade() {
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
    onnx::setup_model().unwrap();
    onnx::setup_vocab().unwrap();
}

// ic_cdk::export_candid!();
