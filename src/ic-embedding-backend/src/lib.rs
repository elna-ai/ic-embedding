mod onnx;
mod storage;

use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager},
    DefaultMemoryImpl,
};
use std::cell::RefCell;

use onnx::{setup_model, setup_vocab};

const WASI_MEMORY_ID: MemoryId = MemoryId::new(0);

const VOCAB_FILE: &str = "vocab.txt";
const MODEL_FILE: &str = "model.onnx";

thread_local! {
    // The memory manager is used for simulating multiple memories.
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
}

#[target_feature(enable = "simd128")]
#[ic_cdk::update]
fn get_embeddings(text: String) -> Vec<f32> {
    let embeddings = onnx::inference(&text).unwrap();
    let flattened_vec: Vec<f32> = embeddings.iter().cloned().collect();

    flattened_vec
}

#[ic_cdk::init]
fn init() {
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
}

#[ic_cdk::post_upgrade]
fn post_upgrade() {
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
}

/// Clears the Vocab file.
/// This is used for incremental chunk uploading of large files.
#[ic_cdk::update]
fn clear_vocab_bytes() {
    storage::clear_bytes(VOCAB_FILE);
}

/// Clears the model file.
/// This is used for incremental chunk uploading of large files.
#[ic_cdk::update]
fn clear_model_bytes() {
    storage::clear_bytes(MODEL_FILE);
}

/// Appends the given chunk to the Vocab file.
/// This is used for incremental chunk uploading of large files.
#[ic_cdk::update]
fn append_vocab_bytes(bytes: Vec<u8>) {
    storage::append_bytes(VOCAB_FILE, bytes);
}

/// Appends the given chunk to the model file.
/// This is used for incremental chunk uploading of large files.
#[ic_cdk::update]
fn append_model_bytes(bytes: Vec<u8>) {
    storage::append_bytes(MODEL_FILE, bytes);
}

/// Once the model files have been incrementally uploaded,
/// this function loads them into in-memory models.
#[ic_cdk::update]
fn setup() -> Result<(), String> {
    setup_vocab(storage::bytes(VOCAB_FILE))
        .map_err(|err| format!("Failed to setup model: {:?}", err))?;
    setup_model(storage::bytes(MODEL_FILE)).map_err(|err| format!("Failed to setup model: {}", err))
}

// #[ic_cdk::update]
// fn test() {
//     println!("hello");
// }

ic_cdk::export_candid!();
