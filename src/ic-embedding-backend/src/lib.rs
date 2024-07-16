use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager},
    DefaultMemoryImpl,
};
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::BertVocab;
use std::cell::RefCell;

const WASI_MEMORY_ID: MemoryId = MemoryId::new(0);

thread_local! {
    // The memory manager is used for simulating multiple memories.
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
}

#[ic_cdk::query]
fn greet(name: String) -> String {
    format!("Hello, {}!", name)
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

const VOCAB: &'static [u8] = include_bytes!("../assets/bert-base-uncased-vocab.txt");

#[ic_cdk::query]
fn tokenize(text: String) -> (Vec<i64>, Vec<i64>) {
    let vocab = BertVocab::from_bytes(VOCAB).unwrap();

    let lowercase: bool = true;
    let strip_accents: bool = true;

    let bert_tokenizer = BertTokenizer::from_existing_vocab(vocab, lowercase, strip_accents);

    let tokens = bert_tokenizer.encode(&text, None, 128, &TruncationStrategy::LongestFirst, 0);
    ic_cdk::println!("{:?}", tokens);
    let input_ids = tokens.token_ids;

    // Generate and print the attention mask

    let attention_mask: Vec<i64> = input_ids
        .iter()
        .map(|&id| if id == 0 { 0 } else { 1 })
        .collect();

    (input_ids, attention_mask)
}

// ic_cdk::export_candid!();
