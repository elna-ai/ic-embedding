use ndarray::Axis;
use prost::Message;
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::BertVocab;
use std::cell::RefCell;
use tract_onnx::prelude::*;

type Array = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>;
type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

thread_local! {
    static MODEL: RefCell<Option<Model>> = RefCell::new(None);
    static VOCAB: RefCell<Option<BertVocab>>=RefCell::new(None);
}

const VOCAB_BYTES: &'static [u8] = include_bytes!("../assets/onnx/vocab.txt");
const MODEL_BYTES: &'static [u8] = include_bytes!("../assets/onnx/model.onnx");
const TARGET_LEN: usize = 256;

pub fn setup_model() -> TractResult<()> {
    let bytes = bytes::Bytes::from_static(MODEL_BYTES);
    let proto: tract_onnx::pb::ModelProto = tract_onnx::pb::ModelProto::decode(bytes)?;
    let model = tract_onnx::onnx()
        .model_for_proto_model(&proto)?
        .into_optimized()?
        .into_runnable()?;
    MODEL.with_borrow_mut(|m| {
        *m = Some(model);
    });
    Ok(())
}

pub fn setup_vocab() -> Result<(), ()> {
    let vocab = BertVocab::from_bytes(VOCAB_BYTES).unwrap();
    VOCAB.with_borrow_mut(|m| {
        *m = Some(vocab);
    });
    Ok(())
}

fn pad_vector<T: Default + Clone>(input_ids: Vec<T>, target_len: usize) -> Vec<T> {
    let mut padded = input_ids.clone();
    padded.resize(target_len, T::default());
    padded
}

fn normalize(v: &Array) -> Array {
    let norms = v.map_axis(Axis(1), |row| row.iter().map(|x| x * x).sum::<f32>().sqrt());
    let norms = norms.mapv(|x| if x == 0.0 { 1e-12 } else { x });
    let normalized = v / &norms.insert_axis(Axis(1));
    normalized
}

fn fmt(n: u64) -> String {
    n.to_string()
        .as_bytes()
        .rchunks(3)
        .rev()
        .map(std::str::from_utf8)
        .collect::<Result<Vec<&str>, _>>()
        .unwrap()
        .join("_")
}

fn get_tokens(sentence: &str) -> Result<(Vec<i64>, Vec<i32>, Vec<i8>), ()> {
    let instructions_before = ic_cdk::api::instruction_counter();

    let lowercase: bool = true;
    let strip_accents: bool = true;
    VOCAB.with_borrow(|vocab| {
        let vocab = vocab.clone().unwrap();

        let bert_tokenizer = BertTokenizer::from_existing_vocab(vocab, lowercase, strip_accents);

        let tokens = bert_tokenizer.encode(
            sentence,
            None,
            TARGET_LEN,
            &TruncationStrategy::DoNotTruncate,
            0,
        );

        let input_ids = tokens.token_ids;
        let input_ids: Vec<i64> = pad_vector(input_ids, TARGET_LEN);

        // Generate and print the attention mask
        let attention_mask = input_ids
            .iter()
            .map(|&id| if id == 0 { 0 } else { 1 })
            .collect();

        let attention_mask = pad_vector(attention_mask, TARGET_LEN);

        let segment_ids = tokens.segment_ids;
        let token_type_ids = pad_vector(segment_ids, TARGET_LEN);

        let instructions = ic_cdk::api::instruction_counter() - instructions_before;

        ic_cdk::println!(
            "Tokenization:     {:>12} Wasm instructions",
            fmt(instructions)
        );

        Ok((input_ids, attention_mask, token_type_ids))
    })
}

pub fn inference(sentence: &str) -> Result<Array, ()> {
    let (input_ids, attention_mask, token_type_ids) = get_tokens(sentence).unwrap();
    let instructions_before = ic_cdk::api::instruction_counter();

    MODEL.with_borrow(|model| {
        let model = model.as_ref().unwrap();

        let input_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, TARGET_LEN),
            input_ids.iter().map(|&x| x as i64).collect(),
        )
        .unwrap()
        .into();

        let attention_mask_non_tensor = attention_mask.clone();

        let attention_mask: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, TARGET_LEN),
            attention_mask.iter().map(|&x| x as i64).collect(),
        )
        .unwrap()
        .into();

        let target_len = attention_mask.shape()[1];

        let token_type_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, TARGET_LEN),
            token_type_ids.iter().map(|&x| x as i64).collect(),
        )
        .unwrap()
        .into();

        let outputs = model
            .run(tvec!(
                input_ids.into(),
                attention_mask.into(),
                token_type_ids.into()
            ))
            .unwrap();

        let logits: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<ndarray::IxDynImpl>> =
            outputs[0].to_array_view::<f32>().unwrap();

        // Expand the attention mask

        let input_mask_expanded = tract_ndarray::Array3::from_shape_fn(
            (1, target_len, logits.shape()[2]),
            |(_, i, _)| attention_mask_non_tensor[i],
        );

        let logits_fixed: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Ix3> = logits
            .into_dimensionality()
            .expect("Expected 3-dimensional array");

        let input_mask_expanded = input_mask_expanded.mapv(|x| x as f32);

        let weighted_last_hidden_state: ndarray::ArrayBase<
            ndarray::OwnedRepr<f32>,
            ndarray::Dim<[usize; 3]>,
        > = &logits_fixed * &input_mask_expanded;

        let sum_weighted = weighted_last_hidden_state.sum_axis(ndarray::Axis(1));
        let sum_mask = input_mask_expanded.sum_axis(ndarray::Axis(1));

        let mean_pooled = &sum_weighted / &sum_mask.mapv(|x| x.max(1e-9));

        let embeddings = normalize(&mean_pooled);

        let instructions = ic_cdk::api::instruction_counter() - instructions_before;

        ic_cdk::println!("Inference:     {:>12} Wasm instructions", fmt(instructions));

        Ok(embeddings)
    })
}
