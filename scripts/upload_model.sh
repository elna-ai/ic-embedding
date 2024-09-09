#!/bin/bash
set -e

dfx canister call ic-embedding-backend clear_vocab_bytes
dfx canister call ic-embedding-backend clear_model_bytes
ic-file-uploader ic-embedding-backend append_vocab_bytes src/ic-embedding-backend/assets/onnx/vocab.txt
ic-file-uploader ic-embedding-backend append_model_bytes src/ic-embedding-backend/assets/onnx/model.onnx
dfx canister call ic-embedding-backend setup