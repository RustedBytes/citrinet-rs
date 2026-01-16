# citrinet-rs

Minimal Citrinet inference example in Rust using ONNX Runtime (`ort`) and `kaldi-native-fbank` for feature extraction.

## Run
```bash
cargo run --release -- \
  --model model.onnx \
  --tokens tokens.txt \
  --audio test_wavs/0.wav
```

Inputs are expected to be mono 16 kHz WAV audio and a token file formatted as `symbol index` per line (batch size 1). Defaults match the sample paths above.
