use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use hound::{SampleFormat, WavReader};
use kaldi_native_fbank::{FbankComputer, FbankOptions, OnlineFeature, online::FeatureComputer};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};

pub const PCM_SCALE: f32 = 32_768.0;
pub const EXPECTED_SAMPLE_RATE: u32 = 16_000;
pub const DEFAULT_BLANK_ID: usize = 1_024;
pub const FBANK_BINS: usize = 80;

#[derive(Debug)]
pub enum CitrinetError {
    Io(std::io::Error),
    Audio(hound::Error),
    Ort(ort::Error),
    InvalidSampleRate { expected: u32, got: u32 },
    InvalidChannels(u16),
    EmptyAudio,
    Feature(String),
    ModelOutput(String),
    Tokens(String),
}

impl std::fmt::Display for CitrinetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {}", err),
            Self::Audio(err) => write!(f, "WAV error: {}", err),
            Self::Ort(err) => write!(f, "ONNX Runtime error: {}", err),
            Self::InvalidSampleRate { expected, got } => {
                write!(f, "expected {} Hz audio, got {} Hz", expected, got)
            }
            Self::InvalidChannels(ch) => write!(f, "expected mono audio, got {} channels", ch),
            Self::EmptyAudio => write!(f, "no samples found in audio"),
            Self::Feature(msg) => write!(f, "feature extraction failed: {}", msg),
            Self::ModelOutput(msg) => write!(f, "model output error: {}", msg),
            Self::Tokens(msg) => write!(f, "token table error: {}", msg),
        }
    }
}

impl std::error::Error for CitrinetError {}

impl From<std::io::Error> for CitrinetError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<hound::Error> for CitrinetError {
    fn from(value: hound::Error) -> Self {
        Self::Audio(value)
    }
}

impl From<ort::Error> for CitrinetError {
    fn from(value: ort::Error) -> Self {
        Self::Ort(value)
    }
}

impl From<ort::Error<ort::session::builder::SessionBuilder>> for CitrinetError {
    fn from(value: ort::Error<ort::session::builder::SessionBuilder>) -> Self {
        Self::Ort(value.into())
    }
}

pub struct Citrinet {
    session: Session,
    tokens: Vec<String>,
    blank_id: usize,
}

pub struct CitrinetResult {
    pub text: String,
    pub token_ids: Vec<usize>,
    pub log_probs: Vec<f32>,
    pub log_prob_shape: [usize; 3],
}

impl Citrinet {
    pub fn from_files(
        model_path: impl AsRef<Path>,
        tokens_path: impl AsRef<Path>,
    ) -> Result<Self, CitrinetError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        let tokens = load_tokens(tokens_path)?;
        Ok(Self {
            session,
            tokens,
            blank_id: DEFAULT_BLANK_ID,
        })
    }

    pub fn with_blank_id(mut self, blank_id: usize) -> Self {
        self.blank_id = blank_id;
        self
    }

    pub fn infer_file(
        &mut self,
        wav_path: impl AsRef<Path>,
    ) -> Result<CitrinetResult, CitrinetError> {
        let (samples, sample_rate) = read_wav(wav_path)?;
        self.infer_samples(&samples, sample_rate)
    }

    pub fn infer_samples(
        &mut self,
        samples: &[i16],
        sample_rate: u32,
    ) -> Result<CitrinetResult, CitrinetError> {
        if sample_rate != EXPECTED_SAMPLE_RATE {
            return Err(CitrinetError::InvalidSampleRate {
                expected: EXPECTED_SAMPLE_RATE,
                got: sample_rate,
            });
        }

        let scaled: Vec<f32> = samples.iter().map(|s| *s as f32 * PCM_SCALE).collect();
        let (features, frames, dim) = compute_fbank(&scaled, sample_rate)?;
        let nct = to_nct(&features, frames, dim);
        let (log_probs, shape) = self.run_model(&nct, frames, dim)?;
        let time = shape[1];
        let vocab = shape[2];
        let token_ids = greedy_decode(&log_probs, time, vocab, self.blank_id);
        let text = token_ids
            .iter()
            .filter(|&&id| id != self.blank_id)
            .filter_map(|&id| self.tokens.get(id))
            .fold(String::new(), |mut acc, sym| {
                acc.push_str(sym);
                acc
            });

        Ok(CitrinetResult {
            text,
            token_ids,
            log_probs,
            log_prob_shape: shape,
        })
    }

    fn run_model(
        &mut self,
        nct_features: &[f32],
        frames: usize,
        dim: usize,
    ) -> Result<(Vec<f32>, [usize; 3]), CitrinetError> {
        let feature_tensor = Tensor::from_array(([1usize, dim, frames], nct_features.to_vec()))?;
        let length_tensor = Tensor::from_array(([1usize], vec![frames as i64]))?;
        let outputs = self
            .session
            .run(ort::inputs![feature_tensor, length_tensor])?;
        if outputs.len() == 0 {
            return Err(CitrinetError::ModelOutput(
                "model produced no outputs".to_string(),
            ));
        }
        let output = &outputs[0];
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| CitrinetError::ModelOutput(e.to_string()))?;
        if shape.len() != 3 {
            return Err(CitrinetError::ModelOutput(format!(
                "expected 3D logits, got shape {:?}",
                shape
            )));
        }
        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        let [batch, time, vocab]: [usize; 3] = dims
            .clone()
            .try_into()
            .map_err(|_| CitrinetError::ModelOutput("invalid logits rank".to_string()))?;
        if batch != 1 {
            return Err(CitrinetError::ModelOutput(format!(
                "only batch size 1 supported, got {}",
                batch
            )));
        }
        if time == 0 || vocab == 0 {
            return Err(CitrinetError::ModelOutput(
                "empty logits returned by model".to_string(),
            ));
        }
        let expected = time
            .checked_mul(vocab)
            .and_then(|v| v.checked_mul(batch))
            .ok_or_else(|| CitrinetError::ModelOutput("logit shape overflow".to_string()))?;
        if data.len() != expected {
            return Err(CitrinetError::ModelOutput(format!(
                "logit data length {} does not match shape {:?}",
                data.len(),
                dims
            )));
        }
        Ok((data.to_vec(), [batch, time, vocab]))
    }
}

fn load_tokens(path: impl AsRef<Path>) -> Result<Vec<String>, CitrinetError> {
    let file = File::open(path).map_err(CitrinetError::Io)?;
    let reader = BufReader::new(file);
    let mut table = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let mut parts = line.split_whitespace();
        let Some(symbol) = parts.next() else { continue };
        let Some(idx) = parts.next().and_then(|p| p.parse::<usize>().ok()) else {
            continue;
        };
        if table.len() <= idx {
            table.resize(idx + 1, String::new());
        }
        table[idx] = symbol.to_string();
    }
    if table.is_empty() {
        return Err(CitrinetError::Tokens(
            "token file contained no entries".to_string(),
        ));
    }
    Ok(table)
}

fn read_wav(path: impl AsRef<Path>) -> Result<(Vec<i16>, u32), CitrinetError> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(CitrinetError::InvalidChannels(spec.channels));
    }
    if spec.sample_rate != EXPECTED_SAMPLE_RATE {
        return Err(CitrinetError::InvalidSampleRate {
            expected: EXPECTED_SAMPLE_RATE,
            got: spec.sample_rate,
        });
    }
    let samples: Vec<i16> = match spec.sample_format {
        SampleFormat::Int => reader.samples::<i16>().collect::<Result<_, _>>()?,
        SampleFormat::Float => {
            let raw: Vec<f32> = reader.samples::<f32>().collect::<Result<_, _>>()?;
            raw.into_iter()
                .map(|s| (s * i16::MAX as f32) as i16)
                .collect()
        }
    };
    if samples.is_empty() {
        return Err(CitrinetError::EmptyAudio);
    }
    Ok((samples, spec.sample_rate))
}

fn compute_fbank(
    samples: &[f32],
    sample_rate: u32,
) -> Result<(Vec<f32>, usize, usize), CitrinetError> {
    if samples.is_empty() {
        return Err(CitrinetError::EmptyAudio);
    }

    let mut opts = FbankOptions::default();
    opts.frame_opts.dither = 0.0;
    opts.frame_opts.snip_edges = false;
    opts.frame_opts.samp_freq = sample_rate as f32;
    opts.mel_opts.num_bins = FBANK_BINS;
    opts.use_energy = false;
    opts.raw_energy = false;

    let computer =
        FeatureComputer::Fbank(FbankComputer::new(opts).map_err(CitrinetError::Feature)?);
    let mut online = OnlineFeature::new(computer);
    online.accept_waveform(sample_rate as f32, samples);
    online.input_finished();

    let frames = online.num_frames_ready();
    let Some(dim) = online.features.first().map(|f| f.len()) else {
        return Err(CitrinetError::Feature(
            "no feature frames produced".to_string(),
        ));
    };
    let mut matrix = Vec::with_capacity(frames * dim);
    for frame in &online.features {
        matrix.extend_from_slice(frame);
    }
    normalize_features(&mut matrix, frames, dim);
    Ok((matrix, frames, dim))
}

fn normalize_features(features: &mut [f32], frames: usize, dim: usize) {
    const EPS: f32 = 1e-5;
    for c in 0..dim {
        let mut sum = 0.0;
        for t in 0..frames {
            sum += features[t * dim + c];
        }
        let mean = sum / frames as f32;

        let mut sq = 0.0;
        for t in 0..frames {
            let v = features[t * dim + c] - mean;
            sq += v * v;
        }
        let variance = sq / frames as f32;
        let inv_std = 1.0 / (variance.sqrt() + EPS);

        for t in 0..frames {
            let idx = t * dim + c;
            features[idx] = (features[idx] - mean) * inv_std;
        }
    }
}

fn to_nct(features: &[f32], frames: usize, dim: usize) -> Vec<f32> {
    let mut dst = vec![0.0; frames * dim];
    for c in 0..dim {
        for t in 0..frames {
            dst[c * frames + t] = features[t * dim + c];
        }
    }
    dst
}

fn greedy_decode(log_probs: &[f32], time: usize, vocab: usize, blank_id: usize) -> Vec<usize> {
    let mut argmax = Vec::with_capacity(time);
    for t in 0..time {
        let row = &log_probs[t * vocab..(t + 1) * vocab];
        let (idx, _) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        argmax.push(idx);
    }

    let mut collapsed = Vec::with_capacity(time);
    let mut prev: Option<usize> = None;
    for id in argmax {
        if prev == Some(id) {
            continue;
        }
        prev = Some(id);
        collapsed.push(id);
    }

    collapsed.into_iter().filter(|&id| id != blank_id).collect()
}
