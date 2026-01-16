use std::path::PathBuf;

use clap::Parser;

use citrinet_rs::Citrinet;

#[derive(Parser, Debug)]
#[command(author, version, about = "Citrinet inference demo", long_about = None)]
struct Cli {
    /// Path to ONNX model
    #[arg(short = 'm', long, default_value = "model.onnx")]
    model: PathBuf,

    /// Path to tokens.txt
    #[arg(short = 't', long, default_value = "tokens.txt")]
    tokens: PathBuf,

    /// Path to audio file (mono 16 kHz WAV)
    #[arg(short = 'a', long, default_value = "test_wavs/0.wav")]
    audio: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();

    let mut model = Citrinet::from_files(&args.model, &args.tokens)?;
    let result = model.infer_file(&args.audio)?;

    println!("Transcript: {}", result.text);
    println!(
        "Frames: {}, tokens emitted: {}",
        result.log_prob_shape[1],
        result.token_ids.len()
    );

    Ok(())
}
