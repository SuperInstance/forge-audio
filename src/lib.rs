//! Forge Audio — Decompose audio into tiles for Plato agents

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTile {
    pub id: Uuid,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub start_sample: usize,
    pub rms: f64,
    pub peak: f64,
    pub zcr: f64, // zero-crossing rate
    pub spectral_centroid: f64,
    pub meta: HashMap<String, String>,
}

pub struct AudioDecomposer {
    pub sample_rate: u32,
    pub chunk_samples: usize,
}

impl AudioDecomposer {
    pub fn new(sample_rate: u32, chunk_duration_ms: u32) -> Self {
        let chunk_samples = ((sample_rate as u64 * chunk_duration_ms as u64) / 1000) as usize;
        Self { sample_rate, chunk_samples: chunk_samples.max(1) }
    }

    pub fn decompose(&self, samples: &[f32]) -> Vec<AudioTile> {
        samples.chunks(self.chunk_samples).enumerate().map(|(i, chunk)| {
            let rms = Self::compute_rms(chunk);
            let peak = Self::compute_peak(chunk);
            let zcr = Self::compute_zcr(chunk);
            let centroid = Self::compute_spectral_centroid(chunk);
            let mut meta = HashMap::new();
            meta.insert("chunk_index".into(), i.to_string());
            meta.insert("duration_ms".into(), ((chunk.len() as f64 / self.sample_rate as f64) * 1000.0).to_string());
            AudioTile {
                id: Uuid::new_v4(),
                samples: chunk.to_vec(),
                sample_rate: self.sample_rate,
                start_sample: i * self.chunk_samples,
                rms, peak, zcr, spectral_centroid: centroid, meta,
            }
        }).collect()
    }

    pub fn compute_rms(samples: &[f32]) -> f64 {
        if samples.is_empty() { return 0.0; }
        let sum: f64 = samples.iter().map(|s| (*s as f64).powi(2)).sum();
        (sum / samples.len() as f64).sqrt()
    }

    pub fn compute_peak(samples: &[f32]) -> f64 {
        samples.iter().map(|s| s.abs() as f64).fold(0.0, f64::max)
    }

    pub fn compute_zcr(samples: &[f32]) -> f64 {
        if samples.len() < 2 { return 0.0; }
        let crossings = samples.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
        crossings as f64 / (samples.len() - 1) as f64
    }

    /// Simple spectral centroid using DFT magnitude at frequency bins
    pub fn compute_spectral_centroid(samples: &[f32]) -> f64 {
        if samples.is_empty() { return 0.0; }
        let n = samples.len().next_power_of_two();
        let mut real: Vec<f64> = samples.iter().map(|s| *s as f64).collect();
        real.resize(n, 0.0);
        // Simple DFT for first N/2 bins (not FFT but correct for small chunks)
        let half = n / 2;
        let mut magnitudes = Vec::with_capacity(half);
        for k in 0..half {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            for t in 0..n {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
                sum_re += real[t] * angle.cos();
                sum_im += real[t] * angle.sin();
            }
            magnitudes.push((sum_re * sum_re + sum_im * sum_im).sqrt());
        }
        let total_mag: f64 = magnitudes.iter().sum();
        if total_mag == 0.0 { return 0.0; }
        let weighted: f64 = magnitudes.iter().enumerate().map(|(k, m)| k as f64 * m).sum();
        weighted / total_mag
    }

    pub fn reconstruct(tiles: &[AudioTile]) -> Vec<f32> {
        tiles.iter().flat_map(|t| t.samples.iter().copied()).collect()
    }

    pub fn filter_by_rms(&self, tiles: &[AudioTile], min_rms: f64) -> Vec<AudioTile> {
        tiles.iter().filter(|t| t.rms >= min_rms).cloned().collect()
    }

    pub fn filter_by_zcr(&self, tiles: &[AudioTile], min_zcr: f64, max_zcr: f64) -> Vec<AudioTile> {
        tiles.iter().filter(|t| t.zcr >= min_zcr && t.zcr <= max_zcr).cloned().collect()
    }

    pub fn stats(tiles: &[AudioTile]) -> AudioStats {
        if tiles.is_empty() {
            return AudioStats { chunk_count: 0, total_samples: 0, duration_seconds: 0.0, avg_rms: 0.0, peak_rms: 0.0, avg_zcr: 0.0 };
        }
        let total_samples: usize = tiles.iter().map(|t| t.samples.len()).sum();
        let avg_rms: f64 = tiles.iter().map(|t| t.rms).sum::<f64>() / tiles.len() as f64;
        let peak_rms = tiles.iter().map(|t| t.rms).fold(0.0, f64::max);
        let avg_zcr: f64 = tiles.iter().map(|t| t.zcr).sum::<f64>() / tiles.len() as f64;
        let sample_rate = tiles[0].sample_rate as f64;
        AudioStats { chunk_count: tiles.len(), total_samples, duration_seconds: total_samples as f64 / sample_rate, avg_rms, peak_rms, avg_zcr }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioStats {
    pub chunk_count: usize,
    pub total_samples: usize,
    pub duration_seconds: f64,
    pub avg_rms: f64,
    pub peak_rms: f64,
    pub avg_zcr: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f32, sr: u32, duration_ms: u32) -> Vec<f32> {
        let n = ((sr as f64 * duration_ms as f64) / 1000.0) as usize;
        (0..n).map(|i| {
            let t = i as f32 / sr as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
        }).collect()
    }

    fn silence(sr: u32, duration_ms: u32) -> Vec<f32> {
        vec![0.0f32; ((sr as f64 * duration_ms as f64) / 1000.0) as usize]
    }

    #[test]
    fn test_decompose_basic() {
        let d = AudioDecomposer::new(16000, 100); // 100ms chunks
        let samples = sine_wave(440.0, 16000, 500); // 500ms
        let tiles = d.decompose(&samples);
        assert_eq!(tiles.len(), 5); // 5 chunks of 100ms
    }

    #[test]
    fn test_decompose_partial() {
        let d = AudioDecomposer::new(16000, 100);
        let samples = sine_wave(440.0, 16000, 250); // 250ms
        let tiles = d.decompose(&samples);
        assert!(tiles.len() >= 2);
        // Last tile may be smaller
        assert!(tiles.last().unwrap().samples.len() <= 1600);
    }

    #[test]
    fn test_rms_sine() {
        let samples = sine_wave(440.0, 16000, 100);
        let rms = AudioDecomposer::compute_rms(&samples);
        assert!(rms > 0.2 && rms < 0.5); // RMS of 0.5 amplitude sine ≈ 0.354
    }

    #[test]
    fn test_rms_silence() {
        let rms = AudioDecomposer::compute_rms(&silence(16000, 100));
        assert!(rms.abs() < 0.001);
    }

    #[test]
    fn test_peak() {
        let samples = sine_wave(440.0, 16000, 100);
        let peak = AudioDecomposer::compute_peak(&samples);
        assert!((peak - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_zcr_sine() {
        // 440Hz sine at 16000Hz SR → ~440 crossings per second → ZCR ≈ 440/16000 * 2 ≈ 0.055
        let samples = sine_wave(440.0, 16000, 100);
        let zcr = AudioDecomposer::compute_zcr(&samples);
        assert!(zcr > 0.02 && zcr < 0.1);
    }

    #[test]
    fn test_zcr_silence() {
        let zcr = AudioDecomposer::compute_zcr(&silence(16000, 100));
        assert!(zcr.abs() < 0.001);
    }

    #[test]
    fn test_spectral_centroid() {
        let samples = sine_wave(440.0, 16000, 100);
        let centroid = AudioDecomposer::compute_spectral_centroid(&samples);
        assert!(centroid > 0.0); // Should detect energy around bin for 440Hz
    }

    #[test]
    fn test_reconstruct() {
        let d = AudioDecomposer::new(16000, 100);
        let original = sine_wave(440.0, 16000, 300);
        let tiles = d.decompose(&original);
        let reconstructed = AudioDecomposer::reconstruct(&tiles);
        assert_eq!(original.len(), reconstructed.len());
        for (a, b) in original.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 0.0001);
        }
    }

    #[test]
    fn test_filter_by_rms() {
        let d = AudioDecomposer::new(16000, 100);
        let mut samples = sine_wave(440.0, 16000, 200);
        samples.extend(silence(16000, 200));
        let tiles = d.decompose(&samples);
        let loud = d.filter_by_rms(&tiles, 0.05);
        assert!(loud.len() < tiles.len());
    }

    #[test]
    fn test_stats() {
        let d = AudioDecomposer::new(16000, 100);
        let tiles = d.decompose(&sine_wave(440.0, 16000, 300));
        let stats = AudioDecomposer::stats(&tiles);
        assert_eq!(stats.chunk_count, 3);
        assert!(stats.avg_rms > 0.0);
    }

    #[test]
    fn test_stats_empty() {
        let stats = AudioDecomposer::stats(&[]);
        assert_eq!(stats.chunk_count, 0);
    }

    #[test]
    fn test_tile_serialization() {
        let d = AudioDecomposer::new(16000, 100);
        let tiles = d.decompose(&sine_wave(440.0, 16000, 100));
        let json = serde_json::to_string(&tiles[0]).unwrap();
        let back: AudioTile = serde_json::from_str(&json).unwrap();
        assert_eq!(tiles[0].id, back.id);
    }

    #[test]
    fn test_empty_input() {
        let d = AudioDecomposer::new(16000, 100);
        let tiles = d.decompose(&[]);
        assert!(tiles.is_empty());
    }
}
