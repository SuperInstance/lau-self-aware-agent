//! Signature Analysis
//!
//! What patterns does an agent reproduce across outputs? Identifies the
//! "handwriting" — structural, stylistic, and mathematical signatures
//! that distinguish one model from another.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A feature extracted from agent output for signature analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureFeature {
    pub name: String,
    pub value: f64,
}

/// The full agent signature — a fingerprint of consistent patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSignature {
    pub agent_id: String,
    /// Feature vector characterizing the agent's "handwriting".
    pub features: Vec<SignatureFeature>,
    /// Overall distinctiveness score (how unique is this agent).
    pub distinctiveness: f64,
    /// Confidence based on sample size.
    pub confidence: f64,
}

/// A recorded output for analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSample {
    pub agent_id: String,
    /// Structural metrics extracted from the output.
    pub metrics: HashMap<String, f64>,
    /// Text-based features (word counts, patterns, etc.)
    pub text_features: HashMap<String, f64>,
}

/// The signature analyzer.
pub struct SignatureAnalyzer {
    samples: Vec<OutputSample>,
    /// Known signature features to track.
    tracked_features: Vec<String>,
}

impl SignatureAnalyzer {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            tracked_features: vec![
                "avg_function_length".into(),
                "comment_density".into(),
                "test_count_ratio".into(),
                "error_handling_density".into(),
                "abstraction_depth".into(),
                "naming_convention_score".into(),
                "modularity_score".into(),
                "doc_coverage".into(),
                "type_complexity".into(),
                "pattern_consistency".into(),
                " recursion_frequency".into(),
                "struct_count".into(),
                "trait_usage".into(),
                "unsafe_usage".into(),
                "generics_depth".into(),
            ],
        }
    }

    pub fn add_sample(&mut self, sample: OutputSample) {
        self.samples.push(sample);
    }

    /// Analyze the signature of a specific agent.
    pub fn analyze(&self, agent_id: &str) -> Option<AgentSignature> {
        let agent_samples: Vec<_> = self.samples.iter()
            .filter(|s| s.agent_id == agent_id)
            .collect();

        if agent_samples.is_empty() {
            return None;
        }

        // Compute mean and variance for each tracked feature
        let mut features = Vec::new();
        for feat_name in &self.tracked_features {
            let values: Vec<f64> = agent_samples.iter()
                .filter_map(|s| s.metrics.get(feat_name).copied())
                .collect();

            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = if values.len() > 1 {
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
                } else {
                    0.0
                };
                features.push(SignatureFeature {
                    name: format!("{}_mean", feat_name),
                    value: mean,
                });
                features.push(SignatureFeature {
                    name: format!("{}_variance", feat_name),
                    value: variance,
                });
            }
        }

        // Add text features
        let all_text_keys: Vec<String> = agent_samples.iter()
            .flat_map(|s| s.text_features.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for key in &all_text_keys {
            let values: Vec<f64> = agent_samples.iter()
                .filter_map(|s| s.text_features.get(key).copied())
                .collect();
            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                features.push(SignatureFeature {
                    name: format!("text_{}_mean", key),
                    value: mean,
                });
            }
        }

        // Distinctiveness: compare with other agents
        let distinctiveness = self.compute_distinctiveness(agent_id, &features);

        let confidence = (1.0 - (-(agent_samples.len() as f64) / 10.0).exp()).min(1.0);

        Some(AgentSignature {
            agent_id: agent_id.to_string(),
            features,
            distinctiveness,
            confidence,
        })
    }

    fn compute_distinctiveness(&self, agent_id: &str, features: &[SignatureFeature]) -> f64 {
        let other_ids: Vec<String> = self.samples.iter()
            .filter(|s| s.agent_id != agent_id)
            .map(|s| s.agent_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        if other_ids.is_empty() {
            return 1.0; // Unique by default
        }

        let feat_map: HashMap<&str, f64> = features.iter().map(|f| (f.name.as_str(), f.value)).collect();

        // Compute distinctiveness directly from other agents' samples
        // (avoid recursive analyze call which would cause infinite recursion)
        let mut max_dist = 0.0_f64;
        for other_id in &other_ids {
            let other_samples: Vec<_> = self.samples.iter()
                .filter(|s| s.agent_id == *other_id)
                .collect();

            let mut other_feats: HashMap<String, f64> = HashMap::new();
            for feat_name in &self.tracked_features {
                let values: Vec<f64> = other_samples.iter()
                    .filter_map(|s| s.metrics.get(feat_name).copied())
                    .collect();
                if !values.is_empty() {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    other_feats.insert(format!("{}_mean", feat_name), mean);
                    other_feats.insert(format!("{}_variance", feat_name), 0.0);
                }
            }

            let mut dist = 0.0;
            for (name, val) in &feat_map {
                if let Some(oval) = other_feats.get(*name) {
                    dist += (val - oval).powi(2);
                }
            }
            max_dist = max_dist.max(dist);
        }

        // Normalize to 0..1
        (1.0 - (-max_dist).exp()).min(1.0)
    }

    /// Compare two agents' signatures.
    pub fn compare(&self, agent_a: &str, agent_b: &str) -> Option<SignatureComparison> {
        let sig_a = self.analyze(agent_a)?;
        let sig_b = self.analyze(agent_b)?;

        let map_a: HashMap<&str, f64> = sig_a.features.iter().map(|f| (f.name.as_str(), f.value)).collect();
        let map_b: HashMap<&str, f64> = sig_b.features.iter().map(|f| (f.name.as_str(), f.value)).collect();

        let mut divergent = Vec::new();
        let mut convergent = Vec::new();

        for (name, va) in &map_a {
            if let Some(vb) = map_b.get(name) {
                let diff = (va - vb).abs();
                if diff > 0.3 {
                    divergent.push((name.to_string(), *va, *vb));
                } else if diff < 0.05 {
                    convergent.push((name.to_string(), *va, *vb));
                }
            }
        }

        let overall_distance = {
            let mut d = 0.0;
            for (name, va) in &map_a {
                if let Some(vb) = map_b.get(name) {
                    d += (va - vb).powi(2);
                }
            }
            d.sqrt()
        };

        Some(SignatureComparison {
            agent_a: agent_a.to_string(),
            agent_b: agent_b.to_string(),
            overall_distance,
            divergent_features: divergent,
            convergent_features: convergent,
        })
    }

    /// Find the most consistent features (lowest variance) across all of an agent's outputs.
    pub fn most_consistent_features(&self, agent_id: &str) -> Vec<(String, f64)> {
        let agent_samples: Vec<_> = self.samples.iter()
            .filter(|s| s.agent_id == agent_id)
            .collect();

        let mut result = Vec::new();
        for feat_name in &self.tracked_features {
            let values: Vec<f64> = agent_samples.iter()
                .filter_map(|s| s.metrics.get(feat_name).copied())
                .collect();
            if values.len() > 1 {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
                result.push((feat_name.clone(), variance));
            }
        }
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureComparison {
    pub agent_a: String,
    pub agent_b: String,
    pub overall_distance: f64,
    pub divergent_features: Vec<(String, f64, f64)>,
    pub convergent_features: Vec<(String, f64, f64)>,
}

impl Default for SignatureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
