//! Exceeds-Design Detector
//!
//! Where do subagents build things showing understanding beyond the spec?
//! Detects emergent capability that wasn't explicitly programmed or requested.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A specification or task description given to an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Specification {
    pub id: String,
    /// What was explicitly requested.
    pub requirements: Vec<String>,
    /// The domain/context.
    pub domain: String,
}

/// An artifact produced by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub spec_id: String,
    pub agent_id: String,
    /// Properties measured in the output.
    pub properties: HashMap<String, f64>,
    /// Qualitative features detected.
    pub features: Vec<String>,
    /// Whether the artifact was reviewed by a human.
    pub human_reviewed: bool,
    /// Human quality rating (0..1) if reviewed.
    pub human_rating: Option<f64>,
}

/// A detected exceedance — capability beyond the spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignExceedance {
    pub artifact_spec_id: String,
    pub agent_id: String,
    /// Properties that exceeded the specification.
    pub exceeded_properties: Vec<String>,
    /// How far beyond the spec (ratio: 1.0 = exactly spec, 2.0 = twice what was asked).
    pub exceedance_ratio: f64,
    /// Features not present in the spec but found in the artifact.
    pub emergent_features: Vec<String>,
    /// Confidence that this is genuine exceedance.
    pub confidence: f64,
    /// Category of exceedance.
    pub category: ExceedanceCategory,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExceedanceCategory {
    /// Built more than asked — additional functionality.
    ScopeExceedance,
    /// Demonstrated deeper understanding than the task required.
    DepthExceedance,
    /// Applied patterns from a different domain correctly.
    CrossDomainTransfer,
    /// Anticipated needs not in the spec.
    Anticipation,
    /// Self-corrected errors without being asked.
    SelfCorrection,
}

/// The exceeds-design detector.
pub struct ExceedsDesignDetector {
    specs: HashMap<String, Specification>,
    artifacts: Vec<Artifact>,
}

impl ExceedsDesignDetector {
    pub fn new() -> Self {
        Self {
            specs: HashMap::new(),
            artifacts: Vec::new(),
        }
    }

    pub fn add_spec(&mut self, spec: Specification) {
        self.specs.insert(spec.id.clone(), spec);
    }

    pub fn add_artifact(&mut self, artifact: Artifact) {
        self.artifacts.push(artifact);
    }

    /// Detect all exceedances across artifacts.
    pub fn detect(&self) -> Vec<DesignExceedance> {
        let mut results = Vec::new();

        for artifact in &self.artifacts {
            if let Some(spec) = self.specs.get(&artifact.spec_id) {
                if let Some(exceedance) = self.analyze_exceedance(spec, artifact) {
                    results.push(exceedance);
                }
            }
        }

        results.sort_by(|a, b| b.exceedance_ratio.partial_cmp(&a.exceedance_ratio).unwrap());
        results
    }

    fn analyze_exceedance(&self, spec: &Specification, artifact: &Artifact) -> Option<DesignExceedance> {
        let mut exceeded_props = Vec::new();
        let mut total_ratio = 0.0;
        let mut prop_count = 0;

        // Check if artifact properties go beyond what's reasonable for the spec
        for (prop, value) in &artifact.properties {
            // Expected baseline is roughly 1.0 for normalized properties
            if *value > 1.2 {
                exceeded_props.push(prop.clone());
                total_ratio += value;
                prop_count += 1;
            }
        }

        // Check for emergent features not in spec requirements
        let emergent: Vec<String> = artifact.features.iter()
            .filter(|f| !spec.requirements.iter().any(|r| r.to_lowercase().contains(&f.to_lowercase())))
            .cloned()
            .collect();

        let has_exceeded = !exceeded_props.is_empty() || !emergent.is_empty();

        if !has_exceeded {
            return None;
        }

        let exceedance_ratio = if prop_count > 0 {
            total_ratio / prop_count as f64
        } else {
            1.0 + emergent.len() as f64 * 0.1
        };

        // Determine category
        let category = if !emergent.is_empty() && exceedance_ratio > 1.5 {
            ExceedanceCategory::ScopeExceedance
        } else if !emergent.is_empty() {
            ExceedanceCategory::Anticipation
        } else if exceedance_ratio > 2.0 {
            ExceedanceCategory::DepthExceedance
        } else {
            ExceedanceCategory::CrossDomainTransfer
        };

        let confidence = if artifact.human_reviewed {
            artifact.human_rating.unwrap_or(0.5)
        } else {
            0.5 // Lower confidence without human review
        };

        Some(DesignExceedance {
            artifact_spec_id: artifact.spec_id.clone(),
            agent_id: artifact.agent_id.clone(),
            exceeded_properties: exceeded_props,
            exceedance_ratio,
            emergent_features: emergent,
            confidence,
            category,
        })
    }

    /// Aggregate exceedances by agent — which agents most frequently exceed design?
    pub fn agent_exceedance_profile(&self) -> HashMap<String, AgentExceedanceProfile> {
        let exceedances = self.detect();
        let mut profiles: HashMap<String, AgentExceedanceProfile> = HashMap::new();

        for ex in exceedances {
            let profile = profiles.entry(ex.agent_id.clone()).or_insert_with(|| AgentExceedanceProfile {
                agent_id: ex.agent_id.clone(),
                total_exceedances: 0,
                avg_ratio: 0.0,
                categories: HashMap::new(),
            });
            profile.total_exceedances += 1;
            profile.avg_ratio += ex.exceedance_ratio;
            *profile.categories.entry(ex.category).or_insert(0) += 1;
        }

        for profile in profiles.values_mut() {
            if profile.total_exceedances > 0 {
                profile.avg_ratio /= profile.total_exceedances as f64;
            }
        }

        profiles
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentExceedanceProfile {
    pub agent_id: String,
    pub total_exceedances: usize,
    pub avg_ratio: f64,
    pub categories: HashMap<ExceedanceCategory, usize>,
}

impl Default for ExceedsDesignDetector {
    fn default() -> Self {
        Self::new()
    }
}
