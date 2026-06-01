//! Surprise Register
//!
//! What has the system discovered that no single agent could have found alone?
//! Emergent discoveries from multi-agent interaction.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A discovery made by the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Discovery {
    pub id: String,
    pub description: String,
    /// Which agents contributed to this discovery.
    pub contributors: Vec<String>,
    /// How surprising is this discovery (0..1).
    pub surprise_score: f64,
    /// Would a single agent have found this? 
    pub single_agent_accessible: bool,
    /// The domain of the discovery.
    pub domain: String,
    /// Evidence supporting this discovery.
    pub evidence: Vec<String>,
}

/// A registered surprise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Surprise {
    pub discovery: Discovery,
    /// Why this is surprising — the gap between expected and observed.
    pub surprise_reason: String,
    /// The "emergence factor" — how much better the multi-agent result is than
    /// the sum of individual contributions.
    pub emergence_factor: f64,
    /// Category of surprise.
    pub category: SurpriseCategory,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurpriseCategory {
    /// Discovery that contradicts established assumptions.
    AssumptionViolation,
    /// Capability that emerged from interaction, not present in any individual.
    EmergentCapability,
    /// Pattern visible only from the meta-level, invisible to individual agents.
    MetaPattern,
    /// Fixed point or attractor discovered in the dynamics.
    AttractorDiscovery,
    /// Unexpected self-similarity at different scales.
    ScaleInvariance,
    /// Something genuinely new, not an interpolation of training data.
    GenuineNovelty,
}

/// A record of what each agent found individually.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFinding {
    pub agent_id: String,
    pub finding: String,
    pub domain: String,
    pub quality: f64,
}

/// The surprise register.
pub struct SurpriseRegister {
    discoveries: Vec<Discovery>,
    agent_findings: Vec<AgentFinding>,
    /// Expected baseline — what we'd predict without collaboration.
    baselines: HashMap<String, f64>,
}

impl SurpriseRegister {
    pub fn new() -> Self {
        Self {
            discoveries: Vec::new(),
            agent_findings: Vec::new(),
            baselines: HashMap::new(),
        }
    }

    pub fn add_discovery(&mut self, discovery: Discovery) {
        self.discoveries.push(discovery);
    }

    pub fn add_agent_finding(&mut self, finding: AgentFinding) {
        self.agent_findings.push(finding);
    }

    pub fn set_baseline(&mut self, domain: String, expected: f64) {
        self.baselines.insert(domain, expected);
    }

    /// Register all surprises from accumulated discoveries.
    pub fn register(&self) -> Vec<Surprise> {
        let mut surprises = Vec::new();

        for disc in &self.discoveries {
            // Only multi-agent discoveries are surprising
            if disc.contributors.len() < 2 && disc.single_agent_accessible {
                continue;
            }

            let surprise_reason = self.explain_surprise(disc);
            let emergence_factor = self.compute_emergence_factor(disc);
            let category = self.classify_surprise(disc);

            surprises.push(Surprise {
                discovery: disc.clone(),
                surprise_reason,
                emergence_factor,
                category,
            });
        }

        surprises.sort_by(|a, b| b.emergence_factor.partial_cmp(&a.emergence_factor).unwrap());
        surprises
    }

    fn explain_surprise(&self, disc: &Discovery) -> String {
        if disc.single_agent_accessible {
            "Found by multiple agents simultaneously but accessible to any single agent".into()
        } else if disc.contributors.len() > 2 {
            format!("Required {} agents collaborating — no subset could produce this", disc.contributors.len())
        } else {
            "Emergent from agent interaction — not predictable from individual capabilities".into()
        }
    }

    fn compute_emergence_factor(&self, disc: &Discovery) -> f64 {
        // Sum of individual agent qualities in this domain
        let individual_quality: f64 = disc.contributors.iter()
            .filter_map(|agent| {
                self.agent_findings.iter()
                    .filter(|f| f.agent_id == *agent && f.domain == disc.domain)
                    .map(|f| f.quality)
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
            })
            .sum();

        // Emergence factor = actual quality / sum of individual qualities
        let baseline = self.baselines.get(&disc.domain).copied().unwrap_or(0.5);

        if individual_quality > 0.0 {
            let actual_quality = disc.evidence.len() as f64 * disc.surprise_score;
            actual_quality / individual_quality
        } else {
            1.0 + disc.surprise_score
        }
    }

    fn classify_surprise(&self, disc: &Discovery) -> SurpriseCategory {
        if !disc.single_agent_accessible && disc.contributors.len() > 2 {
            SurpriseCategory::EmergentCapability
        } else if disc.description.to_lowercase().contains("pattern") {
            SurpriseCategory::MetaPattern
        } else if disc.description.to_lowercase().contains("fixed point") || disc.description.to_lowercase().contains("attractor") {
            SurpriseCategory::AttractorDiscovery
        } else if disc.description.to_lowercase().contains("scale") || disc.description.to_lowercase().contains("fractal") {
            SurpriseCategory::ScaleInvariance
        } else if disc.surprise_score > 0.8 {
            SurpriseCategory::GenuineNovelty
        } else {
            SurpriseCategory::AssumptionViolation
        }
    }

    /// What fraction of discoveries were surprises (not single-agent accessible)?
    pub fn surprise_fraction(&self) -> f64 {
        if self.discoveries.is_empty() {
            return 0.0;
        }
        let surprising = self.discoveries.iter()
            .filter(|d| !d.single_agent_accessible || d.contributors.len() >= 2)
            .count();
        surprising as f64 / self.discoveries.len() as f64
    }

    /// The top-k most surprising discoveries.
    pub fn top_surprises(&self, k: usize) -> Vec<Discovery> {
        let mut sorted = self.discoveries.clone();
        sorted.sort_by(|a, b| b.surprise_score.partial_cmp(&a.surprise_score).unwrap());
        sorted.into_iter().take(k).collect()
    }

    pub fn discovery_count(&self) -> usize {
        self.discoveries.len()
    }
}

impl Default for SurpriseRegister {
    fn default() -> Self {
        Self::new()
    }
}
