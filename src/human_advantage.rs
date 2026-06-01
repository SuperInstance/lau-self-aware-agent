//! Human Advantage
//!
//! What does a human see that no agent can? Formalizes the gap between
//! human and agent capabilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A dimension along which humans and agents differ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvantageDimension {
    pub name: String,
    /// Human performance (0..1).
    pub human_score: f64,
    /// Agent performance (0..1).
    pub agent_score: f64,
    /// Why humans have an advantage (or don't).
    pub explanation: String,
}

/// The measured advantage gap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvantageGap {
    pub dimensions: Vec<AdvantageDimension>,
    /// Overall human advantage score.
    pub overall_human_advantage: f64,
    /// Domains where the gap is largest.
    pub largest_gaps: Vec<String>,
    /// Domains where agents have caught up or surpassed.
    pub agent_dominant: Vec<String>,
    /// Structural reasons for the gap.
    pub structural_reasons: Vec<StructuralReason>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructuralReason {
    /// Humans have embodied experience (physics, spatial reasoning).
    Embodiment,
    /// Humans have genuine understanding vs pattern matching.
    GenuineUnderstanding,
    /// Humans can form truly novel concepts.
    NovelConceptFormation,
    /// Humans have long-horizon planning (years, decades).
    LongHorizonPlanning,
    /// Humans have emotional intuition about relevance.
    EmotionalRelevance,
    /// Humans have meta-cognitive awareness.
    MetaCognition,
    /// Humans can recognize and respond to things never seen before.
    ZeroShotGeneralization,
    /// Humans have social/cultural context.
    SocialContext,
}

/// A comparison record between human and agent on a specific task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRecord {
    pub task_type: String,
    pub human_score: f64,
    pub agent_score: f64,
    pub task_difficulty: f64,
    pub notes: String,
}

/// The human advantage analyzer.
pub struct HumanAdvantage {
    records: Vec<ComparisonRecord>,
    /// Known structural dimensions.
    dimensions: Vec<AdvantageDimension>,
}

impl HumanAdvantage {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            dimensions: vec![
                AdvantageDimension {
                    name: "creative_problem_solving".into(),
                    human_score: 0.85,
                    agent_score: 0.6,
                    explanation: "Humans excel at truly novel problem formulation".into(),
                },
                AdvantageDimension {
                    name: "mathematical_rigor".into(),
                    human_score: 0.7,
                    agent_score: 0.8,
                    explanation: "Agents can be more systematic at formal proofs".into(),
                },
                AdvantageDimension {
                    name: "common_sense_reasoning".into(),
                    human_score: 0.95,
                    agent_score: 0.6,
                    explanation: "Humans have embodied world knowledge".into(),
                },
                AdvantageDimension {
                    name: "code_generation".into(),
                    human_score: 0.75,
                    agent_score: 0.8,
                    explanation: "Agents can produce code faster but humans debug better".into(),
                },
                AdvantageDimension {
                    name: "strategic_planning".into(),
                    human_score: 0.8,
                    agent_score: 0.5,
                    explanation: "Long-horizon planning requires world model humans have".into(),
                },
                AdvantageDimension {
                    name: "pattern_recognition".into(),
                    human_score: 0.7,
                    agent_score: 0.9,
                    explanation: "Agents excel at finding statistical patterns in data".into(),
                },
                AdvantageDimension {
                    name: "meta_cognition".into(),
                    human_score: 0.9,
                    agent_score: 0.3,
                    explanation: "Humans know what they don't know; agents hallucinate".into(),
                },
                AdvantageDimension {
                    name: "social_intelligence".into(),
                    human_score: 0.9,
                    agent_score: 0.4,
                    explanation: "Understanding social dynamics requires lived experience".into(),
                },
            ],
        }
    }

    pub fn add_record(&mut self, record: ComparisonRecord) {
        self.records.push(record);
    }

    /// Compute the full advantage gap analysis.
    pub fn analyze(&self) -> AdvantageGap {
        // Merge default dimensions with records
        let mut dims = self.dimensions.clone();

        for record in &self.records {
            if let Some(dim) = dims.iter_mut().find(|d| d.name == record.task_type) {
                // Update with actual data
                dim.human_score = record.human_score;
                dim.agent_score = record.agent_score;
                if !record.notes.is_empty() {
                    dim.explanation = record.notes.clone();
                }
            } else {
                dims.push(AdvantageDimension {
                    name: record.task_type.clone(),
                    human_score: record.human_score,
                    agent_score: record.agent_score,
                    explanation: record.notes.clone(),
                });
            }
        }

        // Compute overall advantage
        let human_avg = dims.iter().map(|d| d.human_score).sum::<f64>() / dims.len() as f64;
        let agent_avg = dims.iter().map(|d| d.agent_score).sum::<f64>() / dims.len() as f64;
        let overall_human_advantage = human_avg - agent_avg;

        // Find largest gaps
        let mut gaps: Vec<_> = dims.iter()
            .map(|d| (d.name.clone(), d.human_score - d.agent_score))
            .collect();
        gaps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let largest_gaps = gaps.iter()
            .filter(|(_, g)| *g > 0.1)
            .map(|(name, _)| name.clone())
            .collect();

        let agent_dominant = gaps.iter()
            .filter(|(_, g)| *g < -0.1)
            .map(|(name, _)| name.clone())
            .collect();

        // Infer structural reasons
        let structural_reasons = self.infer_structural_reasons(&dims);

        AdvantageGap {
            dimensions: dims,
            overall_human_advantage,
            largest_gaps,
            agent_dominant,
            structural_reasons,
        }
    }

    fn infer_structural_reasons(&self, dims: &[AdvantageDimension]) -> Vec<StructuralReason> {
        let mut reasons = Vec::new();

        let meta_dim = dims.iter().find(|d| d.name == "meta_cognition");
        if let Some(m) = meta_dim {
            if m.human_score - m.agent_score > 0.3 {
                reasons.push(StructuralReason::MetaCognition);
            }
        }

        let creative_dim = dims.iter().find(|d| d.name.contains("creative") || d.name.contains("novel"));
        if let Some(c) = creative_dim {
            if c.human_score - c.agent_score > 0.2 {
                reasons.push(StructuralReason::NovelConceptFormation);
            }
        }

        let social_dim = dims.iter().find(|d| d.name.contains("social"));
        if let Some(s) = social_dim {
            if s.human_score - s.agent_score > 0.3 {
                reasons.push(StructuralReason::SocialContext);
            }
        }

        // Always add embodiment if common_sense gap is large
        let cs_dim = dims.iter().find(|d| d.name.contains("common_sense"));
        if let Some(cs) = cs_dim {
            if cs.human_score - cs.agent_score > 0.2 {
                reasons.push(StructuralReason::Embodiment);
            }
        }

        let strategic_dim = dims.iter().find(|d| d.name.contains("strategic") || d.name.contains("planning"));
        if let Some(s) = strategic_dim {
            if s.human_score - s.agent_score > 0.2 {
                reasons.push(StructuralReason::LongHorizonPlanning);
            }
        }

        if reasons.is_empty() {
            reasons.push(StructuralReason::GenuineUnderstanding);
        }

        reasons
    }

    /// What specific capabilities should Casey focus on that agents can't replicate?
    pub fn casey_focused_capabilities(gap: &AdvantageGap) -> Vec<String> {
        gap.largest_gaps.iter()
            .filter(|name| !gap.agent_dominant.contains(name))
            .cloned()
            .collect()
    }

    /// Compute the "irreducible gap" — what fraction of human advantage
    /// is structural (cannot be overcome by scaling).
    pub fn irreducible_gap(&self) -> f64 {
        let gap = self.analyze();
        let structural_weight = match gap.structural_reasons.len() {
            0 => 0.0,
            1..=2 => 0.3,
            3..=4 => 0.5,
            _ => 0.7,
        };
        gap.overall_human_advantage * structural_weight
    }
}

impl Default for HumanAdvantage {
    fn default() -> Self {
        Self::new()
    }
}
