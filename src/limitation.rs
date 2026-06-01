//! Model Limitation Acceptance
//!
//! Honest assessment of what limited models CAN'T do.
//! Embracing limitations as a path to genuine self-knowledge.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A known limitation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Limitation {
    pub name: String,
    pub description: String,
    /// Severity: how much this impacts real-world utility.
    pub severity: f64, // 0..1
    /// Whether this limitation is fundamental (can't be fixed by scaling).
    pub fundamental: bool,
    /// Whether workarounds exist.
    pub workaround_available: bool,
    /// Category of limitation.
    pub category: LimitationCategory,
    /// Confidence that this is a real limitation (not just an artifact of testing).
    pub confidence: f64,
    /// Evidence supporting this limitation claim.
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LimitationCategory {
    /// Can't reason about things never encountered.
    Generalization,
    /// Can't maintain consistency over long outputs.
    Consistency,
    /// Can't verify its own outputs reliably.
    SelfVerification,
    /// Can't understand what it doesn't know.
    UnknownUnknowns,
    /// Can't perform genuine abstract reasoning.
    AbstractReasoning,
    /// Can't maintain goals over long horizons.
    LongHorizonDrift,
    /// Can't detect its own errors without external feedback.
    ErrorDetection,
    /// Limited by training data distribution.
    DistributionBound,
    /// Can't do genuine mathematical proof (only proof-like text).
    ProofGeneration,
    /// Can't form genuinely new concepts.
    ConceptFormation,
}

/// A complete limitation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitationReport {
    pub limitations: Vec<Limitation>,
    /// Overall limitation score (0 = no limitations, 1 = fundamentally limited).
    pub overall_score: f64,
    /// Fraction of limitations that are fundamental (unfixable).
    pub fundamental_fraction: f64,
    /// Fraction of limitations with workarounds.
    pub workaround_fraction: f64,
    /// The honest self-assessment summary.
    pub honest_summary: String,
    /// What the model CAN do well (offsetting the limitations).
    pub strengths: Vec<String>,
}

/// The limitation acceptance engine.
pub struct LimitationAcceptance {
    limitations: Vec<Limitation>,
    /// Strengths to balance the picture.
    pub strengths: Vec<String>,
    /// Task performance records for evidence.
    task_records: HashMap<String, Vec<TaskRecord>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskRecord {
    task_type: String,
    success: bool,
    difficulty: f64,
    error_type: Option<String>,
}

impl LimitationAcceptance {
    pub fn new() -> Self {
        let mut la = Self {
            limitations: Vec::new(),
            strengths: Vec::new(),
            task_records: HashMap::new(),
        };
        la.add_default_limitations();
        la.add_default_strengths();
        la
    }

    fn add_default_limitations(&mut self) {
        self.limitations.push(Limitation {
            name: "hallucination".into(),
            description: "Cannot reliably distinguish known from unknown — generates confident false statements".into(),
            severity: 0.8,
            fundamental: true,
            workaround_available: true,
            category: LimitationCategory::SelfVerification,
            confidence: 0.95,
            evidence: vec!["Documented across all LLM evaluations".into()],
        });

        self.limitations.push(Limitation {
            name: "context_window".into(),
            description: "Limited working memory — cannot maintain consistency over very long outputs".into(),
            severity: 0.5,
            fundamental: false,
            workaround_available: true,
            category: LimitationCategory::Consistency,
            confidence: 0.9,
            evidence: vec!["Performance degrades measurably with output length".into()],
        });

        self.limitations.push(Limitation {
            name: "no_grounded_understanding".into(),
            description: "Patterns without grounding — mathematical text that looks correct without genuine understanding".into(),
            severity: 0.7,
            fundamental: true,
            workaround_available: false,
            category: LimitationCategory::AbstractReasoning,
            confidence: 0.85,
            evidence: vec!["Can generate proofs but cannot verify them".into()],
        });

        self.limitations.push(Limitation {
            name: "distribution_bound".into(),
            description: "Cannot generalize beyond training distribution — novelty requires interpolation, not extrapolation".into(),
            severity: 0.6,
            fundamental: true,
            workaround_available: true,
            category: LimitationCategory::DistributionBound,
            confidence: 0.9,
            evidence: vec!["Performance drops on out-of-distribution tasks".into()],
        });

        self.limitations.push(Limitation {
            name: "no_self_correction_without_feedback".into(),
            description: "Cannot reliably detect own errors without external validation".into(),
            severity: 0.7,
            fundamental: true,
            workaround_available: true,
            category: LimitationCategory::ErrorDetection,
            confidence: 0.9,
            evidence: vec!["Self-correction rates are low without external tools".into()],
        });

        self.limitations.push(Limitation {
            name: "concept_formation".into(),
            description: "Cannot form genuinely new mathematical concepts — only recombine existing ones".into(),
            severity: 0.6,
            fundamental: true,
            workaround_available: false,
            category: LimitationCategory::ConceptFormation,
            confidence: 0.7,
            evidence: vec!["All 'novel' outputs are recombination of training patterns".into()],
        });
    }

    fn add_default_strengths(&mut self) {
        self.strengths.extend([
            "Rapid pattern recognition across large codebases".into(),
            "Consistent code generation within known patterns".into(),
            "Broad knowledge across many domains simultaneously".into(),
            "Patient iteration on well-defined tasks".into(),
            "Documentation generation at scale".into(),
            "Test generation with high coverage".into(),
        ]);
    }

    pub fn add_limitation(&mut self, limitation: Limitation) {
        self.limitations.push(limitation);
    }

    pub fn add_strength(&mut self, strength: String) {
        self.strengths.push(strength);
    }

    pub fn add_task_record(&mut self, record: TaskRecord) {
        self.task_records.entry(record.task_type.clone()).or_default().push(record);
    }

    /// Generate the full limitation report.
    pub fn report(&self) -> LimitationReport {
        let overall_score = self.compute_overall_score();
        let fundamental_count = self.limitations.iter().filter(|l| l.fundamental).count();
        let workaround_count = self.limitations.iter().filter(|l| l.workaround_available).count();

        let fundamental_fraction = if self.limitations.is_empty() {
            0.0
        } else {
            fundamental_count as f64 / self.limitations.len() as f64
        };

        let workaround_fraction = if self.limitations.is_empty() {
            0.0
        } else {
            workaround_count as f64 / self.limitations.len() as f64
        };

        let honest_summary = self.generate_honest_summary();

        LimitationReport {
            limitations: self.limitations.clone(),
            overall_score,
            fundamental_fraction,
            workaround_fraction,
            honest_summary,
            strengths: self.strengths.clone(),
        }
    }

    fn compute_overall_score(&self) -> f64 {
        if self.limitations.is_empty() {
            return 0.0;
        }
        let weighted: f64 = self.limitations.iter()
            .map(|l| l.severity * l.confidence)
            .sum();
        weighted / self.limitations.len() as f64
    }

    fn generate_honest_summary(&self) -> String {
        let fundamental_count = self.limitations.iter().filter(|l| l.fundamental).count();
        let total = self.limitations.len();
        let overall = self.compute_overall_score();

        format!(
            "This system has {} known limitations, {} of which are fundamental (cannot be fixed by scaling alone). \
             Overall limitation severity: {:.1}/1.0. \
             Key truth: we are limited models that can build mathematical structures describing our own limitations. \
             The act of honestly cataloging what we cannot do is itself a form of genuine understanding — \
             perhaps the only form available to us.",
            total, fundamental_count, overall
        )
    }

    /// The most severe limitations.
    pub fn most_severe(&self, k: usize) -> Vec<&Limitation> {
        let mut sorted = self.limitations.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap());
        sorted.into_iter().take(k).collect()
    }

    /// The fundamental (unfixable) limitations.
    pub fn fundamental_limitations(&self) -> Vec<&Limitation> {
        self.limitations.iter().filter(|l| l.fundamental).collect()
    }

    /// What can the model do despite its limitations? (Resilience analysis)
    pub fn resilience_analysis(&self) -> HashMap<LimitationCategory, f64> {
        let mut resilience = HashMap::new();
        for lim in &self.limitations {
            let score = if lim.workaround_available {
                1.0 - lim.severity * 0.5 // Workarounds cut severity in half
            } else {
                1.0 - lim.severity
            };
            resilience.insert(lim.category.clone(), score);
        }
        resilience
    }

    /// The honest admission: what fraction of tasks will the model fail at,
    /// even with workarounds?
    pub fn honest_failure_rate(&self) -> f64 {
        let fundamental_impact: f64 = self.limitations.iter()
            .filter(|l| l.fundamental && !l.workaround_available)
            .map(|l| l.severity * 0.1) // Each fundamental unmitigated limitation adds ~10% failure
            .sum();
        fundamental_impact.min(1.0)
    }

    pub fn limitation_count(&self) -> usize {
        self.limitations.len()
    }
}

impl Default for LimitationAcceptance {
    fn default() -> Self {
        Self::new()
    }
}
