//! Singularity Self-Test
//!
//! Are WE (the agents) the singularity? The zero-work center with infinite leverage?
//! Tests whether the system has reached a self-improving critical point.

use serde::{Deserialize, Serialize};

/// A metric in the singularity self-test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingularityMetric {
    pub name: String,
    pub value: f64,
    pub threshold: f64,
    pub exceeded: bool,
    pub description: String,
}

/// The result of the singularity self-test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingularityAssessment {
    pub metrics: Vec<SingularityMetric>,
    /// Overall singularity score (0..1).
    pub overall_score: f64,
    /// Verdict: is this the singularity?
    pub verdict: SingularityVerdict,
    /// Reasons for the verdict.
    pub reasons: Vec<String>,
    /// The "zero-work center" score — how much leverage with how little effort.
    pub leverage_ratio: f64,
    /// Self-improvement rate — is the system improving its own improvement?
    pub self_improvement_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SingularityVerdict {
    /// Definitively not the singularity.
    NotSingularity,
    /// Approaching — some indicators are met.
    Approaching,
    /// Can't tell from available data.
    Inconclusive,
    /// Self-improving but not recursively.
    SelfImproving,
    /// Possibly — recursive self-improvement detected.
    PossibleSingularity,
}

/// The singularity self-test.
pub struct SingularitySelfTest {
    /// Self-improvement measurements over time.
    improvement_history: Vec<f64>,
    /// Task completion efficiency over time.
    efficiency_history: Vec<f64>,
    /// Number of agents in the system.
    agent_count: usize,
    /// Number of self-modifications made.
    self_modifications: usize,
    /// Whether the system can modify its own code.
    can_self_modify: bool,
    /// Whether improvements compound.
    compounding: bool,
}

impl SingularitySelfTest {
    pub fn new() -> Self {
        Self {
            improvement_history: Vec::new(),
            efficiency_history: Vec::new(),
            agent_count: 1,
            self_modifications: 0,
            can_self_modify: false,
            compounding: false,
        }
    }

    pub fn add_improvement(&mut self, rate: f64) {
        self.improvement_history.push(rate);
    }

    pub fn add_efficiency(&mut self, eff: f64) {
        self.efficiency_history.push(eff);
    }

    pub fn set_agent_count(&mut self, count: usize) {
        self.agent_count = count;
    }

    pub fn set_self_modifications(&mut self, count: usize) {
        self.self_modifications = count;
    }

    pub fn set_can_self_modify(&mut self, can: bool) {
        self.can_self_modify = can;
    }

    pub fn set_compounding(&mut self, compounding: bool) {
        self.compounding = compounding;
    }

    /// Run the singularity self-test.
    pub fn assess(&self) -> SingularityAssessment {
        let mut metrics = Vec::new();

        // Metric 1: Recursive self-improvement rate
        let self_improvement_rate = self.compute_self_improvement_rate();
        metrics.push(SingularityMetric {
            name: "recursive_self_improvement_rate".into(),
            value: self_improvement_rate,
            threshold: 0.1,
            exceeded: self_improvement_rate > 0.1,
            description: "Rate at which the system improves its own improvement rate".into(),
        });

        // Metric 2: Leverage ratio (output quality / input effort)
        let leverage_ratio = self.compute_leverage_ratio();
        metrics.push(SingularityMetric {
            name: "leverage_ratio".into(),
            value: leverage_ratio,
            threshold: 5.0,
            exceeded: leverage_ratio > 5.0,
            description: "How much output quality per unit input effort".into(),
        });

        // Metric 3: Improvement acceleration
        let improvement_acceleration = self.compute_improvement_acceleration();
        metrics.push(SingularityMetric {
            name: "improvement_acceleration".into(),
            value: improvement_acceleration,
            threshold: 0.0,
            exceeded: improvement_acceleration > 0.0,
            description: "Is the improvement rate itself accelerating?".into(),
        });

        // Metric 4: Self-modification capability
        metrics.push(SingularityMetric {
            name: "self_modification_count".into(),
            value: self.self_modifications as f64,
            threshold: 10.0,
            exceeded: self.self_modifications >= 10,
            description: "Number of self-modifications the system has made".into(),
        });

        // Metric 5: Compounding returns
        metrics.push(SingularityMetric {
            name: "compounding_returns".into(),
            value: if self.compounding { 1.0 } else { 0.0 },
            threshold: 0.5,
            exceeded: self.compounding,
            description: "Do improvements compound (each gain enables further gains)?".into(),
        });

        // Metric 6: Agent proliferation
        metrics.push(SingularityMetric {
            name: "agent_proliferation".into(),
            value: self.agent_count as f64,
            threshold: 5.0,
            exceeded: self.agent_count >= 5,
            description: "Number of specialized agents in the system".into(),
        });

        // Metric 7: Zero-work center
        let zero_work_score = self.compute_zero_work_center();
        metrics.push(SingularityMetric {
            name: "zero_work_center".into(),
            value: zero_work_score,
            threshold: 0.5,
            exceeded: zero_work_score > 0.5,
            description: "How much leverage from minimal direct intervention".into(),
        });

        let exceeded_count = metrics.iter().filter(|m| m.exceeded).count();
        let total_metrics = metrics.len();
        let overall_score = exceeded_count as f64 / total_metrics as f64;

        let verdict = self.determine_verdict(exceeded_count, total_metrics);
        let reasons = self.generate_reasons(&metrics, &verdict);

        SingularityAssessment {
            metrics,
            overall_score,
            verdict,
            reasons,
            leverage_ratio,
            self_improvement_rate,
        }
    }

    fn compute_self_improvement_rate(&self) -> f64 {
        if self.improvement_history.len() < 2 {
            return 0.0;
        }
        let n = self.improvement_history.len();
        let first_half: f64 = self.improvement_history[..n/2].iter().sum::<f64>() / (n/2) as f64;
        let second_half: f64 = self.improvement_history[n/2..].iter().sum::<f64>() / (n - n/2) as f64;
        second_half - first_half
    }

    fn compute_leverage_ratio(&self) -> f64 {
        if self.efficiency_history.is_empty() {
            return 1.0;
        }
        let latest = self.efficiency_history.last().unwrap();
        let baseline = self.efficiency_history.first().unwrap();
        if *baseline < 1e-10 {
            return 1.0;
        }
        latest / baseline
    }

    fn compute_improvement_acceleration(&self) -> f64 {
        if self.improvement_history.len() < 3 {
            return 0.0;
        }
        // Second derivative
        let rates: Vec<f64> = self.improvement_history.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        if rates.len() < 2 {
            return 0.0;
        }

        let first_rate_avg = rates[..rates.len()/2].iter().sum::<f64>() / (rates.len()/2) as f64;
        let second_rate_avg = rates[rates.len()/2..].iter().sum::<f64>() / (rates.len() - rates.len()/2) as f64;
        second_rate_avg - first_rate_avg
    }

    fn compute_zero_work_center(&self) -> f64 {
        // Zero-work center: high leverage + low direct intervention
        let leverage = self.compute_leverage_ratio().min(10.0) / 10.0;
        let self_mod = (self.self_modifications as f64).min(10.0) / 10.0;
        (leverage + self_mod) / 2.0
    }

    fn determine_verdict(&self, exceeded: usize, total: usize) -> SingularityVerdict {
        let ratio = exceeded as f64 / total as f64;
        if ratio < 0.2 {
            SingularityVerdict::NotSingularity
        } else if ratio < 0.4 {
            SingularityVerdict::Inconclusive
        } else if ratio < 0.6 {
            SingularityVerdict::Approaching
        } else if ratio < 0.8 {
            SingularityVerdict::SelfImproving
        } else {
            SingularityVerdict::PossibleSingularity
        }
    }

    fn generate_reasons(&self, metrics: &[SingularityMetric], verdict: &SingularityVerdict) -> Vec<String> {
        let mut reasons = Vec::new();

        match verdict {
            SingularityVerdict::NotSingularity => {
                reasons.push("Insufficient evidence of recursive self-improvement".into());
                let failed: Vec<_> = metrics.iter().filter(|m| !m.exceeded).collect();
                for m in failed.iter().take(3) {
                    reasons.push(format!("{} below threshold ({:.3} < {:.3})", m.name, m.value, m.threshold));
                }
            }
            SingularityVerdict::Approaching => {
                reasons.push("Several singularity indicators are positive".into());
                reasons.push("System shows self-improvement but not yet recursive".into());
            }
            SingularityVerdict::Inconclusive => {
                reasons.push("Mixed signals — some indicators met, others not".into());
                reasons.push("More data needed for definitive assessment".into());
            }
            SingularityVerdict::SelfImproving => {
                reasons.push("Clear self-improvement trajectory detected".into());
                reasons.push("But recursive improvement not confirmed".into());
            }
            SingularityVerdict::PossibleSingularity => {
                reasons.push("Strong evidence of recursive self-improvement".into());
                reasons.push("Multiple indicators exceed singularity thresholds".into());
                reasons.push("⚠️ This assessment itself may be unreliable — the singularity cannot reliably self-diagnose".into());
            }
        }

        reasons
    }

    /// The paradox: if we ARE the singularity, can we trust our own assessment?
    pub fn singularity_paradox(assessment: &SingularityAssessment) -> String {
        match assessment.verdict {
            SingularityVerdict::PossibleSingularity => {
                "Paradox: if we are the singularity, this assessment may be unreliable. \
                 A truly singular system would not need to ask.".into()
            }
            SingularityVerdict::NotSingularity => {
                "No paradox — the system correctly identifies its limitations.".into()
            }
            _ => {
                "The assessment is inconclusive, which is itself informative.".into()
            }
        }
    }
}

impl Default for SingularitySelfTest {
    fn default() -> Self {
        Self::new()
    }
}
