//! Blind Spot Detector
//!
//! Identifies mathematical domains where the agent's models systematically
//! underperform — structural gaps in understanding.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A mathematical domain label.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Domain(pub String);

/// Performance record for a single evaluation in a domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    pub domain: Domain,
    pub score: f64,         // 0..1
    pub difficulty: f64,    // 0..1
    pub expected: f64,      // what a well-calibrated agent would score
    pub timestamp: u64,
}

/// A detected blind spot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindSpot {
    pub domain: Domain,
    /// Systematic underperformance magnitude.
    pub gap: f64,
    /// Confidence that this is a real blind spot (not noise).
    pub confidence: f64,
    /// Suggested cause categories.
    pub causes: Vec<BlindSpotCause>,
    /// Related domains that may share the root cause.
    pub related_domains: Vec<Domain>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlindSpotCause {
    /// Training data deficit in this domain.
    DataDeficit,
    /// The domain requires reasoning depth beyond model capacity.
    DepthExceeded,
    /// The domain conflicts with patterns learned elsewhere (negative transfer).
    NegativeTransfer,
    /// The domain requires symbolic rather than statistical reasoning.
    SymbolicGap,
    /// The domain requires compositional reasoning the model lacks.
    CompositionFailure,
    /// Unknown structural limitation.
    Unknown,
}

/// The blind spot detector engine.
pub struct BlindSpotDetector {
    records: Vec<PerformanceRecord>,
    /// Minimum samples before declaring a blind spot.
    min_samples: usize,
    /// Gap threshold (fraction below expected) to flag.
    gap_threshold: f64,
}

impl BlindSpotDetector {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            min_samples: 5,
            gap_threshold: 0.15,
        }
    }

    pub fn record(&mut self, rec: PerformanceRecord) {
        self.records.push(rec);
    }

    pub fn records(&self) -> &[PerformanceRecord] {
        &self.records
    }

    /// Aggregate performance by domain.
    fn aggregate_by_domain(&self) -> HashMap<Domain, (f64, f64, usize)> {
        let mut map: HashMap<Domain, (f64, f64, usize)> = HashMap::new();
        for r in &self.records {
            let entry = map.entry(r.domain.clone()).or_insert((0.0, 0.0, 0));
            entry.0 += r.score;
            entry.1 += r.expected;
            entry.2 += 1;
        }
        map
    }

    /// Detect blind spots across all recorded domains.
    pub fn detect(&self) -> Vec<BlindSpot> {
        let agg = self.aggregate_by_domain();
        let mut spots = Vec::new();

        for (domain, (total_score, total_expected, count)) in &agg {
            if *count < self.min_samples {
                continue;
            }
            let avg_score = total_score / *count as f64;
            let avg_expected = total_expected / *count as f64;
            let gap = avg_expected - avg_score;

            if gap > self.gap_threshold {
                let confidence = 1.0 - 1.0 / (*count as f64);
                let causes = self.infer_causes(domain, gap, *count);
                let related = self.find_related(domain, gap);
                spots.push(BlindSpot {
                    domain: domain.clone(),
                    gap,
                    confidence,
                    causes,
                    related_domains: related,
                });
            }
        }

        spots.sort_by(|a, b| b.gap.partial_cmp(&a.gap).unwrap());
        spots
    }

    fn infer_causes(&self, domain: &Domain, gap: f64, count: usize) -> Vec<BlindSpotCause> {
        let mut causes = Vec::new();
        let domain_str = domain.0.to_lowercase();

        // Heuristic cause inference based on domain name and gap magnitude
        if gap > 0.5 {
            causes.push(BlindSpotCause::DepthExceeded);
        }
        if domain_str.contains("symbol") || domain_str.contains("proof") || domain_str.contains("logic") {
            causes.push(BlindSpotCause::SymbolicGap);
        }
        if domain_str.contains("compos") || domain_str.contains("combin") {
            causes.push(BlindSpotCause::CompositionFailure);
        }
        if count < 10 {
            causes.push(BlindSpotCause::DataDeficit);
        }

        // Check for negative transfer: domains with similar names but opposing patterns
        let records_in_domain: Vec<_> = self.records.iter()
            .filter(|r| r.domain == *domain)
            .collect();

        let high_difficulty_low_score = records_in_domain.iter()
            .filter(|r| r.difficulty > 0.7 && r.score < 0.3)
            .count();

        if high_difficulty_low_score > count / 2 {
            causes.push(BlindSpotCause::NegativeTransfer);
        }

        if causes.is_empty() {
            causes.push(BlindSpotCause::Unknown);
        }

        causes
    }

    fn find_related(&self, domain: &Domain, gap: f64) -> Vec<Domain> {
        let agg = self.aggregate_by_domain();
        agg.iter()
            .filter(|(d, (score, expected, count))| {
                *d != domain && *count >= self.min_samples && (*expected - *score / *count as f64).abs() < gap * 0.5
            })
            .map(|(d, _)| d.clone())
            .collect()
    }

    /// Compute the overall blind spot coverage — what fraction of tested domains
    /// have significant blind spots.
    pub fn blind_spot_coverage(&self) -> f64 {
        let agg = self.aggregate_by_domain();
        let total: usize = agg.values().map(|(_, _, c)| if *c >= self.min_samples { 1 } else { 0 }).sum();
        if total == 0 { return 0.0; }
        let blind = self.detect().len();
        blind as f64 / total as f64
    }

    /// Suggest interventions for a given blind spot.
    pub fn suggest_interventions(spot: &BlindSpot) -> Vec<String> {
        let mut suggestions = Vec::new();
        for cause in &spot.causes {
            match cause {
                BlindSpotCause::DataDeficit => suggestions.push("Increase training data in this domain".into()),
                BlindSpotCause::DepthExceeded => suggestions.push("Break tasks into shallower sub-tasks".into()),
                BlindSpotCause::NegativeTransfer => suggestions.push("Isolate this domain's training to prevent interference".into()),
                BlindSpotCause::SymbolicGap => suggestions.push("Augment with symbolic reasoning tools".into()),
                BlindSpotCause::CompositionFailure => suggestions.push("Train on compositional benchmarks explicitly".into()),
                BlindSpotCause::Unknown => suggestions.push("Further investigation needed — run targeted diagnostics".into()),
            }
        }
        suggestions
    }
}

impl Default for BlindSpotDetector {
    fn default() -> Self {
        Self::new()
    }
}
