//! Boundary Mapper
//!
//! Where does the agent's capability curve sharply? What types of tasks fail?
//! Maps the frontier between what the agent can and cannot do.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A task attempted by the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAttempt {
    /// Coordinates in task-space.
    pub task_coords: Vec<f64>,
    /// Whether the task succeeded.
    pub succeeded: bool,
    /// Confidence of success prediction (0..1).
    pub predicted_confidence: f64,
    /// Task type label.
    pub task_type: String,
    /// Difficulty score (0..1).
    pub difficulty: f64,
}

/// A mapped boundary point on the capability frontier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityBoundary {
    /// Approximate boundary location.
    pub coords: Vec<f64>,
    /// Gradient of the success probability at this point.
    pub gradient_magnitude: f64,
    /// Task types that fail at this boundary.
    pub failing_types: Vec<String>,
    /// Task types that succeed near this boundary.
    pub succeeding_types: Vec<String>,
    /// Sharpness of the boundary (how quickly success drops).
    pub sharpness: f64,
}

/// Boundary mapping result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryMap {
    pub boundaries: Vec<CapabilityBoundary>,
    pub dimension: usize,
    /// Overall boundary volume fraction (what fraction of task-space is beyond capability).
    pub beyond_fraction: f64,
}

/// The boundary mapper engine.
pub struct BoundaryMapper {
    attempts: Vec<TaskAttempt>,
    dimension: usize,
    resolution: usize,
}

impl BoundaryMapper {
    pub fn new(dimension: usize) -> Self {
        Self {
            attempts: Vec::new(),
            dimension,
            resolution: 10,
        }
    }

    pub fn add_attempt(&mut self, attempt: TaskAttempt) {
        assert_eq!(attempt.task_coords.len(), self.dimension);
        self.attempts.push(attempt);
    }

    pub fn set_resolution(&mut self, r: usize) {
        self.resolution = r;
    }

    /// Map the capability boundary.
    pub fn map(&self) -> BoundaryMap {
        if self.attempts.is_empty() {
            return BoundaryMap {
                boundaries: Vec::new(),
                dimension: self.dimension,
                beyond_fraction: 0.0,
            };
        }

        let mut boundaries = Vec::new();

        // For each pair of nearby success/failure points, locate the boundary
        for i in 0..self.attempts.len() {
            for j in (i + 1)..self.attempts.len() {
                let a = &self.attempts[i];
                let b = &self.attempts[j];

                if a.succeeded == b.succeeded {
                    continue;
                }

                let (success, fail) = if a.succeeded { (a, b) } else { (b, a) };

                let dist: f64 = success.task_coords.iter()
                    .zip(fail.task_coords.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if dist > 2.0 {
                    continue; // Too far apart — not a clean boundary
                }

                // Estimate boundary midpoint
                let midpoint: Vec<f64> = success.task_coords.iter()
                    .zip(fail.task_coords.iter())
                    .map(|(a, b)| (a + b) / 2.0)
                    .collect();

                // Compute gradient: direction from success to failure
                let gradient: Vec<f64> = fail.task_coords.iter()
                    .zip(success.task_coords.iter())
                    .map(|(f, s)| (f - s) / dist)
                    .collect();

                let grad_mag = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

                // Classify task types on each side
                let mut failing_types = vec![fail.task_type.clone()];
                let mut succeeding_types = vec![success.task_type.clone()];

                // Check nearby points for additional type info
                for attempt in &self.attempts {
                    let d: f64 = attempt.task_coords.iter()
                        .zip(midpoint.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    if d < dist {
                        if attempt.succeeded && !succeeding_types.contains(&attempt.task_type) {
                            succeeding_types.push(attempt.task_type.clone());
                        } else if !attempt.succeeded && !failing_types.contains(&attempt.task_type) {
                            failing_types.push(attempt.task_type.clone());
                        }
                    }
                }

                // Sharpness: how abruptly does capability drop?
                let predicted_delta = (success.predicted_confidence - fail.predicted_confidence).abs();
                let sharpness = if dist > 0.0 { predicted_delta / dist } else { 0.0 };

                boundaries.push(CapabilityBoundary {
                    coords: midpoint,
                    gradient_magnitude: grad_mag,
                    failing_types,
                    succeeding_types,
                    sharpness,
                });
            }
        }

        // Deduplicate nearby boundaries
        boundaries = self.deduplicate(boundaries);

        // Compute beyond-fraction
        let total = self.attempts.len();
        let failed = self.attempts.iter().filter(|a| !a.succeeded).count();
        let beyond_fraction = failed as f64 / total as f64;

        BoundaryMap {
            boundaries,
            dimension: self.dimension,
            beyond_fraction,
        }
    }

    fn deduplicate(&self, boundaries: Vec<CapabilityBoundary>) -> Vec<CapabilityBoundary> {
        let mut result = Vec::new();
        for b in boundaries {
            let is_duplicate = result.iter().any(|existing: &CapabilityBoundary| {
                let dist: f64 = existing.coords.iter()
                    .zip(b.coords.iter())
                    .map(|(a, x)| (a - x).powi(2))
                    .sum::<f64>()
                    .sqrt();
                dist < 0.5
            });
            if !is_duplicate {
                result.push(b);
            }
        }
        result
    }

    /// Find the most critical boundaries — where the most task types fail.
    pub fn critical_boundaries(map: &BoundaryMap, top_k: usize) -> Vec<&CapabilityBoundary> {
        let mut indexed: Vec<_> = map.boundaries.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.failing_types.len().cmp(&a.1.failing_types.len()));
        indexed.into_iter().take(top_k).map(|(_, b)| b).collect()
    }

    /// Find the sharpest boundaries — where capability drops most abruptly.
    pub fn sharpest_boundaries(map: &BoundaryMap, top_k: usize) -> Vec<&CapabilityBoundary> {
        let mut indexed: Vec<_> = map.boundaries.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.sharpness.partial_cmp(&a.1.sharpness).unwrap());
        indexed.into_iter().take(top_k).map(|(_, b)| b).collect()
    }

    /// Compute the "capability frontier" — the theoretical maximum capability
    /// given observed boundaries.
    pub fn frontier_volume(map: &BoundaryMap) -> f64 {
        if map.boundaries.is_empty() {
            return 1.0;
        }
        // Approximate: each boundary carves out a region
        1.0 - map.beyond_fraction
    }

    pub fn attempt_count(&self) -> usize {
        self.attempts.len()
    }
}
