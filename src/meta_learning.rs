//! Meta-Learning Rate
//!
//! How fast does the system improve at improving? The derivative of the
//! learning rate — meta-learning.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A learning measurement at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningMeasurement {
    /// Time step or episode number.
    pub step: f64,
    /// Performance score at this step.
    pub performance: f64,
    /// The task domain.
    pub domain: String,
}

/// A trajectory of learning over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningTrajectory {
    pub measurements: Vec<LearningMeasurement>,
    /// The learning rate (slope of performance improvement).
    pub learning_rate: f64,
    /// The meta-learning rate (how fast learning rate itself improves).
    pub meta_learning_rate: f64,
    /// Is learning accelerating, decelerating, or constant?
    pub learning_regime: LearningRegime,
    /// Predicted asymptotic performance (ceiling).
    pub predicted_ceiling: f64,
    /// Steps until reaching 95% of ceiling.
    pub steps_to_ceiling: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningRegime {
    /// Learning rate is increasing.
    Accelerating,
    /// Learning rate is constant.
    Linear,
    /// Learning rate is decreasing.
    Decelerating,
    /// Plateaued — no improvement.
    Plateaued,
}

/// The meta-learning rate analyzer.
pub struct MetaLearningRate {
    trajectories: Vec<LearningMeasurement>,
}

impl MetaLearningRate {
    pub fn new() -> Self {
        Self {
            trajectories: Vec::new(),
        }
    }

    pub fn add_measurement(&mut self, m: LearningMeasurement) {
        self.trajectories.push(m);
    }

    /// Analyze the learning trajectory, computing both learning rate and meta-learning rate.
    pub fn analyze(&self) -> LearningTrajectory {
        let mut measurements = self.trajectories.clone();
        measurements.sort_by(|a, b| a.step.partial_cmp(&b.step).unwrap());

        if measurements.len() < 2 {
            return LearningTrajectory {
                measurements,
                learning_rate: 0.0,
                meta_learning_rate: 0.0,
                learning_regime: LearningRegime::Plateaued,
                predicted_ceiling: 0.0,
                steps_to_ceiling: None,
            };
        }

        // Compute learning rate at each point (finite differences)
        let learning_rates = self.compute_local_learning_rates(&measurements);

        // Overall learning rate: linear fit slope
        let (learning_rate, _) = self.linear_fit(&measurements);

        // Meta-learning rate: slope of learning rates over time
        let meta_learning_rate = self.compute_meta_learning_rate(&measurements, &learning_rates);

        // Determine regime
        let learning_regime = if learning_rates.len() < 2 {
            LearningRegime::Plateaued
        } else {
            let first_half: f64 = learning_rates[..learning_rates.len() / 2].iter().sum::<f64>()
                / (learning_rates.len() / 2) as f64;
            let second_half: f64 = learning_rates[learning_rates.len() / 2..].iter().sum::<f64>()
                / (learning_rates.len() - learning_rates.len() / 2) as f64;

            if (second_half - first_half).abs() < 1e-6 {
                LearningRegime::Linear
            } else if second_half > first_half {
                LearningRegime::Accelerating
            } else {
                LearningRegime::Decelerating
            }
        };

        // Predict ceiling using logistic model: p(t) = L / (1 + exp(-k(t - t0)))
        let (predicted_ceiling, steps_to_ceiling) = self.predict_ceiling(&measurements);

        LearningTrajectory {
            measurements,
            learning_rate,
            meta_learning_rate,
            learning_regime,
            predicted_ceiling,
            steps_to_ceiling,
        }
    }

    fn compute_local_learning_rates(&self, measurements: &[LearningMeasurement]) -> Vec<f64> {
        let mut rates = Vec::new();
        for i in 1..measurements.len() {
            let dt = measurements[i].step - measurements[i - 1].step;
            let dp = measurements[i].performance - measurements[i - 1].performance;
            if dt.abs() > 1e-10 {
                rates.push(dp / dt);
            }
        }
        rates
    }

    fn compute_meta_learning_rate(&self, measurements: &[LearningMeasurement], learning_rates: &[f64]) -> f64 {
        if learning_rates.len() < 2 {
            return 0.0;
        }

        // Fit a line to learning rates to get their slope = meta-learning rate
        let n = learning_rates.len() as f64;
        let steps: Vec<f64> = (0..learning_rates.len()).map(|i| measurements[i + 1].step).collect();

        let sum_x: f64 = steps.iter().sum();
        let sum_y: f64 = learning_rates.iter().sum();
        let sum_xy: f64 = steps.iter().zip(learning_rates.iter()).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = steps.iter().map(|x| x * x).sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denom
    }

    fn linear_fit(&self, measurements: &[LearningMeasurement]) -> (f64, f64) {
        let n = measurements.len() as f64;
        let sum_x: f64 = measurements.iter().map(|m| m.step).sum();
        let sum_y: f64 = measurements.iter().map(|m| m.performance).sum();
        let sum_xy: f64 = measurements.iter().map(|m| m.step * m.performance).sum();
        let sum_x2: f64 = measurements.iter().map(|m| m.step * m.step).sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return (0.0, sum_y / n);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;
        (slope, intercept)
    }

    fn predict_ceiling(&self, measurements: &[LearningMeasurement]) -> (f64, Option<f64>) {
        if measurements.len() < 3 {
            let max_p = measurements.iter().map(|m| m.performance).fold(0.0_f64, f64::max);
            return (max_p, None);
        }

        // Use the last few points to extrapolate
        let last_n = measurements.len().min(10);
        let recent = &measurements[measurements.len() - last_n..];

        let (slope, intercept) = self.linear_fit(recent);

        // Estimate ceiling as where improvement becomes negligible
        if slope.abs() < 1e-6 {
            let ceiling = recent.last().unwrap().performance;
            return (ceiling, None);
        }

        // Simple ceiling estimate: extrapolate with diminishing returns
        let last_perf = recent.last().unwrap().performance;
        let ceiling = last_perf + slope * 100.0; // Extrapolate 100 more steps
        let ceiling = ceiling.min(1.0); // Can't exceed 1.0

        // Steps to 95% of ceiling
        let target = ceiling * 0.95;
        if slope.abs() > 1e-10 {
            let steps = (target - last_perf) / slope;
            if steps > 0.0 {
                return (ceiling, Some(steps));
            }
        }

        (ceiling, None)
    }

    /// Compare meta-learning rates across domains.
    pub fn compare_domains(trajectories: &[LearningTrajectory]) -> Vec<(String, f64)> {
        let mut results: Vec<_> = trajectories.iter()
            .filter(|t| !t.measurements.is_empty())
            .map(|t| {
                let domain = t.measurements.first().unwrap().domain.clone();
                (domain, t.meta_learning_rate)
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    /// The "meta-meta-learning" question: is the meta-learning rate itself improving?
    pub fn meta_meta_rate(trajectories: &[LearningTrajectory]) -> f64 {
        let rates: Vec<f64> = trajectories.iter()
            .map(|t| t.meta_learning_rate)
            .collect();

        if rates.len() < 2 {
            return 0.0;
        }

        let n = rates.len() as f64;
        let sum_x: f64 = (0..rates.len()).map(|i| i as f64).sum();
        let sum_y: f64 = rates.iter().sum();
        let sum_xy: f64 = (0..rates.len()).map(|i| i as f64 * rates[i]).sum();
        let sum_x2: f64 = (0..rates.len()).map(|i| (i as f64).powi(2)).sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denom
    }
}

impl Default for MetaLearningRate {
    fn default() -> Self {
        Self::new()
    }
}
