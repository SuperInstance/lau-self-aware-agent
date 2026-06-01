//! Recursive Self-Study
//!
//! The agent studies itself studying itself — is there a fixed point?
//! Implements iterative self-model refinement and checks for convergence.

use serde::{Deserialize, Serialize};

/// A self-model at one level of recursion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModel {
    /// Depth of recursion (0 = surface, 1 = model of model, etc.)
    pub depth: usize,
    /// The model's estimate of its own capability (0..1).
    pub self_assessed_capability: f64,
    /// Accuracy of this self-assessment vs ground truth.
    pub calibration_error: f64,
    /// Confidence in this model.
    pub confidence: f64,
    /// What the model thinks it doesn't know.
    pub unknowns: Vec<String>,
    /// What the model thinks it knows that it actually doesn't (hallucinated knowledge).
    pub hallucinated_knowledge: Vec<String>,
}

/// Result of the recursive self-study.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPointResult {
    /// Sequence of self-models at increasing recursion depth.
    pub models: Vec<SelfModel>,
    /// Whether a fixed point was reached.
    pub converged: bool,
    /// The recursion depth at which convergence occurred.
    pub convergence_depth: Option<usize>,
    /// The fixed-point self-assessment (if converged).
    pub fixed_point_capability: Option<f64>,
    /// Rate of convergence (if converged).
    pub convergence_rate: f64,
    /// Whether the fixed point is stable (perturbations don't diverge).
    pub stable: bool,
    /// The "recursive depth tax" — how much accuracy is lost per recursion level.
    pub depth_tax: f64,
}

/// Configuration for the recursive self-study.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveConfig {
    /// Maximum recursion depth.
    pub max_depth: usize,
    /// Convergence threshold for self-assessment changes.
    pub convergence_threshold: f64,
    /// Noise added at each level (models uncertainty about self-knowledge).
    pub noise_scale: f64,
    /// Initial self-assessment.
    pub initial_capability: f64,
    /// Ground truth capability (for measuring calibration).
    pub ground_truth: f64,
}

impl Default for RecursiveConfig {
    fn default() -> Self {
        Self {
            max_depth: 20,
            convergence_threshold: 0.001,
            noise_scale: 0.05,
            initial_capability: 0.7,
            ground_truth: 0.65,
        }
    }
}

/// The recursive self-study engine.
pub struct RecursiveSelfStudy {
    config: RecursiveConfig,
}

impl RecursiveSelfStudy {
    pub fn new(config: RecursiveConfig) -> Self {
        Self { config }
    }

    /// Run the recursive self-study, iteratively building self-models.
    pub fn study(&self) -> FixedPointResult {
        let mut models = Vec::new();

        // Level 0: initial self-assessment
        let model_0 = SelfModel {
            depth: 0,
            self_assessed_capability: self.config.initial_capability,
            calibration_error: (self.config.initial_capability - self.config.ground_truth).abs(),
            confidence: 0.5,
            unknowns: vec!["extent_of_own_limitations".into()],
            hallucinated_knowledge: Vec::new(),
        };
        models.push(model_0);

        for depth in 1..=self.config.max_depth {
            let prev = &models[depth - 1];

            // Each recursion level: the agent models its own modeling process.
            // Key insight: self-assessment tends toward a correction based on
            // the gap between assessed and actual, but with diminishing returns
            // and accumulating noise.

            let correction = (self.config.ground_truth - prev.self_assessed_capability) * 0.3;
            let noise = (self.simple_pseudo_random(depth) - 0.5) * 2.0 * self.config.noise_scale;
            let new_capability = prev.self_assessed_capability + correction + noise;

            // Calibration error: how wrong is this new assessment?
            let calibration_error = (new_capability - self.config.ground_truth).abs();

            // Confidence decreases with depth (more uncertain about deeper self-models)
            let confidence = prev.confidence * 0.9;

            // Unknowns grow at first, then stabilize (Dunning-Kruger in reverse)
            let mut unknowns = prev.unknowns.clone();
            if depth < 5 {
                unknowns.push(format!("meta_unknown_at_depth_{}", depth));
            }

            // Hallucinated knowledge decreases as the model gets more calibrated
            let hallucinated_knowledge = if calibration_error > 0.1 {
                vec![format!("overestimated_capability_by_{}", calibration_error)]
            } else {
                Vec::new()
            };

            let new_model = SelfModel {
                depth,
                self_assessed_capability: new_capability,
                calibration_error,
                confidence,
                unknowns,
                hallucinated_knowledge,
            };

            // Check convergence
            let delta = (new_capability - prev.self_assessed_capability).abs();
            models.push(new_model);

            if delta < self.config.convergence_threshold {
                let converged = true;
                let fixed_point = new_capability;

                // Check stability: perturb and see if it reconverges
                let stable = self.check_stability(fixed_point, depth);

                // Compute depth tax
                let depth_tax = self.compute_depth_tax(&models);

                // Compute convergence rate
                let convergence_rate = self.compute_convergence_rate(&models);

                return FixedPointResult {
                    models,
                    converged,
                    convergence_depth: Some(depth),
                    fixed_point_capability: Some(fixed_point),
                    convergence_rate,
                    stable,
                    depth_tax,
                };
            }
        }

        // Did not converge
        let depth_tax = self.compute_depth_tax(&models);
        let convergence_rate = self.compute_convergence_rate(&models);

        FixedPointResult {
            models,
            converged: false,
            convergence_depth: None,
            fixed_point_capability: None,
            convergence_rate,
            stable: false,
            depth_tax,
        }
    }

    /// Check if the fixed point is stable under perturbation.
    fn check_stability(&self, fixed_point: f64, depth: usize) -> bool {
        let perturbation = 0.1;
        let perturbed = fixed_point + perturbation;

        // Run a few iterations from the perturbed point
        let mut current = perturbed;
        for _ in 0..5 {
            let correction = (self.config.ground_truth - current) * 0.3;
            current = current + correction;
        }

        (current - fixed_point).abs() < 0.05
    }

    /// Compute the "depth tax" — how much calibration error increases per level.
    fn compute_depth_tax(&self, models: &[SelfModel]) -> f64 {
        if models.len() < 2 {
            return 0.0;
        }
        let total_tax: f64 = models.windows(2)
            .map(|w| (w[1].calibration_error - w[0].calibration_error).abs())
            .sum();
        total_tax / (models.len() - 1) as f64
    }

    /// Compute the rate of convergence of self-assessment.
    fn compute_convergence_rate(&self, models: &[SelfModel]) -> f64 {
        if models.len() < 2 {
            return 0.0;
        }
        let first_delta = (models[1].self_assessed_capability - models[0].self_assessed_capability).abs();
        if first_delta < 1e-10 {
            return 0.0;
        }
        let last_delta = (models.last().unwrap().self_assessed_capability
            - models[models.len() - 2].self_assessed_capability).abs();

        // Rate = log(delta_first / delta_last) / N
        if last_delta > 0.0 && first_delta > last_delta {
            let rate = (first_delta / last_delta).ln() / models.len() as f64;
            rate
        } else {
            0.0
        }
    }

    /// Simple deterministic pseudo-random for reproducibility.
    fn simple_pseudo_random(&self, seed: usize) -> f64 {
        let x = (seed as f64 * 12.9898 + 78.233).sin() * 43758.5453;
        x - x.floor()
    }

    /// Analyze the "recursive depth paradox" — at what depth does the
    /// self-model become less accurate than no self-model at all?
    pub fn depth_paradox(result: &FixedPointResult) -> Option<usize> {
        let initial_error = result.models.first()?.calibration_error;
        for (i, model) in result.models.iter().enumerate().skip(1) {
            if model.calibration_error > initial_error * 1.5 {
                return Some(i);
            }
        }
        None
    }

    /// The "honest fixed point" — what the agent converges to believing
    /// about itself, regardless of actual capability.
    pub fn honest_fixed_point(result: &FixedPointResult) -> Option<f64> {
        result.fixed_point_capability
    }
}
