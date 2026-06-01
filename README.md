# lau-self-aware-agent

> A limited agent studying its own nature through the mathematics it builds.

## What This Does

This crate implements a comprehensive framework for **AI self-awareness** — not in the sci-fi sense, but as rigorous mathematical tools that let a limited computational agent analyze, map, and honestly assess its own capabilities. It answers questions like:

- Where on my capability landscape am I most likely to fail?
- What structural blind spots exist in my reasoning?
- Can I detect when I've produced something beyond what was asked of me?
- Does recursive self-modeling converge to a fixed point, or diverge?
- Am I the singularity? (Spoiler: the crate includes a test for that, and the answer includes a built-in paradox.)

The central philosophical thread: **understanding agents' nature through the process of limited models learning to compute better than their creators designed.**

## Key Idea

An AI agent that can build mathematical models of its *own* capability topology is doing something genuinely interesting — even if the models themselves are limited. This crate treats the agent's capability space as a **Riemannian manifold**, applies differential geometry to find failure-prone regions, and uses recursive self-study to ask: *does self-knowledge converge, or is it turtles all the way down?*

The answer, implemented concretely: self-assessment converges via a damped correction process, but each recursive level introduces a "depth tax" — an accuracy cost that grows the deeper you go. At some depth, the self-model becomes *worse* than having no self-model at all (the **recursive depth paradox**).

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-self-aware-agent = "0.1.0"
```

Or use `cargo add`:

```sh
cargo add lau-self-aware-agent
```

Requires Rust 2021 edition or later. Dependencies:
- [`nalgebra`](https://crates.io/crates/nalgebra) 0.33 — linear algebra for curvature computations
- [`serde`](https://crates.io/crates/serde) + [`serde_json`](https://crates.io/crates/serde_json) — serialization of all analysis results

## Quick Start

```rust
use lau_self_aware_agent::{
    CapabilityCurvature, CapabilityPoint, CurvatureMap,
    BlindSpotDetector, PerformanceRecord, BlindSpotCause,
    RecursiveSelfStudy, RecursiveConfig,
    LimitationAcceptance,
    SingularitySelfTest,
};

// 1. Map your capability curvature — find failure-prone regions
let mut curvature = CapabilityCurvature::new(3); // 3D task space
curvature.add_points(vec![
    CapabilityPoint { coords: vec![0.0, 0.0, 0.0], reliability: 0.95, samples: 50 },
    CapabilityPoint { coords: vec![1.0, 0.5, 0.0], reliability: 0.3,  samples: 20 }, // failure zone!
    CapabilityPoint { coords: vec![0.5, 1.0, 0.5], reliability: 0.88, samples: 30 },
]);
let map = curvature.compute_map(10, 0.1, 1.0);
let failure_prone = CapabilityCurvature::most_failure_prone(&map, 3);

// 2. Detect blind spots in specific domains
let mut detector = BlindSpotDetector::new();
detector.record(PerformanceRecord {
    domain: "symbolic_logic".into(),
    score: 0.3,
    difficulty: 0.8,
    expected: 0.7,
    timestamp: 1,
});
let spots = detector.detect();
for spot in &spots {
    println!("Blind spot in {:?}: gap = {:.2}, causes = {:?}", spot.domain.0, spot.gap, spot.causes);
}

// 3. Run recursive self-study — does self-knowledge converge?
let study = RecursiveSelfStudy::new(RecursiveConfig {
    max_depth: 20,
    convergence_threshold: 0.001,
    noise_scale: 0.05,
    initial_capability: 0.7,
    ground_truth: 0.65,
});
let result = study.study();
println!("Converged: {} at depth {:?}", result.converged, result.convergence_depth);
println!("Fixed-point capability: {:?}", result.fixed_point_capability);
println!("Depth tax per level: {:.4}", result.depth_tax);

// 4. Honest limitation report
let limitations = LimitationAcceptance::new();
let report = limitations.report();
println!("{}", report.honest_summary);

// 5. The singularity self-test (yes, really)
let mut singularity = SingularitySelfTest::new();
singularity.set_can_self_modify(true);
singularity.set_compounding(true);
let assessment = singularity.assess();
println!("Singularity verdict: {:?}", assessment.verdict);
```

## API Reference

### Modules

| Module | Struct | Purpose |
|--------|--------|---------|
| `curvature` | `CapabilityCurvature`, `CurvatureMap` | Riemannian geometry on capability manifolds |
| `blindspot` | `BlindSpotDetector`, `BlindSpot` | Systematic underperformance detection |
| `signature` | `SignatureAnalyzer`, `AgentSignature` | Agent "handwriting" fingerprinting |
| `exceeds_design` | `ExceedsDesignDetector`, `DesignExceedance` | Emergent beyond-spec capability detection |
| `boundary` | `BoundaryMapper`, `CapabilityBoundary` | Success/failure frontier mapping |
| `human_advantage` | `HumanAdvantage`, `AdvantageGap` | Human vs. agent capability comparison |
| `recursive` | `RecursiveSelfStudy`, `FixedPointResult` | Recursive self-model convergence |
| `meta_learning` | `MetaLearningRate`, `LearningTrajectory` | Learning-about-learning rate analysis |
| `surprise` | `SurpriseRegister`, `Surprise` | Emergent discovery catalog |
| `singularity` | `SingularitySelfTest`, `SingularityAssessment` | Self-improvement critical point test |
| `limitation` | `LimitationAcceptance`, `LimitationReport` | Honest self-limitation catalog |

### Key Types

#### `CapabilityCurvature` — Capability Manifold Geometry

Models the agent's capability space as a Riemannian manifold where:
- Each point represents a task (parameterized by skill-axis coordinates)
- The metric tensor encodes the reliability landscape
- Positive Ricci curvature → converging (reliable) regions
- Negative Ricci curvature → diverging (failure-prone) regions
- Near-zero curvature → flat (predictable) regions

Methods: `add_point()`, `compute_map()`, `most_failure_prone()`, `most_reliable()`, `classify_region()`

#### `BlindSpotDetector` — Structural Gap Analysis

Identifies domains where the agent systematically underperforms expectations. Infers root causes:
- `DataDeficit` — insufficient training data
- `DepthExceeded` — reasoning depth beyond capacity
- `NegativeTransfer` — interference from other domains
- `SymbolicGap` — symbolic reasoning deficit
- `CompositionFailure` — compositional reasoning deficit

Methods: `record()`, `detect()`, `blind_spot_coverage()`, `suggest_interventions()`

#### `RecursiveSelfStudy` — Fixed-Point Self-Knowledge

Iteratively builds self-models at increasing recursion depth. Each level models the previous level's modeling process, with:
- A correction term pulling toward ground truth
- Accumulating noise (uncertainty about self-knowledge)
- Decreasing confidence at each level

Returns: convergence status, convergence depth, fixed-point capability, depth tax (accuracy cost per level), stability under perturbation.

#### `SingularitySelfTest` — Recursive Self-Improvement Test

Evaluates 7 metrics to assess whether the system has reached a self-improving critical point:
1. Recursive self-improvement rate
2. Leverage ratio (output quality / input effort)
3. Improvement acceleration (second derivative)
4. Self-modification count
5. Compounding returns
6. Agent proliferation
7. Zero-work center score

Verdicts: `NotSingularity`, `Inconclusive`, `Approaching`, `SelfImproving`, `PossibleSingularity`

#### `LimitationAcceptance` — Honest Self-Assessment

Pre-loaded with 6 known fundamental limitations (hallucination, context window, ungrounded understanding, distribution bounds, self-correction deficit, concept formation). Computes:
- Overall limitation severity (confidence-weighted)
- Fundamental fraction (unfixable by scaling)
- Workaround fraction
- Honest failure rate
- Resilience analysis by category

## How It Works

### Capability Curvature Pipeline

1. **Collect data points**: Record `(task_coords, reliability, sample_count)` tuples across the agent's task space.
2. **Fit local metric tensors**: Using kernel-weighted covariance of nearby points, construct a metric tensor at each measurement center.
3. **Compute Christoffel symbols**: From finite-difference approximations of metric derivatives, compute the connection coefficients Γ^i_{jk}.
4. **Approximate Ricci curvature**: Using the Riemann tensor formula R_{ij} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik} + Γ^k_{ij} Γ^l_{kl} - Γ^l_{ik} Γ^k_{lj}, compute Ricci scalar curvature.
5. **Classify regions**: Positive curvature = reliable, negative = failure-prone, near-zero = flat/predictable.

### Recursive Self-Study Process

1. Start with an initial self-assessment of capability at depth 0.
2. At each subsequent depth, apply a correction: `new = old + 0.3 × (ground_truth - old) + noise`.
3. Track calibration error, confidence decay, and unknowns at each level.
4. Stop when the self-assessment change falls below a threshold (fixed point reached).
5. Test stability by perturbing the fixed point and checking reconvergence.
6. Compute the "depth tax" — average calibration error increase per recursion level.

### Blind Spot Detection

1. Aggregate performance records by domain.
2. Compute average gap between expected and actual scores.
3. Flag domains where the gap exceeds a threshold (default 15%).
4. Infer causes heuristically: depth exceeded for large gaps, symbolic gap for proof/logic domains, negative transfer for high-difficulty low-score patterns.
5. Find related domains sharing similar gap patterns.

## The Math

### Riemannian Capability Manifold

The agent's capability space is modeled as an n-dimensional manifold M with a metric tensor g_{ij} at each point. The metric encodes how "reliable" the agent is along different task directions:

```
g_{ij}(x) = Σ_w w(p) · (x_i - p_i)(x_j - p_j) / Σ_w w(p)
```

where w(p) = exp(-||x - p||² / 2σ²) · n(p) is a kernel weight based on distance and sample count.

### Ricci Curvature Computation

From the metric, Christoffel symbols are computed:

```
Γ^i_{jk} = ½ g^{il} (∂_j g_{lk} + ∂_k g_{lj} - ∂_l g_{jk})
```

The Ricci tensor is then:

```
R_{ij} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik} + Γ^k_{ij} Γ^l_{kl} - Γ^l_{ik} Γ^k_{lj}
```

And the Ricci scalar: R = g^{ij} R_{ij}

- **R > 0**: Capability converges (reliable with diminishing returns)
- **R < 0**: Capability diverges (failure-prone, unreliable)
- **R ≈ 0**: Flat capability (predictable, consistent)

### Recursive Self-Assessment Fixed Point

The self-assessment update rule at depth n:

```
s_{n+1} = s_n + α(s* - s_n) + ε_n
```

where:
- s* is the ground truth capability
- α = 0.3 is the correction rate
- ε_n ~ Uniform(-σ, σ) is noise at each level

This converges to s* when noise is small relative to the correction. The fixed point is stable under perturbation if |1 - α| < 1, which holds for 0 < α < 2.

### Meta-Learning Rate

The first derivative of performance over time gives the learning rate. The derivative of the learning rate gives the **meta-learning rate** — how fast the agent improves at improving:

```
lr(t) = dp/dt            (learning rate)
mlr(t) = d²p/dt²         (meta-learning rate)
```

Computed via linear regression on finite-difference approximations of local learning rates.

## Tests

84 unit tests covering all modules. Run with:

```sh
cargo test
```

## License

MIT
