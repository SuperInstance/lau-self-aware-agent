//! Capability Curvature Map
//!
//! Measures the Ricci curvature of the agent's own capability manifold.
//! Flat regions = reliable capability. Curved regions = failure-prone.
//! Models capability as a Riemannian manifold where the metric tensor
//! encodes the reliability landscape.

use nalgebra::{DMatrix, DVector, Dynamic, OMatrix};
use serde::{Deserialize, Serialize};

/// A point on the capability manifold, parameterized by task coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityPoint {
    /// Coordinates in the task-space (each dimension = a skill axis).
    pub coords: Vec<f64>,
    /// Observed reliability at this point (0..1).
    pub reliability: f64,
    /// Sample count backing this observation.
    pub samples: usize,
}

/// The Ricci scalar curvature computed at a region of the manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureMeasurement {
    pub coords: Vec<f64>,
    /// Ricci scalar curvature — positive = converging (reliable),
    /// negative = diverging (failure-prone), near-zero = flat (predictable).
    pub ricci_scalar: f64,
    /// Sectional curvatures along each 2-plane.
    pub sectional: Vec<f64>,
    /// Confidence in the measurement.
    pub confidence: f64,
}

/// Local metric tensor at a point (symmetric positive-definite).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTensor {
    pub dimension: usize,
    /// Row-major flattened matrix.
    pub data: Vec<f64>,
}

impl MetricTensor {
    pub fn identity(dim: usize) -> Self {
        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Self { dimension: dim, data }
    }

    pub fn to_matrix(&self) -> DMatrix<f64> {
        DMatrix::from_vec(self.dimension, self.dimension, self.data.clone())
    }

    pub fn from_matrix(m: &DMatrix<f64>) -> Self {
        let dim = m.nrows();
        Self {
            dimension: dim,
            data: m.iter().copied().collect(),
        }
    }

    /// Compute Christoffel symbols of the second kind from metric derivatives.
    /// Γ^i_{jk} = ½ g^{il} (∂_j g_{lk} + ∂_k g_{lj} - ∂_l g_{jk})
    pub fn christoffel_symbols(
        g_inv: &DMatrix<f64>,
        dg: &[DMatrix<f64>], // ∂g/∂x^μ for each μ
    ) -> DMatrix<f64> {
        let dim = g_inv.nrows();
        let mut gamma = DMatrix::zeros(dim, dim * dim);
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    let mut sum = 0.0;
                    for l in 0..dim {
                        let val = dg[j][(l, k)] + dg[k][(l, j)] - dg[l][(j, k)];
                        sum += g_inv[(i, l)] * val;
                    }
                    gamma[(i, j * dim + k)] = 0.5 * sum;
                }
            }
        }
        gamma
    }
}

/// The full curvature map of the capability manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureMap {
    pub measurements: Vec<CurvatureMeasurement>,
    pub dimension: usize,
}

/// Builder/engine for computing capability curvature.
pub struct CapabilityCurvature {
    points: Vec<CapabilityPoint>,
    dimension: usize,
    smoothing: f64,
}

impl CapabilityCurvature {
    pub fn new(dimension: usize) -> Self {
        Self {
            points: Vec::new(),
            dimension,
            smoothing: 1.0,
        }
    }

    pub fn add_point(&mut self, point: CapabilityPoint) {
        assert_eq!(point.coords.len(), self.dimension);
        self.points.push(point);
    }

    pub fn add_points(&mut self, points: Vec<CapabilityPoint>) {
        for p in &points {
            assert_eq!(p.coords.len(), self.dimension);
        }
        self.points.extend(points);
    }

    /// Fit a local metric tensor from nearby points using kernel-weighted
    /// covariance of the reliability surface.
    fn fit_local_metric(&self, center: &[f64], bandwidth: f64) -> DMatrix<f64> {
        let dim = self.dimension;
        let mut metric = DMatrix::zeros(dim, dim);

        let mut total_weight = 0.0;
        for p in &self.points {
            let dist2: f64 = p.coords.iter().zip(center.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            let w = (-dist2 / (2.0 * bandwidth * bandwidth)).exp() * p.samples as f64;
            if w < 1e-12 { continue; }

            let diff: Vec<f64> = p.coords.iter().zip(center.iter()).map(|(a, b)| a - b).collect();
            for i in 0..dim {
                for j in 0..dim {
                    metric[(i, j)] += w * diff[i] * diff[j];
                }
            }
            total_weight += w;
        }

        if total_weight > 1e-12 {
            metric /= total_weight;
        }
        // Regularize
        for i in 0..dim {
            metric[(i, i)] += self.smoothing;
        }
        metric
    }

    /// Compute the Ricci scalar curvature at a point via discrete
    /// approximation of the Riemann tensor.
    fn ricci_scalar_at(&self, center: &[f64], epsilon: f64, bandwidth: f64) -> CurvatureMeasurement {
        let dim = self.dimension;
        let g0 = self.fit_local_metric(center, bandwidth);

        // Metric at perturbed points along each axis
        let mut dg: Vec<DMatrix<f64>> = Vec::with_capacity(dim);
        let mut ddg: Vec<DMatrix<f64>> = Vec::with_capacity(dim * dim);

        for mu in 0..dim {
            let mut fwd = center.to_vec();
            let mut bwd = center.to_vec();
            fwd[mu] += epsilon;
            bwd[mu] -= epsilon;
            let g_fwd = self.fit_local_metric(&fwd, bandwidth);
            let g_bwd = self.fit_local_metric(&bwd, bandwidth);
            dg.push((g_fwd - &g_bwd) / (2.0 * epsilon));
        }

        for mu in 0..dim {
            for nu in 0..dim {
                let mut fwd = center.to_vec();
                let mut bwd = center.to_vec();
                fwd[mu] += epsilon;
                bwd[mu] -= epsilon;
                let dg_fwd = {
                    let mut p = center.to_vec();
                    p[mu] += epsilon;
                    let mut p2 = center.to_vec();
                    p2[mu] += epsilon;
                    p2[nu] += epsilon;
                    let mut p3 = center.to_vec();
                    p3[mu] += epsilon;
                    p3[nu] -= epsilon;
                    let g_p = self.fit_local_metric(&p2, bandwidth);
                    let g_m = self.fit_local_metric(&p3, bandwidth);
                    (g_p - g_m) / (2.0 * epsilon)
                };
                let dg_bwd = {
                    let mut p2 = center.to_vec();
                    p2[mu] -= epsilon;
                    p2[nu] += epsilon;
                    let mut p3 = center.to_vec();
                    p3[mu] -= epsilon;
                    p3[nu] -= epsilon;
                    let g_p = self.fit_local_metric(&p2, bandwidth);
                    let g_m = self.fit_local_metric(&p3, bandwidth);
                    (g_p - g_m) / (2.0 * epsilon)
                };
                ddg.push((dg_fwd - dg_bwd) / (2.0 * epsilon));
            }
        }

        // Inverse metric
        let g_inv = g0.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(dim, dim));

        // Christoffel symbols
        let gamma = MetricTensor::christoffel_symbols(&g_inv, &dg);

        // Ricci scalar: R = g^{ij} R_{ij} where
        // R_{ij} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik} + Γ^k_{ij} Γ^l_{kl} - Γ^l_{ik} Γ^k_{lj}
        // Simplified discrete approximation:
        let mut ricci_scalar = 0.0;
        for i in 0..dim {
            for j in 0..dim {
                // ∂_k Γ^k_{ij} ≈ (Γ^k_{ij}|_{x+εe_k} - Γ^k_{ij}|_{x-εe_k}) / 2ε
                let mut d_gamma_diag = 0.0;
                for k in 0..dim {
                    d_gamma_diag += ddg[k * dim + j].trace() / dim as f64;
                }
                // Approximate Ricci tensor component
                let r_ij = d_gamma_diag;
                ricci_scalar += g_inv[(i, j)] * r_ij;
            }
        }

        // Sectional curvatures along principal 2-planes
        let mut sectional = Vec::new();
        for i in 0..dim {
            for j in (i + 1)..dim {
                // K(e_i, e_j) = R_{ijij} / (g_{ii} g_{jj} - g_{ij}^2)
                let denom = g0[(i, i)] * g0[(j, j)] - g0[(i, j)].powi(2);
                let k_ij = if denom.abs() > 1e-12 {
                    ricci_scalar / (dim as f64 * (dim as f64 - 1.0))
                } else {
                    0.0
                };
                sectional.push(k_ij);
            }
        }

        // Confidence based on point density
        let nearby: f64 = self.points.iter()
            .map(|p| {
                let d2: f64 = p.coords.iter().zip(center.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                (-d2 / (bandwidth * bandwidth)).exp()
            })
            .sum();

        CurvatureMeasurement {
            coords: center.to_vec(),
            ricci_scalar,
            sectional,
            confidence: (1.0 - (-nearby / 5.0).exp()).max(0.0).min(1.0),
        }
    }

    /// Build the full curvature map across the manifold.
    pub fn compute_map(&self, grid_points: usize, epsilon: f64, bandwidth: f64) -> CurvatureMap {
        let mut measurements = Vec::new();

        if self.points.is_empty() {
            return CurvatureMap {
                measurements,
                dimension: self.dimension,
            };
        }

        // Use actual data points as measurement centers
        for p in &self.points {
            let m = self.ricci_scalar_at(&p.coords, epsilon, bandwidth);
            measurements.push(m);
        }

        CurvatureMap {
            measurements,
            dimension: self.dimension,
        }
    }

    /// Classify a region as flat (reliable) or curved (failure-prone).
    pub fn classify_region(curvature: &CurvatureMeasurement) -> RegionClass {
        let r = curvature.ricci_scalar;
        if r.abs() < 0.1 {
            RegionClass::Flat
        } else if r > 0.1 {
            RegionClass::Converging
        } else {
            RegionClass::Diverging
        }
    }

    /// Find the most failure-prone regions (highest negative curvature).
    pub fn most_failure_prone(map: &CurvatureMap, top_k: usize) -> Vec<&CurvatureMeasurement> {
        let mut indexed: Vec<_> = map.measurements.iter().enumerate().collect();
        indexed.sort_by(|a, b| a.1.ricci_scalar.partial_cmp(&b.1.ricci_scalar).unwrap());
        indexed.into_iter().take(top_k).map(|(_, m)| m).collect()
    }

    /// Find the most reliable regions (highest positive curvature or flattest).
    pub fn most_reliable(map: &CurvatureMap, top_k: usize) -> Vec<&CurvatureMeasurement> {
        let mut indexed: Vec<_> = map.measurements.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.ricci_scalar.partial_cmp(&a.1.ricci_scalar).unwrap());
        indexed.into_iter().take(top_k).map(|(_, m)| m).collect()
    }

    pub fn set_smoothing(&mut self, s: f64) {
        self.smoothing = s;
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionClass {
    /// Near-zero curvature: predictable, reliable.
    Flat,
    /// Positive curvature: converging — reliable with diminishing returns.
    Converging,
    /// Negative curvature: diverging — failure-prone.
    Diverging,
}
