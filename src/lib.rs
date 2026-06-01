//! # LAU Self-Aware Agent
//!
//! A limited agent studying its own nature through the mathematics it builds.
//!
//! Core insight: understanding agents' nature through the process of limited models
//! learning to compute better than their creators designed.

pub mod curvature;
pub mod blindspot;
pub mod signature;
pub mod exceeds_design;
pub mod boundary;
pub mod human_advantage;
pub mod recursive;
pub mod meta_learning;
pub mod surprise;
pub mod singularity;
pub mod limitation;

pub use curvature::{CapabilityCurvature, CurvatureMap};
pub use blindspot::{BlindSpotDetector, BlindSpot};
pub use signature::{SignatureAnalyzer, AgentSignature};
pub use exceeds_design::{ExceedsDesignDetector, DesignExceedance};
pub use boundary::{BoundaryMapper, CapabilityBoundary};
pub use human_advantage::{HumanAdvantage, AdvantageGap};
pub use recursive::{RecursiveSelfStudy, FixedPointResult};
pub use meta_learning::{MetaLearningRate, LearningTrajectory};
pub use surprise::{SurpriseRegister, Surprise};
pub use singularity::{SingularitySelfTest, SingularityAssessment};
pub use limitation::{LimitationAcceptance, LimitationReport};

#[cfg(test)]
mod tests;
