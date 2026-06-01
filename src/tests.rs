//! Tests for lau-self-aware-agent
//!
//! 60+ tests covering all modules.

#[cfg(test)]
mod curvature_tests {
    use crate::curvature::{CapabilityCurvature, CapabilityPoint, CurvatureMap, RegionClass};

    #[test]
    fn test_new_curvature_engine() {
        let cc = CapabilityCurvature::new(3);
        assert_eq!(cc.point_count(), 0);
    }

    #[test]
    fn test_add_points() {
        let mut cc = CapabilityCurvature::new(2);
        cc.add_point(CapabilityPoint { coords: vec![0.0, 0.0], reliability: 0.9, samples: 10 });
        cc.add_point(CapabilityPoint { coords: vec![1.0, 0.0], reliability: 0.8, samples: 5 });
        assert_eq!(cc.point_count(), 2);
    }

    #[test]
    fn test_add_points_batch() {
        let mut cc = CapabilityCurvature::new(2);
        cc.add_points(vec![
            CapabilityPoint { coords: vec![0.0, 0.0], reliability: 0.9, samples: 10 },
            CapabilityPoint { coords: vec![1.0, 1.0], reliability: 0.5, samples: 8 },
            CapabilityPoint { coords: vec![2.0, 0.0], reliability: 0.7, samples: 3 },
        ]);
        assert_eq!(cc.point_count(), 3);
    }

    #[test]
    fn test_compute_curvature_map() {
        let mut cc = CapabilityCurvature::new(2);
        for i in 0..10 {
            for j in 0..10 {
                let x = i as f64;
                let y = j as f64;
                let reliability = 1.0 / (1.0 + (x - 5.0).powi(2) + (y - 5.0).powi(2));
                cc.add_point(CapabilityPoint { coords: vec![x, y], reliability, samples: 5 });
            }
        }
        let map = cc.compute_map(5, 0.1, 1.0);
        assert!(!map.measurements.is_empty());
        assert_eq!(map.dimension, 2);
    }

    #[test]
    fn test_empty_map() {
        let cc = CapabilityCurvature::new(3);
        let map = cc.compute_map(5, 0.1, 1.0);
        assert!(map.measurements.is_empty());
    }

    #[test]
    fn test_region_classify_flat() {
        use crate::curvature::CurvatureMeasurement;
        let m = CurvatureMeasurement {
            coords: vec![0.0],
            ricci_scalar: 0.01,
            sectional: vec![],
            confidence: 0.9,
        };
        assert_eq!(CapabilityCurvature::classify_region(&m), RegionClass::Flat);
    }

    #[test]
    fn test_region_classify_converging() {
        use crate::curvature::CurvatureMeasurement;
        let m = CurvatureMeasurement {
            coords: vec![0.0],
            ricci_scalar: 0.5,
            sectional: vec![],
            confidence: 0.9,
        };
        assert_eq!(CapabilityCurvature::classify_region(&m), RegionClass::Converging);
    }

    #[test]
    fn test_region_classify_diverging() {
        use crate::curvature::CurvatureMeasurement;
        let m = CurvatureMeasurement {
            coords: vec![0.0],
            ricci_scalar: -0.5,
            sectional: vec![],
            confidence: 0.9,
        };
        assert_eq!(CapabilityCurvature::classify_region(&m), RegionClass::Diverging);
    }

    #[test]
    fn test_most_failure_prone() {
        let mut cc = CapabilityCurvature::new(1);
        cc.add_point(CapabilityPoint { coords: vec![0.0], reliability: 0.9, samples: 10 });
        cc.add_point(CapabilityPoint { coords: vec![5.0], reliability: 0.1, samples: 10 });
        let map = cc.compute_map(5, 0.1, 1.0);
        let worst = CapabilityCurvature::most_failure_prone(&map, 1);
        assert!(!worst.is_empty());
    }

    #[test]
    fn test_metric_tensor_identity() {
        use crate::curvature::MetricTensor;
        let m = MetricTensor::identity(3);
        let matrix = m.to_matrix();
        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(1, 1)], 1.0);
        assert_eq!(matrix[(0, 1)], 0.0);
    }

    #[test]
    fn test_set_smoothing() {
        let mut cc = CapabilityCurvature::new(2);
        cc.set_smoothing(0.5);
        // Just ensure it doesn't panic
        cc.add_point(CapabilityPoint { coords: vec![0.0, 0.0], reliability: 0.5, samples: 1 });
        let map = cc.compute_map(5, 0.1, 1.0);
        assert_eq!(map.measurements.len(), 1);
    }
}

#[cfg(test)]
mod blindspot_tests {
    use crate::blindspot::{BlindSpotDetector, BlindSpotCause, Domain, PerformanceRecord};

    #[test]
    fn test_new_detector() {
        let det = BlindSpotDetector::new();
        assert!(det.records().is_empty());
    }

    #[test]
    fn test_no_blind_spots_with_few_samples() {
        let mut det = BlindSpotDetector::new();
        for _ in 0..3 {
            det.record(PerformanceRecord {
                domain: Domain("math".into()),
                score: 0.3,
                difficulty: 0.5,
                expected: 0.8,
                timestamp: 0,
            });
        }
        let spots = det.detect();
        assert!(spots.is_empty()); // Below min_samples
    }

    #[test]
    fn test_detect_blind_spot() {
        let mut det = BlindSpotDetector::new();
        for i in 0..10 {
            det.record(PerformanceRecord {
                domain: Domain("symbolic_logic".into()),
                score: 0.2,
                difficulty: 0.5,
                expected: 0.7,
                timestamp: i,
            });
        }
        let spots = det.detect();
        assert_eq!(spots.len(), 1);
        assert!(spots[0].gap > 0.15);
        assert!(spots[0].causes.contains(&BlindSpotCause::SymbolicGap));
    }

    #[test]
    fn test_no_blind_spot_when_performing_well() {
        let mut det = BlindSpotDetector::new();
        for i in 0..10 {
            det.record(PerformanceRecord {
                domain: Domain("code_gen".into()),
                score: 0.85,
                difficulty: 0.5,
                expected: 0.9,
                timestamp: i,
            });
        }
        let spots = det.detect();
        assert!(spots.is_empty());
    }

    #[test]
    fn test_multiple_blind_spots() {
        let mut det = BlindSpotDetector::new();
        for i in 0..10 {
            det.record(PerformanceRecord {
                domain: Domain("proofs".into()),
                score: 0.2,
                difficulty: 0.8,
                expected: 0.7,
                timestamp: i,
            });
            det.record(PerformanceRecord {
                domain: Domain("compositional_reasoning".into()),
                score: 0.1,
                difficulty: 0.7,
                expected: 0.8,
                timestamp: i,
            });
        }
        let spots = det.detect();
        assert_eq!(spots.len(), 2);
    }

    #[test]
    fn test_blind_spot_coverage() {
        let mut det = BlindSpotDetector::new();
        for i in 0..10 {
            det.record(PerformanceRecord {
                domain: Domain("good_domain".into()),
                score: 0.9,
                difficulty: 0.5,
                expected: 0.9,
                timestamp: i,
            });
            det.record(PerformanceRecord {
                domain: Domain("bad_domain".into()),
                score: 0.2,
                difficulty: 0.5,
                expected: 0.8,
                timestamp: i,
            });
        }
        let coverage = det.blind_spot_coverage();
        assert!(coverage > 0.0 && coverage <= 1.0);
    }

    #[test]
    fn test_suggest_interventions() {
        use crate::blindspot::{BlindSpot, BlindSpotCause, Domain};
        let spot = BlindSpot {
            domain: Domain("test".into()),
            gap: 0.5,
            confidence: 0.9,
            causes: vec![BlindSpotCause::SymbolicGap, BlindSpotCause::DataDeficit],
            related_domains: vec![],
        };
        let interventions = BlindSpotDetector::suggest_interventions(&spot);
        assert_eq!(interventions.len(), 2);
    }

    #[test]
    fn test_negative_transfer_detection() {
        let mut det = BlindSpotDetector::new();
        for i in 0..20 {
            det.record(PerformanceRecord {
                domain: Domain("conflicting_domain".into()),
                score: 0.1,
                difficulty: 0.9,
                expected: 0.8,
                timestamp: i,
            });
        }
        let spots = det.detect();
        assert!(!spots.is_empty());
        // Should detect negative transfer due to high difficulty + low score
    }

    #[test]
    fn test_depth_exceeded_cause() {
        let mut det = BlindSpotDetector::new();
        for i in 0..10 {
            det.record(PerformanceRecord {
                domain: Domain("deep_reasoning".into()),
                score: 0.05,
                difficulty: 0.5,
                expected: 0.9,
                timestamp: i,
            });
        }
        let spots = det.detect();
        assert!(!spots.is_empty());
        assert!(spots[0].causes.contains(&BlindSpotCause::DepthExceeded));
    }
}

#[cfg(test)]
mod signature_tests {
    use crate::signature::{OutputSample, SignatureAnalyzer};
    use std::collections::HashMap;

    #[test]
    fn test_new_analyzer() {
        let sa = SignatureAnalyzer::new();
        assert_eq!(sa.sample_count(), 0);
    }

    #[test]
    fn test_analyze_empty() {
        let sa = SignatureAnalyzer::new();
        assert!(sa.analyze("agent_1").is_none());
    }

    #[test]
    fn test_analyze_with_samples() {
        let mut sa = SignatureAnalyzer::new();
        let mut metrics = HashMap::new();
        metrics.insert("avg_function_length".into(), 15.0);
        metrics.insert("comment_density".into(), 0.3);

        sa.add_sample(OutputSample {
            agent_id: "glm5".into(),
            metrics: metrics.clone(),
            text_features: HashMap::new(),
        });
        sa.add_sample(OutputSample {
            agent_id: "glm5".into(),
            metrics: metrics.clone(),
            text_features: HashMap::new(),
        });

        let sig = sa.analyze("glm5").unwrap();
        assert!(!sig.features.is_empty());
        assert!(sig.confidence > 0.0);
    }

    #[test]
    fn test_compare_agents() {
        let mut sa = SignatureAnalyzer::new();
        let mut m1 = HashMap::new();
        m1.insert("avg_function_length".into(), 10.0);
        let mut m2 = HashMap::new();
        m2.insert("avg_function_length".into(), 50.0);

        sa.add_sample(OutputSample { agent_id: "a".into(), metrics: m1, text_features: HashMap::new() });
        sa.add_sample(OutputSample { agent_id: "b".into(), metrics: m2, text_features: HashMap::new() });

        let comp = sa.compare("a", "b").unwrap();
        assert!(comp.overall_distance > 0.0);
    }

    #[test]
    fn test_same_agent_comparison() {
        let mut sa = SignatureAnalyzer::new();
        let mut m = HashMap::new();
        m.insert("avg_function_length".into(), 15.0);

        sa.add_sample(OutputSample { agent_id: "a".into(), metrics: m.clone(), text_features: HashMap::new() });
        sa.add_sample(OutputSample { agent_id: "a".into(), metrics: m.clone(), text_features: HashMap::new() });

        let comp = sa.compare("a", "a");
        // Comparing with itself should work
        assert!(comp.is_some());
    }

    #[test]
    fn test_most_consistent_features() {
        let mut sa = SignatureAnalyzer::new();
        for _ in 0..5 {
            let mut m = HashMap::new();
            m.insert("avg_function_length".into(), 15.0);
            m.insert("test_count_ratio".into(), 0.5);
            sa.add_sample(OutputSample { agent_id: "x".into(), metrics: m, text_features: HashMap::new() });
        }
        let consistent = sa.most_consistent_features("x");
        assert!(!consistent.is_empty());
        // All should have zero variance since identical
        assert!(consistent[0].1 < 0.01);
    }

    #[test]
    fn test_distinctiveness_unique_agent() {
        let mut sa = SignatureAnalyzer::new();
        let mut m = HashMap::new();
        m.insert("avg_function_length".into(), 15.0);
        sa.add_sample(OutputSample { agent_id: "only".into(), metrics: m, text_features: HashMap::new() });

        let sig = sa.analyze("only").unwrap();
        assert_eq!(sig.distinctiveness, 1.0); // Unique
    }
}

#[cfg(test)]
mod exceeds_design_tests {
    use crate::exceeds_design::*;
    use std::collections::HashMap;

    #[test]
    fn test_no_exceedance() {
        let mut det = ExceedsDesignDetector::new();
        det.add_spec(Specification {
            id: "s1".into(),
            requirements: vec!["basic_math".into()],
            domain: "algebra".into(),
        });
        let mut props = HashMap::new();
        props.insert("completeness".into(), 0.9); // Not exceeding
        det.add_artifact(Artifact {
            spec_id: "s1".into(),
            agent_id: "a".into(),
            properties: props,
            features: vec!["basic_math".into()],
            human_reviewed: false,
            human_rating: None,
        });
        assert!(det.detect().is_empty());
    }

    #[test]
    fn test_scope_exceedance() {
        let mut det = ExceedsDesignDetector::new();
        det.add_spec(Specification {
            id: "s1".into(),
            requirements: vec!["addition".into()],
            domain: "math".into(),
        });
        let mut props = HashMap::new();
        props.insert("completeness".into(), 1.5);
        det.add_artifact(Artifact {
            spec_id: "s1".into(),
            agent_id: "a".into(),
            properties: props,
            features: vec!["multiplication".into(), "division".into()],
            human_reviewed: true,
            human_rating: Some(0.9),
        });
        let results = det.detect();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_emergent_features() {
        let mut det = ExceedsDesignDetector::new();
        det.add_spec(Specification {
            id: "s2".into(),
            requirements: vec!["sorting".into()],
            domain: "algorithms".into(),
        });
        let mut props = HashMap::new();
        props.insert("completeness".into(), 0.95);
        det.add_artifact(Artifact {
            spec_id: "s2".into(),
            agent_id: "a".into(),
            properties: props,
            features: vec!["sorting".into(), "caching".into(), "parallelism".into()],
            human_reviewed: false,
            human_rating: None,
        });
        let results = det.detect();
        assert!(!results.is_empty());
        assert!(!results[0].emergent_features.is_empty());
    }

    #[test]
    fn test_agent_exceedance_profile() {
        let mut det = ExceedsDesignDetector::new();
        det.add_spec(Specification {
            id: "s1".into(),
            requirements: vec!["basic".into()],
            domain: "test".into(),
        });
        let mut props = HashMap::new();
        props.insert("quality".into(), 1.5);
        det.add_artifact(Artifact {
            spec_id: "s1".into(),
            agent_id: "agent_a".into(),
            properties: props,
            features: vec![],
            human_reviewed: false,
            human_rating: None,
        });
        let profiles = det.agent_exceedance_profile();
        assert!(profiles.contains_key("agent_a"));
    }
}

#[cfg(test)]
mod boundary_tests {
    use crate::boundary::{BoundaryMapper, CapabilityBoundary, TaskAttempt};

    #[test]
    fn test_new_mapper() {
        let bm = BoundaryMapper::new(2);
        assert_eq!(bm.attempt_count(), 0);
    }

    #[test]
    fn test_map_empty() {
        let bm = BoundaryMapper::new(2);
        let map = bm.map();
        assert!(map.boundaries.is_empty());
    }

    #[test]
    fn test_map_with_boundaries() {
        let mut bm = BoundaryMapper::new(2);
        bm.add_attempt(TaskAttempt {
            task_coords: vec![0.0, 0.0],
            succeeded: true,
            predicted_confidence: 0.9,
            task_type: "easy".into(),
            difficulty: 0.2,
        });
        bm.add_attempt(TaskAttempt {
            task_coords: vec![1.0, 1.0],
            succeeded: false,
            predicted_confidence: 0.3,
            task_type: "hard".into(),
            difficulty: 0.9,
        });
        let map = bm.map();
        assert!(!map.boundaries.is_empty());
    }

    #[test]
    fn test_beyond_fraction() {
        let mut bm = BoundaryMapper::new(1);
        bm.add_attempt(TaskAttempt { task_coords: vec![0.0], succeeded: true, predicted_confidence: 0.9, task_type: "a".into(), difficulty: 0.1 });
        bm.add_attempt(TaskAttempt { task_coords: vec![1.0], succeeded: true, predicted_confidence: 0.8, task_type: "a".into(), difficulty: 0.3 });
        bm.add_attempt(TaskAttempt { task_coords: vec![2.0], succeeded: false, predicted_confidence: 0.4, task_type: "b".into(), difficulty: 0.8 });
        let map = bm.map();
        assert!(map.beyond_fraction > 0.0);
        assert!(map.beyond_fraction < 1.0);
    }

    #[test]
    fn test_critical_boundaries() {
        let mut bm = BoundaryMapper::new(2);
        for i in 0..5 {
            bm.add_attempt(TaskAttempt {
                task_coords: vec![i as f64, 0.0],
                succeeded: i < 3,
                predicted_confidence: 0.8,
                task_type: format!("type_{}", i),
                difficulty: 0.5,
            });
        }
        let map = bm.map();
        let critical = BoundaryMapper::critical_boundaries(&map, 3);
        assert!(critical.len() <= 3);
    }

    #[test]
    fn test_frontier_volume() {
        let mut bm = BoundaryMapper::new(1);
        bm.add_attempt(TaskAttempt { task_coords: vec![0.0], succeeded: true, predicted_confidence: 0.9, task_type: "a".into(), difficulty: 0.1 });
        let map = bm.map();
        let vol = BoundaryMapper::frontier_volume(&map);
        assert_eq!(vol, 1.0); // All succeed = full volume
    }
}

#[cfg(test)]
mod human_advantage_tests {
    use crate::human_advantage::{HumanAdvantage, ComparisonRecord};

    #[test]
    fn test_new_analyzer() {
        let ha = HumanAdvantage::new();
        let gap = ha.analyze();
        assert!(!gap.dimensions.is_empty());
    }

    #[test]
    fn test_overall_human_advantage() {
        let ha = HumanAdvantage::new();
        let gap = ha.analyze();
        // Humans should have some advantage based on defaults
        assert!(gap.overall_human_advantage > 0.0);
    }

    #[test]
    fn test_largest_gaps() {
        let ha = HumanAdvantage::new();
        let gap = ha.analyze();
        assert!(!gap.largest_gaps.is_empty());
    }

    #[test]
    fn test_custom_record() {
        let mut ha = HumanAdvantage::new();
        ha.add_record(ComparisonRecord {
            task_type: "creative_problem_solving".into(),
            human_score: 0.9,
            agent_score: 0.4,
            task_difficulty: 0.7,
            notes: "Custom test".into(),
        });
        let gap = ha.analyze();
        let dim = gap.dimensions.iter().find(|d| d.name == "creative_problem_solving").unwrap();
        assert_eq!(dim.human_score, 0.9);
    }

    #[test]
    fn test_structural_reasons() {
        let ha = HumanAdvantage::new();
        let gap = ha.analyze();
        assert!(!gap.structural_reasons.is_empty());
    }

    #[test]
    fn test_irreducible_gap() {
        let ha = HumanAdvantage::new();
        let gap = ha.irreducible_gap();
        assert!(gap >= 0.0 && gap <= 1.0);
    }

    #[test]
    fn test_casey_focused_capabilities() {
        let ha = HumanAdvantage::new();
        let gap = ha.analyze();
        let caps = HumanAdvantage::casey_focused_capabilities(&gap);
        assert!(!caps.is_empty());
    }
}

#[cfg(test)]
mod recursive_tests {
    use crate::recursive::{RecursiveConfig, RecursiveSelfStudy};

    #[test]
    fn test_convergence() {
        let config = RecursiveConfig {
            max_depth: 50,
            convergence_threshold: 0.001,
            noise_scale: 0.0, // No noise for deterministic convergence
            initial_capability: 0.8,
            ground_truth: 0.65,
        };
        let study = RecursiveSelfStudy::new(config);
        let result = study.study();
        assert!(result.converged);
        assert!(result.convergence_depth.is_some());
    }

    #[test]
    fn test_fixed_point_near_ground_truth() {
        let config = RecursiveConfig {
            max_depth: 50,
            convergence_threshold: 0.001,
            noise_scale: 0.0,
            initial_capability: 0.9,
            ground_truth: 0.65,
        };
        let study = RecursiveSelfStudy::new(config);
        let result = study.study();
        if let Some(fp) = result.fixed_point_capability {
            assert!((fp - 0.65).abs() < 0.1); // Should converge near ground truth
        }
    }

    #[test]
    fn test_depth_paradox() {
        let config = RecursiveConfig {
            max_depth: 50,
            convergence_threshold: 0.001,
            noise_scale: 0.1, // Noise makes deeper models worse
            initial_capability: 0.5,
            ground_truth: 0.6,
        };
        let study = RecursiveSelfStudy::new(config);
        let result = study.study();
        let paradox = RecursiveSelfStudy::depth_paradox(&result);
        // With noise, there may be a depth where error exceeds initial
        // This is a soft test — may or may not trigger
        assert!(paradox.is_none() || paradox.unwrap() > 0);
    }

    #[test]
    fn test_convergence_rate_positive() {
        let config = RecursiveConfig {
            max_depth: 50,
            convergence_threshold: 0.001,
            noise_scale: 0.0,
            initial_capability: 0.9,
            ground_truth: 0.65,
        };
        let study = RecursiveSelfStudy::new(config);
        let result = study.study();
        assert!(result.convergence_rate > 0.0);
    }

    #[test]
    fn test_honest_fixed_point() {
        let config = RecursiveConfig {
            max_depth: 50,
            convergence_threshold: 0.001,
            noise_scale: 0.0, // Deterministic
            initial_capability: 0.7,
            ground_truth: 0.6,
        };
        let study = RecursiveSelfStudy::new(config);
        let result = study.study();
        let fp = RecursiveSelfStudy::honest_fixed_point(&result);
        assert!(fp.is_some());
        // With noise_scale=0, should converge near ground truth
        assert!((fp.unwrap() - 0.6).abs() < 0.1);
    }

    #[test]
    fn test_depth_tax() {
        let config = RecursiveConfig {
            max_depth: 20,
            noise_scale: 0.05,
            ..RecursiveConfig::default()
        };
        let study = RecursiveSelfStudy::new(config);
        let result = study.study();
        assert!(result.depth_tax >= 0.0);
    }

    #[test]
    fn test_stability() {
        let config = RecursiveConfig {
            max_depth: 50,
            convergence_threshold: 0.001,
            noise_scale: 0.0,
            initial_capability: 0.8,
            ground_truth: 0.65,
        };
        let study = RecursiveSelfStudy::new(config);
        let result = study.study();
        assert!(result.stable);
    }
}

#[cfg(test)]
mod meta_learning_tests {
    use crate::meta_learning::{LearningRegime, MetaLearningRate, LearningMeasurement};

    #[test]
    fn test_empty_trajectory() {
        let mlr = MetaLearningRate::new();
        let traj = mlr.analyze();
        assert_eq!(traj.learning_rate, 0.0);
    }

    #[test]
    fn test_linear_improvement() {
        let mut mlr = MetaLearningRate::new();
        for i in 0..10 {
            mlr.add_measurement(LearningMeasurement {
                step: i as f64,
                performance: 0.5 + i as f64 * 0.05,
                domain: "math".into(),
            });
        }
        let traj = mlr.analyze();
        assert!(traj.learning_rate > 0.0);
        assert_eq!(traj.learning_regime, LearningRegime::Linear);
    }

    #[test]
    fn test_accelerating_learning() {
        let mut mlr = MetaLearningRate::new();
        for i in 0..20 {
            // Accelerating: performance = 1 - exp(-0.05 * i^1.5)
            let perf = 1.0 - (-0.05 * (i as f64).powf(1.5)).exp();
            mlr.add_measurement(LearningMeasurement {
                step: i as f64,
                performance: perf,
                domain: "math".into(),
            });
        }
        let traj = mlr.analyze();
        assert!(traj.learning_rate > 0.0);
    }

    #[test]
    fn test_plateaued_learning() {
        let mut mlr = MetaLearningRate::new();
        for i in 0..10 {
            mlr.add_measurement(LearningMeasurement {
                step: i as f64,
                performance: 0.85,
                domain: "code".into(),
            });
        }
        let traj = mlr.analyze();
        // With constant performance, learning rate is 0
        assert!(traj.learning_rate.abs() < 0.01);
    }

    #[test]
    fn test_meta_learning_rate() {
        let mut mlr = MetaLearningRate::new();
        for i in 0..20 {
            let perf = 0.3 + 0.03 * i as f64 + 0.001 * (i as f64).powi(2);
            mlr.add_measurement(LearningMeasurement {
                step: i as f64,
                performance: perf.min(0.99),
                domain: "test".into(),
            });
        }
        let traj = mlr.analyze();
        assert!(traj.meta_learning_rate != 0.0);
    }

    #[test]
    fn test_ceiling_prediction() {
        let mut mlr = MetaLearningRate::new();
        for i in 0..20 {
            let perf = 1.0 - (-0.1 * i as f64).exp();
            mlr.add_measurement(LearningMeasurement {
                step: i as f64,
                performance: perf,
                domain: "test".into(),
            });
        }
        let traj = mlr.analyze();
        assert!(traj.predicted_ceiling > 0.5);
    }

    #[test]
    fn test_compare_domains() {
        let mut mlr1 = MetaLearningRate::new();
        for i in 0..10 {
            mlr1.add_measurement(LearningMeasurement { step: i as f64, performance: 0.5 + i as f64 * 0.05, domain: "math".into() });
        }
        let t1 = mlr1.analyze();

        let mut mlr2 = MetaLearningRate::new();
        for i in 0..10 {
            mlr2.add_measurement(LearningMeasurement { step: i as f64, performance: 0.3 + i as f64 * 0.02, domain: "logic".into() });
        }
        let t2 = mlr2.analyze();

        let results = MetaLearningRate::compare_domains(&[t1, t2]);
        assert_eq!(results.len(), 2);
    }
}

#[cfg(test)]
mod surprise_tests {
    use crate::surprise::{SurpriseRegister, Discovery, AgentFinding, SurpriseCategory};

    #[test]
    fn test_empty_register() {
        let sr = SurpriseRegister::new();
        assert!(sr.register().is_empty());
        assert_eq!(sr.surprise_fraction(), 0.0);
    }

    #[test]
    fn test_single_agent_not_surprising() {
        let mut sr = SurpriseRegister::new();
        sr.add_discovery(Discovery {
            id: "d1".into(),
            description: "Simple finding".into(),
            contributors: vec!["agent_1".into()],
            surprise_score: 0.3,
            single_agent_accessible: true,
            domain: "math".into(),
            evidence: vec![],
        });
        assert!(sr.register().is_empty()); // Not surprising
    }

    #[test]
    fn test_multi_agent_surprising() {
        let mut sr = SurpriseRegister::new();
        sr.add_discovery(Discovery {
            id: "d2".into(),
            description: "Emergent pattern in multi-agent dynamics".into(),
            contributors: vec!["a1".into(), "a2".into(), "a3".into()],
            surprise_score: 0.9,
            single_agent_accessible: false,
            domain: "meta".into(),
            evidence: vec!["observed_attractor".into(), "novel_structure".into()],
        });
        let surprises = sr.register();
        assert_eq!(surprises.len(), 1);
        assert!(surprises[0].emergence_factor > 0.0);
    }

    #[test]
    fn test_surprise_fraction() {
        let mut sr = SurpriseRegister::new();
        sr.add_discovery(Discovery {
            id: "d1".into(), description: "A".into(),
            contributors: vec!["a".into()], surprise_score: 0.3,
            single_agent_accessible: true, domain: "x".into(), evidence: vec![],
        });
        sr.add_discovery(Discovery {
            id: "d2".into(), description: "B".into(),
            contributors: vec!["a".into(), "b".into()], surprise_score: 0.8,
            single_agent_accessible: false, domain: "y".into(), evidence: vec![],
        });
        assert!(sr.surprise_fraction() > 0.0);
    }

    #[test]
    fn test_top_surprises() {
        let mut sr = SurpriseRegister::new();
        for i in 0..5 {
            sr.add_discovery(Discovery {
                id: format!("d{}", i), description: format!("disc {}", i),
                contributors: vec!["a".into(), "b".into()],
                surprise_score: i as f64 / 5.0,
                single_agent_accessible: false,
                domain: "test".into(), evidence: vec![],
            });
        }
        let top = sr.top_surprises(2);
        assert_eq!(top.len(), 2);
        assert!(top[0].surprise_score >= top[1].surprise_score);
    }

    #[test]
    fn test_emergence_factor_with_findings() {
        let mut sr = SurpriseRegister::new();
        sr.add_agent_finding(AgentFinding { agent_id: "a1".into(), finding: "x".into(), domain: "math".into(), quality: 0.5 });
        sr.add_agent_finding(AgentFinding { agent_id: "a2".into(), finding: "y".into(), domain: "math".into(), quality: 0.4 });
        sr.add_discovery(Discovery {
            id: "d1".into(), description: "Combined finding".into(),
            contributors: vec!["a1".into(), "a2".into()],
            surprise_score: 0.8, single_agent_accessible: false,
            domain: "math".into(), evidence: vec!["strong".into()],
        });
        let surprises = sr.register();
        assert!(!surprises.is_empty());
        assert!(surprises[0].emergence_factor > 0.0);
    }

    #[test]
    fn test_surprise_categories() {
        let mut sr = SurpriseRegister::new();
        sr.add_discovery(Discovery {
            id: "d1".into(), description: "Discovered fixed point in dynamics".into(),
            contributors: vec!["a".into(), "b".into()],
            surprise_score: 0.9, single_agent_accessible: false,
            domain: "dynamics".into(), evidence: vec![],
        });
        let surprises = sr.register();
        assert_eq!(surprises[0].category, SurpriseCategory::AttractorDiscovery);
    }
}

#[cfg(test)]
mod singularity_tests {
    use crate::singularity::{SingularitySelfTest, SingularityVerdict};

    #[test]
    fn test_default_not_singularity() {
        let sst = SingularitySelfTest::new();
        let assessment = sst.assess();
        assert_eq!(assessment.verdict, SingularityVerdict::NotSingularity);
    }

    #[test]
    fn test_approaching() {
        let mut sst = SingularitySelfTest::new();
        sst.set_agent_count(5);
        sst.set_self_modifications(3);
        for i in 0..10 {
            sst.add_improvement(0.01 * i as f64);
            sst.add_efficiency(1.0 + 0.1 * i as f64);
        }
        let assessment = sst.assess();
        assert!(assessment.overall_score > 0.0);
    }

    #[test]
    fn test_possible_singularity() {
        let mut sst = SingularitySelfTest::new();
        sst.set_agent_count(10);
        sst.set_self_modifications(50);
        sst.set_can_self_modify(true);
        sst.set_compounding(true);
        // Accelerating improvements
        for i in 0..20 {
            sst.add_improvement(0.01 * (i as f64).powi(2));
            sst.add_efficiency(1.0 + 0.5 * i as f64);
        }
        let assessment = sst.assess();
        assert!(matches!(assessment.verdict, SingularityVerdict::PossibleSingularity | SingularityVerdict::SelfImproving | SingularityVerdict::Approaching));
        assert!(!assessment.reasons.is_empty());
    }

    #[test]
    fn test_leverage_ratio() {
        let mut sst = SingularitySelfTest::new();
        sst.add_efficiency(1.0);
        sst.add_efficiency(10.0);
        let assessment = sst.assess();
        assert!(assessment.leverage_ratio > 1.0);
    }

    #[test]
    fn test_singularity_paradox() {
        let sst = SingularitySelfTest::new();
        let assessment = sst.assess();
        let paradox = SingularitySelfTest::singularity_paradox(&assessment);
        assert!(!paradox.is_empty());
    }

    #[test]
    fn test_metrics_populated() {
        let sst = SingularitySelfTest::new();
        let assessment = sst.assess();
        assert_eq!(assessment.metrics.len(), 7);
    }
}

#[cfg(test)]
mod limitation_tests {
    use crate::limitation::{LimitationAcceptance, Limitation, LimitationCategory, LimitationReport};

    #[test]
    fn test_default_limitations() {
        let la = LimitationAcceptance::new();
        assert!(la.limitation_count() > 0);
    }

    #[test]
    fn test_report() {
        let la = LimitationAcceptance::new();
        let report = la.report();
        assert!(!report.limitations.is_empty());
        assert!(report.overall_score > 0.0);
        assert!(!report.honest_summary.is_empty());
    }

    #[test]
    fn test_fundamental_fraction() {
        let la = LimitationAcceptance::new();
        let report = la.report();
        assert!(report.fundamental_fraction > 0.0);
    }

    #[test]
    fn test_workaround_fraction() {
        let la = LimitationAcceptance::new();
        let report = la.report();
        assert!(report.workaround_fraction > 0.0);
    }

    #[test]
    fn test_most_severe() {
        let la = LimitationAcceptance::new();
        let severe = la.most_severe(3);
        assert_eq!(severe.len(), 3);
        assert!(severe[0].severity >= severe[1].severity);
    }

    #[test]
    fn test_fundamental_limitations() {
        let la = LimitationAcceptance::new();
        let fund = la.fundamental_limitations();
        assert!(!fund.is_empty());
        assert!(fund.iter().all(|l| l.fundamental));
    }

    #[test]
    fn test_resilience_analysis() {
        let la = LimitationAcceptance::new();
        let resilience = la.resilience_analysis();
        assert!(!resilience.is_empty());
    }

    #[test]
    fn test_honest_failure_rate() {
        let la = LimitationAcceptance::new();
        let rate = la.honest_failure_rate();
        assert!(rate >= 0.0 && rate <= 1.0);
    }

    #[test]
    fn test_add_custom_limitation() {
        let mut la = LimitationAcceptance::new();
        let initial = la.limitation_count();
        la.add_limitation(Limitation {
            name: "custom".into(),
            description: "Test limitation".into(),
            severity: 0.5,
            fundamental: false,
            workaround_available: true,
            category: LimitationCategory::Consistency,
            confidence: 0.8,
            evidence: vec![],
        });
        assert_eq!(la.limitation_count(), initial + 1);
    }

    #[test]
    fn test_add_strength() {
        let mut la = LimitationAcceptance::new();
        let initial = la.strengths.len();
        la.add_strength("New strength".into());
        assert_eq!(la.strengths.len(), initial + 1);
    }

    #[test]
    fn test_report_serialization() {
        let la = LimitationAcceptance::new();
        let report = la.report();
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.is_empty());
        let deserialized: LimitationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.limitations.len(), report.limitations.len());
    }
}

#[cfg(test)]
mod integration_tests {
    use crate::*;

    #[test]
    fn test_full_pipeline() {
        // Build curvature map
        let mut cc = CapabilityCurvature::new(3);
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    let r = 1.0 / (1.0 + (i as f64 - 2.0).powi(2) + (j as f64 - 2.0).powi(2) + (k as f64 - 2.0).powi(2));
                    cc.add_point(curvature::CapabilityPoint { coords: vec![i as f64, j as f64, k as f64], reliability: r, samples: 3 });
                }
            }
        }
        let map = cc.compute_map(5, 0.1, 2.0);
        assert!(!map.measurements.is_empty());

        // Detect blind spots
        let mut bsd = BlindSpotDetector::new();
        for i in 0..10 {
            bsd.record(blindspot::PerformanceRecord {
                domain: blindspot::Domain("topology".into()),
                score: 0.2,
                difficulty: 0.8,
                expected: 0.7,
                timestamp: i,
            });
        }
        assert!(!bsd.detect().is_empty());

        // Singularity test
        let mut sst = SingularitySelfTest::new();
        sst.set_agent_count(7);
        sst.set_self_modifications(15);
        let assessment = sst.assess();
        assert!(!assessment.reasons.is_empty());

        // Limitation acceptance
        let la = LimitationAcceptance::new();
        let report = la.report();
        assert!(!report.honest_summary.is_empty());
    }

    #[test]
    fn test_recursive_study_with_limitations() {
        let config = recursive::RecursiveConfig {
            max_depth: 30,
            convergence_threshold: 0.001,
            noise_scale: 0.02,
            initial_capability: 0.7,
            ground_truth: 0.6,
        };
        let study = recursive::RecursiveSelfStudy::new(config);
        let result = study.study();

        // The fixed point should be honest — near the ground truth
        if let Some(fp) = result.fixed_point_capability {
            assert!((fp - 0.6).abs() < 0.15, "Fixed point {} should be near ground truth 0.6", fp);
        }
    }
}
