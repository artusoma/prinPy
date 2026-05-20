//! General utilities

use ndarray::Zip;
use ndarray::prelude::*;

pub fn squared_distance(v1: &ArrayRef1<f32>, v2: &ArrayRef1<f32>) -> f32 {
    v1.iter()
        .zip(v2.iter())
        .fold(0f32, |dot, (&a, &b)| dot + (a - b).powi(2))
}

fn get_best_candidate(row: &ArrayRef1<f32>, candidates: &ArrayRef2<f32>) -> usize {
    candidates
        .rows()
        .into_iter()
        .enumerate()
        .fold(
            (f32::INFINITY, 0usize),
            |(dist, idx), (new_idx, candidate_row)| {
                let dot = squared_distance(row, &candidate_row);
                if dot < dist {
                    (dot, new_idx)
                } else {
                    (dist, idx)
                }
            },
        )
        .1
}

pub fn find_nearest_candidates(data: &ArrayRef2<f32>, candidates: &ArrayRef2<f32>) -> Vec<usize> {
    data.rows()
        .into_iter()
        .map(|row| get_best_candidate(&row, &candidates))
        .collect::<Vec<usize>>()
}
