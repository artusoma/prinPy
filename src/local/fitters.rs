use super::constrained::{ConstrainedFitError, Fitter};
use crate::utilities::compute_pc1;
use ndarray::{Array1, ArrayRef1, ArrayRef2, Axis};

/// Greedy fitter uses a narrow width around the edge of the circle to compute the
/// mean, using that as the next vertex
#[derive(Debug)]
pub struct GreedyFitter {
    slice_width: f32,
}

impl Default for GreedyFitter {
    fn default() -> Self {
        GreedyFitter { slice_width: 0.9 }
    }
}

impl GreedyFitter {
    pub fn new(slice_width: f32) -> Self {
        Self { slice_width }
    }
}

impl Fitter for GreedyFitter {
    fn fit_segment(
        &self,
        data: &ArrayRef2<f32>,
        dist: &ArrayRef1<f32>,
        radius: f32,
        _vertex: &ArrayRef1<f32>,
    ) -> Result<Array1<f32>, ConstrainedFitError> {
        let inner_radius = (self.slice_width * radius.sqrt()).powi(2);
        let indices: Vec<usize> = (0..dist.len())
            .filter(|&idx| dist[idx] > inner_radius)
            .collect();

        if indices.is_empty() {
            // FALLBACK: If no points are in the outer shell, just take the mean of all points
            // in the circle
            return data
                .mean_axis(Axis(0))
                .ok_or(ConstrainedFitError::EmptySliceError);
        }

        let in_manifold = data.select(Axis(0), &indices);
        in_manifold
            .mean_axis(Axis(0))
            .ok_or(ConstrainedFitError::EmptySliceError)
    }
}

/// Implementation for a fitter that performs PCA on points inside of the radius
/// to find the direction that minimizes variance
#[derive(Debug)]
pub struct SVDFitter;

impl Default for SVDFitter {
    fn default() -> Self {
        Self
    }
}

impl Fitter for SVDFitter {
    fn fit_segment(
        &self,
        data: &ArrayRef2<f32>,
        _sq_distances: &ArrayRef1<f32>,
        sq_radius: f32,
        vertex: &ArrayRef1<f32>,
    ) -> Result<Array1<f32>, ConstrainedFitError> {
        // Center data to vertex.
        // Unwrap because this should not fail.
        let centered = data - vertex.to_shape((1, vertex.len())).unwrap().to_owned();

        // PC1 is a unit vector. Scale by R and add back centering
        let mut direction = compute_pc1(&centered);

        // Check if the direction vector points with or against the average data trend.
        // We can dot product our direction with the sum of all centered points.
        let data_trend = centered.sum_axis(Axis(0));
        if direction.dot(&data_trend) < 0.0 {
            direction *= -1.0; // Flip it so it points toward the points
        }

        // Scale by R and add back the anchor vertex origin
        let radius = sq_radius.sqrt();
        let endpoint = (direction * radius) + vertex;

        Ok(endpoint)
    }
}
