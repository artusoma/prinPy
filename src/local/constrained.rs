//! Module implements constained local principal curve algorithms
//! https://www.sciencedirect.com/science/article/pii/S0377042715005956?via%3Dihub#s000090

use ndarray::{Array, Array1, Array2, ArrayRef1, ArrayRef2, Axis, Zip, s, stack};
use thiserror::Error;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_constrained_iterator_step_by_step() {
        // Create a simple line of points: (0,0), (1,0), (2,0), (3,0)
        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let data_ref = data.view();

        // Tolerance high enough to accept the next point easily
        let mut iterator = ConstrainedFitIterator::new(&data_ref, 1.0, GreedyFitter::default());

        println!("Starting iteration...");

        // Check first point
        if let Some(result) = iterator.next() {
            match result {
                Ok(point) => println!("First candidate point: {:?}", point),
                Err(e) => panic!("First iteration failed with error: {:?}", e),
            }
        } else {
            panic!("Iterator returned None immediately");
        }
    }
}

fn distance(p1: &ArrayRef1<f32>, p2: &ArrayRef1<f32>) -> f32 {
    let diff = p1 - p2;
    diff.dot(&diff).sqrt()
}

/// Gets distance from each data point in data to the line
/// formed between vertices v1 and v2
///
/// For each point, takes the minimum
fn get_distance_to_point(
    data: &ArrayRef2<f32>,
    v1: &ArrayRef1<f32>,
    v2: &ArrayRef1<f32>,
) -> Array1<f32> {
    let l = Zip::from(v1).and(v2).map_collect(|v1, v2| v2 - v1);
    let mut distances = Array1::<f32>::zeros(data.nrows());
    Zip::from(&mut distances)
        .and(data.rows())
        .for_each(|dist, pt| {
            let p1 = Zip::from(pt).and(v1).map_collect(|a, b| a - b);
            *dist = (p1.dot(&p1) - (p1.dot(&l)).powf(2f32) / (l.dot(&l))).sqrt();
        });
    distances
}

/// Trait that performs step 4 in 3.1
pub trait Fitter {
    /// Accepts points in circle, their distances, the max circle, and returns best-fit point
    fn fit_segment(
        &self,
        data: &ArrayRef2<f32>,
        dist: &ArrayRef1<f32>,
        radius: f32,
    ) -> Result<Array1<f32>, ConstrainedFitError>;
}

pub struct GreedyFitter {
    slice_width: f32,
}

impl Default for GreedyFitter {
    fn default() -> Self {
        GreedyFitter { slice_width: 0.9 }
    }
}

impl Fitter for GreedyFitter {
    fn fit_segment(
        &self,
        data: &ArrayRef2<f32>,
        dist: &ArrayRef1<f32>,
        radius: f32,
    ) -> Result<Array1<f32>, ConstrainedFitError> {
        let inner_radius = self.slice_width * radius;
        let indices: Vec<usize> = (0..dist.len())
            .filter(|&idx| dist[idx] > inner_radius)
            .collect();

        if indices.is_empty() {
            // FALLBACK: If no points are in the outer shell, just take the mean of all points in the circle
            return data
                .mean_axis(Axis(0))
                .ok_or(ConstrainedFitError::GenericError);
        }

        let in_manifold = data.select(Axis(0), &indices);
        in_manifold
            .mean_axis(Axis(0))
            .ok_or(ConstrainedFitError::GenericError)
    }
}

/// Error type
#[derive(Error, Debug)]
pub enum ConstrainedFitError {
    #[error("No points in fitter")]
    GenericError,
    #[error("No points in radius")]
    NoPointsInRadius,
}

pub struct ConstrainedFitIterator<'a, F: Fitter> {
    /// Data to fit
    data: &'a ArrayRef2<f32>,
    /// Algorithm to compute next point
    fitter: F,
    /// Max tolerable error before shrinking local search
    max_error: f32,
    /// Remaining points in data that are not fit yet.
    remaining: Vec<bool>,
    /// Previous vertex to build from
    prev_vertex: Array1<f32>,
    /// Final vertex to connect to
    final_vertex: Array1<f32>,
    /// If we just started; used to return the initial vertex
    initializing: bool,
}

impl<'a, F: Fitter> std::iter::Iterator for ConstrainedFitIterator<'a, F> {
    type Item = Result<Array1<f32>, ConstrainedFitError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.initializing {
            self.initializing = false;
            return Some(Ok(self.prev_vertex.clone()));
        }

        if !self.remaining.iter().any(|&r| r) {
            return None;
        }

        Some(self.get_next_vertex())
    }
}

impl<'a, F: Fitter> ConstrainedFitIterator<'a, F> {
    fn calc_error(
        &self,
        data: &ArrayRef2<f32>,
        v1: &ArrayRef1<f32>,
        v2: &ArrayRef1<f32>,
    ) -> Result<f32, ConstrainedFitError> {
        get_distance_to_point(data, v1, v2)
            .mean()
            .ok_or(ConstrainedFitError::GenericError)
    }

    fn get_next_vertex(&mut self) -> Result<Array1<f32>, ConstrainedFitError> {
        let mut radius: f32 = distance(&self.prev_vertex, &self.final_vertex) / 2.;

        // Check if we can complete right to end
        let d2 = (0..self.data.nrows())
            .filter(|i| self.remaining[*i])
            .collect::<Vec<usize>>();
        if self.calc_error(
            &self.data.select(Axis(0), &d2),
            &self.prev_vertex,
            &self.final_vertex,
        )? <= self.max_error
        {
            self.remaining.iter_mut().for_each(|b| *b = false);
            self.prev_vertex = self.final_vertex.clone();
            return Ok(self.final_vertex.clone());
        }

        // Get distances from previuos vertex to each remaining point in data
        let distances = self
            .data
            .rows()
            .into_iter()
            .map(|p| distance(&p, &self.prev_vertex))
            .collect::<Array1<f32>>();

        loop {
            let in_radius = distances
                .iter()
                .zip(self.remaining.iter())
                .map(|(&d, &r)| (d <= radius) && r)
                .collect::<Vec<bool>>();
            let in_idxs: Vec<usize> = (0..distances.len()).filter(|&i| in_radius[i]).collect();

            let in_circle = self.data.select(Axis(0), &in_idxs);
            let in_distances = distances.select(Axis(0), &in_idxs);

            if in_circle.nrows() == 0 {
                return Err(ConstrainedFitError::NoPointsInRadius);
            }

            // Use fitter to find new segment
            let candidate_point = self.fitter.fit_segment(&in_circle, &in_distances, radius)?;

            // Project found points in radius from segment from Pi to Pi+1, getting local error Ei
            let error = self.calc_error(&in_circle, &self.prev_vertex, &candidate_point)?;

            // If Ei <= Emax, discard used points and break with candidate
            if error <= self.max_error {
                self.remaining
                    .iter_mut()
                    .zip(in_radius.iter())
                    .for_each(|(x, &b)| {
                        if b {
                            *x = false
                        }
                    });
                self.prev_vertex = candidate_point.clone();
                break Ok(candidate_point);
            }
            radius /= 2.;
        }
    }

    pub fn new(data: &'a ArrayRef2<f32>, max_error: f32, fitter: F) -> Self {
        let prev_vertex = data.row(0).to_owned();
        let final_vertex = data.row(data.nrows() - 1).to_owned();
        let mut remaining: Vec<bool> = (0..data.nrows()).map(|_| true).collect();
        remaining[0] = false;
        Self {
            data,
            fitter,
            max_error,
            remaining,
            prev_vertex,
            final_vertex,
            initializing: true,
        }
    }
}
