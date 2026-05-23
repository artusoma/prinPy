//! Module implements constained local principal curve algorithms
//! https://www.sciencedirect.com/science/article/pii/S0377042715005956?via%3Dihub#s000090

use crate::utilities::*;
use ndarray::{Array1, Array2, ArrayRef1, ArrayRef2, ArrayViewMut1, ArrayViewMut2, s};
use thiserror::Error;

/// Trait that performs step 4 in 3.1
pub trait Fitter {
    /// Accepts points in circle, their distances, the max circle, and returns best-fit point
    fn fit_segment(
        &self,
        data: &ArrayRef2<f32>,
        sq_distances: &ArrayRef1<f32>,
        sq_radius: f32,
        vertex: &ArrayRef1<f32>,
    ) -> Result<Array1<f32>, ConstrainedFitError>;
}

/// Error type
#[derive(Error, Debug)]
pub enum ConstrainedFitError {
    #[error("No points were found in computational area.")]
    EmptySliceError,
    #[error("No points in found in radius. Please increase errortolerance")]
    NoPointsInRadius,
    #[error("Error calculating SVD")]
    SVDError(#[from] ndarray_linalg::error::LinalgError),
}

/// Iterator that yields the vertices of a constrained local principal curve fit.
///
/// Implements the algorithm from Kégl et al. (2015), section 3.1. Starting from the first
/// data point, each call to [`next`] finds the next best-fit vertex by:
/// 1. Collecting all points within a search radius of the current vertex.
/// 2. Delegating to a [`Fitter`] to find the candidate next vertex.
/// 3. Checking that the mean perpendicular error of points in the radius is within `max_error`.
/// 4. Halving the radius and retrying if the error check fails.
///
/// The first item yielded is always the first data point. The last item yielded is always
/// the last data point. Points are consumed and discarded as they are covered by a segment.
#[derive(Debug)]
pub struct ConstrainedFitIterator<F: Fitter> {
    /// Data to fit
    data: Array2<f32>,
    /// Algorithm to compute next point
    fitter: F,
    /// Max tolerable error before shrinking local search
    max_error: f32,
    /// Remaining points in data that are not fit yet.
    remaining: usize,
    /// Previous vertex to build from
    current_vertex: Array1<f32>,
    /// Final vertex to connect to
    final_vertex: Array1<f32>,
    /// If we just started; used to return the initial vertex
    initializing: bool,
}

impl<F: Fitter> std::iter::Iterator for ConstrainedFitIterator<F> {
    type Item = Result<Array1<f32>, ConstrainedFitError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.initializing {
            self.initializing = false;
            return Some(Ok(self.current_vertex.clone()));
        }

        if self.remaining == 0 {
            return None;
        }

        Some(self.get_next_vertex())
    }
}

/// Partitions `data` and `distances` in-place so that points outside `radius` occupy
/// `0..split` and points inside occupy `split..len`, returning the split index.
///
/// Both slices are reordered identically, so `distances[i]` always corresponds to `data[i]`.
/// The relative order within each partition is not preserved.
fn partition_on_distance(
    radius: f32,
    data: &mut ArrayViewMut2<f32>,
    distances: &mut ArrayViewMut1<f32>,
) -> usize {
    // Walk data. Partition data and distances into two sections:
    // 0..in = inside radius
    // in..end = inside radius (will drop on success)
    let mut write_idx: usize = data.nrows() - 1;
    for read_idx in (0..data.nrows()).rev() {
        // If read  is in the radius, swap with write
        if distances[read_idx] <= radius {
            // Check to avoid double mutable borrow
            if read_idx != write_idx {
                swap_row!(data, read_idx, write_idx, ..);
                swap_row!(distances, read_idx, write_idx);
            }
            write_idx -= 1;
        }
    }
    write_idx + 1
}

fn calc_error(
    data: &ArrayRef2<f32>,
    v1: &ArrayRef1<f32>,
    v2: &ArrayRef1<f32>,
) -> Result<f32, ConstrainedFitError> {
    distance_line_to_point(data, v1, v2)
        .mean()
        .ok_or(ConstrainedFitError::EmptySliceError)
}

impl<F: Fitter> ConstrainedFitIterator<F> {
    /// Advances the iterator by computing the next principal curve vertex.
    ///
    /// Starting from `current_vertex`, the search radius is initialised to the squared distance
    /// to `final_vertex`. The algorithm then loops:
    ///
    /// 1. Partitions the remaining data into points inside and outside the radius.
    /// 2. Calls [`Fitter::fit_segment`] on the inside points to get a candidate vertex.
    /// 3. Computes the mean perpendicular error of inside points to the segment
    ///    `current_vertex → candidate`.
    /// 4. If `error <= max_error`, accepts the candidate: discards covered points,
    ///    updates `current_vertex`, and returns `Ok(candidate)`.
    /// 5. Otherwise halves the squared radius and retries.
    ///
    /// Returns [`ConstrainedFitError::NoPointsInRadius`] if the radius shrinks until
    /// no points remain inside it.
    fn get_next_vertex(&mut self) -> Result<Array1<f32>, ConstrainedFitError> {
        let mut sq_radius: f32 = squared_distance(&self.current_vertex, &self.final_vertex) / 2.0;

        let mut data_ref = self.data.slice_mut(s![0..self.remaining, ..]);

        // Check if we can complete curve to end
        if calc_error(&data_ref, &self.current_vertex, &self.final_vertex)? <= self.max_error {
            self.remaining = 0;
            self.current_vertex = self.final_vertex.clone();
            return Ok(self.final_vertex.clone());
        }

        // Get distances from vertex to
        let mut sq_distances = data_ref
            .rows()
            .into_iter()
            .map(|p| squared_distance(&p, &self.current_vertex))
            .collect::<Array1<f32>>();

        loop {
            let cnt_out = partition_on_distance(
                sq_radius,
                &mut data_ref,
                &mut sq_distances.slice_mut(s![..]),
            );

            let in_circle = data_ref.slice(s![cnt_out..data_ref.nrows(), ..]);
            let in_distances = sq_distances.slice(s![cnt_out..data_ref.nrows()]);

            if in_circle.nrows() == 0 {
                return Err(ConstrainedFitError::NoPointsInRadius);
            }

            // Use fitter to find new segment
            let candidate_point = self.fitter.fit_segment(
                &in_circle,
                &in_distances,
                sq_radius,
                &self.current_vertex,
            )?;

            // Project found points in radius from segment from Pi to Pi+1, getting local error Ei
            let error = calc_error(&in_circle, &self.current_vertex, &candidate_point)?;

            // If Ei <= Emax, discard used points and break with candidate
            if error <= self.max_error {
                self.remaining = cnt_out;
                self.current_vertex = candidate_point.to_owned();
                break Ok(candidate_point);
            }
            sq_radius /= 2.;
        }
    }

    pub fn new(data: &ArrayRef2<f32>, max_error: f32, fitter: F) -> Self {
        let current_vertex = data.row(0).to_owned();
        let final_vertex = data.row(data.nrows() - 1).to_owned();
        let remaining: usize = data.nrows();
        Self {
            data: data.to_owned(),
            fitter,
            max_error,
            remaining,
            current_vertex,
            final_vertex,
            initializing: true,
        }
    }
}
