//! Module implements constained local principal curve algorithms
//! https://www.sciencedirect.com/science/article/pii/S0377042715005956?via%3Dihub#s000090

use ndarray::{Array1, Array2, ArrayRef1, ArrayRef2, Axis, Zip, s, stack};

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
    fn fit_segment(&self, data: &ArrayRef2<f32>, dist: &ArrayRef1<f32>, radius: f32)
    -> Array1<f32>;
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
    ) -> Array1<f32> {
        let inner_radius = self.slice_width * radius;
        let in_manifold = data.select(Axis(0), &{
            (0..dist.len())
                .filter(|&idx| dist[idx] > inner_radius)
                .collect::<Vec<usize>>()
        });
        let average = in_manifold.mean_axis(Axis(0)).unwrap();
        average
    }
}

/// This is an implementation of the general algorithm given in 3.1
pub struct ConstrainedAlgorithm<T: Fitter> {
    fitter: T,
    max_error: f32,
}

impl<T: Fitter> ConstrainedAlgorithm<T> {
    pub fn fit(&self, data: &ArrayRef2<f32>) -> Array2<f32> {
        let mut data = data.to_owned();

        // Initialize list of veritices. Start with first point of array as initial vertex.
        let end_pt = &data.row(data.nrows() - 1).to_owned();
        let mut vertices = Vec::<Array1<f32>>::new();
        vertices.push(data.slice(s![0, ..]).to_owned());

        loop {
            let prev_vertex = vertices.last().unwrap().to_owned();

            // Let upper bound of circle radius be Rr = 2 * dist (Pi, Pe)
            let mut radius: f32 = distance(&prev_vertex, &end_pt) / 2.;

            // Get distances
            let distances = data
                .rows()
                .into_iter()
                .map(|p| distance(&p, &prev_vertex))
                .collect::<Array1<f32>>();

            let candidate_point = loop {
                let (in_idxs, out_idxs): (Vec<usize>, Vec<usize>) =
                    (0..distances.len()).partition(|&i| distances[i] <= radius);

                let in_circle = data.select(Axis(0), &in_idxs);
                let in_distances = distances.select(Axis(0), &in_idxs);

                // Use fitter to find new segment
                let candidate_point = self.fitter.fit_segment(&in_circle, &in_distances, radius);

                // Project found points in radius from segment from Pi to Pi+1, getting local error Ei
                let error = get_distance_to_point(&in_circle, &prev_vertex, &candidate_point)
                    .mean()
                    .unwrap();

                // If Ei <= Emax, discard used points and break with candidate
                if error <= self.max_error {
                    data = data.select(Axis(0), &out_idxs);
                    break candidate_point;
                }
                radius /= 2.;
            };

            // push onto vertices
            vertices.push(candidate_point.to_owned());

            if data.nrows() == 0 {
                let view_vertices: Vec<_> = vertices.iter().map(|v| v.view()).collect();
                break stack(Axis(0), &view_vertices).unwrap();
            }
        }
    }

    pub fn new(fitter: T, tol: f32) -> Self {
        Self {fitter, max_error: tol}
    }
}
