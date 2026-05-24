//! General utilities

use ndarray::Zip;
use ndarray::prelude::*;

/// swap_row! swaps elements in `array` between `row_a` and `row_b`
#[macro_export]
macro_rules! swap_row {
    ($array:ident, $row_a:ident, $row_b:ident $(, $indices:expr),*) => {
        let (mut x, mut y) = $array.multi_slice_mut((s![$row_a $(, $indices),*], s![$row_b $(, $indices),*]));
        x.iter_mut()
            .zip(y.iter_mut())
            .for_each(|(a, b)| std::mem::swap(a, b));
    };
}
pub(crate) use swap_row;

pub fn distance_line_to_point(
    data: &ArrayRef2<f32>,
    v1: &ArrayRef1<f32>,
    v2: &ArrayRef1<f32>,
) -> Array1<f32> {
    let l = Zip::from(v1).and(v2).map_collect(|v1, v2| v2 - v1);
    let l_dot_l = l.dot(&l);
    Zip::from(data.rows()).map_collect(|pt| {
        let (dot_pp, dot_pl) = pt.iter().zip(v1.iter()).zip(l.iter()).fold(
            (0f32, 0f32),
            |(dot1, dot2), ((ptv, v1v), lv)| {
                let d = ptv - v1v;
                (dot1 + d * d, dot2 + d * lv)
            },
        );
        (dot_pp - dot_pl * dot_pl / l_dot_l).sqrt()
    })
}

pub fn outer_product(x: &ArrayRef1<f32>, y: &ArrayRef1<f32>) -> Array2<f32> {
    let (size_x, size_y) = (x.shape()[0], y.shape()[0]);
    let x_reshaped = x.view().into_shape_with_order((size_x, 1)).unwrap();
    let y_reshaped = y.view().into_shape_with_order((1, size_y)).unwrap();
    x_reshaped.dot(&y_reshaped)
}

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

pub fn compute_pc1(data: &Array2<f32>) -> Array1<f32> {
    let b = data.t().dot(data);
    let ncols = data.ncols();

    let mut v: Array1<f32> = Array1::from_elem(ncols, 1.0);
    let norm = v.dot(&v).sqrt(); 
    v.mapv_inplace(|x| x / norm); // Idiomatic in-place division

    let mut prev_v = v.clone();

    for _ in 0..50 {
        let mut v_new = b.dot(&v);

        let norm = v_new.dot(&v_new).sqrt();
        v_new.mapv_inplace(|x| x / norm);

        let dot_product = v_new.dot(&prev_v).abs();
        if 1.0 - dot_product < 1e-6 {
            v = v_new;
            break;
        }

        // Update vectors for the next iteration
        prev_v = v_new.clone();
        v = v_new;
    }

    v
}
