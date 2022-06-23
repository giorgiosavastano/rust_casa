use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;


fn euclidean_cdist(a: Array2<f64>, b: Array2<f64>) -> Array2<f64> {

    let mut c = Array2::<f64>::zeros((a.shape()[0], b.shape()[0]));

    for (i, row_a)in a.outer_iter().enumerate(){
       for (j, row_b)in b.outer_iter().enumerate() {
            let diff = &row_a - &row_b;
            let squa = diff.mapv(|diff| diff.powi(2));
            let sum = squa.sum();

            c[[i, j]] = sum;
       }
    }
    c
}


fn main() {
    let a = Array2::<f64>::random((10, 5), Uniform::new(0., 10.));
    let b = Array2::<f64>::random((10, 5), Uniform::new(0., 10.));
    println!("{:8.4}", a);
    println!("{:8.4}", b);

    let c = euclidean_cdist(a, b);

    println!("{}", c)

}
