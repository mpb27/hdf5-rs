use std::convert::AsRef;
use ndarray::SliceOrIndex;
use hdf5_sys::{
    h5r::{H5Rcreate, H5R_type_t, hdset_reg_ref_t},
};

use hdf5_types::TypeDescriptor;

use crate::internal_prelude::*;

//
// References
//
// - https://hdf5.wiki/index.php/RFC:RFC-THG-2018-07-12
//
//

/// Represents the HDF5 reference object.
#[derive(Clone)]
pub struct Reference
{
    reg_ref: hdset_reg_ref_t,
}

#[cfg(not(hdf5_1_12_0))]
fn h5r_dataset_region() -> H5R_type_t { H5R_type_t::H5R_DATASET_REGION }

#[cfg(hdf5_1_12_0)]
fn h5r_dataset_region() -> H5R_type_t { H5R_type_t::H5R_DATASET_REGION1 }

impl Reference {
    pub fn new_region<S>(parent: &Dataset, slice: S) -> Result<Self>
    where
        S: AsRef<[SliceOrIndex]>,
    {
        h5lock!({

            let space = parent.space().unwrap().copy();
            let _vec = space.select_slice(slice);

            let mut r = Self {
                reg_ref: Default::default(),
            };

            let name = to_cstring(".").unwrap();

            h5call!(H5Rcreate(
                r.reg_ref.as_mut_ptr() as *mut c_void,
                parent.id(),
                name.as_ptr(),
                h5r_dataset_region(),
                space.id()
            )).and(Ok(r))
        })
    }
}


unsafe impl H5Type for Reference {
    #[inline]
    fn type_descriptor() -> TypeDescriptor {
        TypeDescriptor::Reference
    }
}


#[cfg(test)]
pub mod tests {
    use crate::internal_prelude::*;

    #[test]
    pub fn test_references() {
        with_tmp_file(|file| {

            let ds = file.new_dataset::<u32>().create("foo",(1)).unwrap();
            ds.write(&ndarray::arr1(&[42])).unwrap();
            assert!(ds.is_valid());
            assert_eq!(ds.name(), "/foo");

            let r = Reference::new_region(&ds, ndarray::s![0..1]).unwrap();

            let ds = file.new_dataset::<Reference>().create("bar", ()).unwrap();
            ds.write_scalar(&r).unwrap();
            //assert!(at.is_valid());
            //assert_eq!(ds.shape(), vec![1, 2]);
            assert_eq!(ds.name(), "/bar");
            //assert_eq!(file.group("foo").unwrap().dataset("bar").unwrap().shape(), vec![1, 2]);
            //ds.write_scalar()
            //assert_eq!(ds.shape(), vec![2, 3]);
        })
    }
}
