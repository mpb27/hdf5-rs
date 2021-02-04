use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::ops::Deref;

use hdf5_sys::{
    h5::{H5_index_t, H5_iter_order_t},
    h5a::{H5A_info_t, H5A_operator2_t, H5Acreate2, H5Aiterate2},
};

use crate::internal_prelude::*;

/// Represents the HDF5 attribute object.
#[repr(transparent)]
#[derive(Clone)]
pub struct Attribute(Handle);

impl ObjectClass for Attribute {
    const NAME: &'static str = "attribute";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_ATTR];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    // TODO: short_repr()
}

impl Debug for Attribute {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for Attribute {
    type Target = Container;

    fn deref(&self) -> &Container {
        unsafe { self.transmute() }
    }
}

impl Attribute {
    /// Returns names of all the members in the group, non-recursively.
    pub fn attribute_names<T: ObjectClass>(obj: &T) -> Result<Vec<String>> {
        extern "C" fn attributes_callback(
            _id: hid_t, attr_name: *const c_char, _info: *const H5A_info_t, op_data: *mut c_void,
        ) -> herr_t {
            std::panic::catch_unwind(|| {
                let other_data: &mut Vec<String> = unsafe { &mut *(op_data as *mut Vec<String>) };
                other_data.push(string_from_cstr(attr_name));
                0 // Continue iteration
            })
            .unwrap_or(-1)
        }

        let callback_fn: H5A_operator2_t = Some(attributes_callback);
        let iteration_position: *mut hsize_t = &mut { 0_u64 };
        let mut result: Vec<String> = Vec::new();
        let other_data: *mut c_void = &mut result as *mut _ as *mut c_void;

        h5call!(H5Aiterate2(
            obj.handle().id(),
            H5_index_t::H5_INDEX_NAME,
            H5_iter_order_t::H5_ITER_INC,
            iteration_position,
            callback_fn,
            other_data
        ))?;

        Ok(result)
    }
}

#[derive(Clone)]
pub struct AttributeBuilder<T> {
    packed: bool,
    filters: Filters,
    parent: Result<Handle>,
    track_times: bool,
    phantom: PhantomData<T>,
}

impl<T: H5Type> AttributeBuilder<T> {
    /// Create a new dataset builder and bind it to the parent container.
    pub fn new(parent: &Group) -> Self {
        h5lock!({
            // Store the reference to the parent handle and try to increase its reference count.
            let handle = Handle::try_new(parent.id());
            if let Ok(ref handle) = handle {
                handle.incref();
            }

            Self {
                packed: false,
                filters: Filters::default(),
                parent: handle,
                track_times: false,
                phantom: PhantomData,
            }
        })
    }

    /// Create a new dataset builder and bind it to the parent container.
    pub fn new_from_dataset(parent: &Dataset) -> Self {
        h5lock!({
            // Store the reference to the parent handle and try to increase its reference count.
            let handle = Handle::try_new(parent.id());
            if let Ok(ref handle) = handle {
                handle.incref();
            }

            Self {
                packed: false,
                filters: Filters::default(),
                parent: handle,
                track_times: false,
                phantom: std::marker::PhantomData,
            }
        })
    }

    pub fn packed(&mut self, packed: bool) -> &mut Self {
        self.packed = packed;
        self
    }

    /// Enable or disable tracking object modification time (disabled by default).
    pub fn track_times(&mut self, track_times: bool) -> &mut Self {
        self.track_times = track_times;
        self
    }

    fn finalize<D: Dimension>(&self, name: &str, dims: D) -> Result<Attribute> {
        println!("getting type_descriptor");
        let type_descriptor = if self.packed {
            <T as H5Type>::type_descriptor().to_packed_repr()
        } else {
            <T as H5Type>::type_descriptor().to_c_repr()
        };

        h5lock!({
            let datatype = Datatype::from_descriptor(&type_descriptor)?;
            let parent = try_ref_clone!(self.parent);

            let dataspace = Dataspace::try_new(dims, false)?;

            let name = to_cstring(name)?;
            println!("making attr...");
            println!("making attr...");
            Attribute::from_id(h5try!(H5Acreate2(
                parent.id(),
                name.as_ptr(),
                datatype.id(),
                dataspace.id(),
                H5P_DEFAULT,
                H5P_DEFAULT,
            )))
        })
    }

    /// Create the dataset and link it into the file structure.
    pub fn create<D: Dimension>(&self, name: &str, d: D) -> Result<Attribute> {
        self.finalize(name, d)
    }
}

#[cfg(test)]
pub mod attribute_tests {
    use crate::internal_prelude::*;
    use ndarray::{arr2, Array2};
    use std::str::FromStr;
    use types::VarLenUnicode;

    #[test]
    pub fn test_shape_ndim_size() {
        with_tmp_file(|file| {
            let d = file.new_attribute::<f32>().create("name1", (2, 3)).unwrap();
            assert_eq!(d.shape(), vec![2, 3]);
            assert_eq!(d.size(), 6);
            assert_eq!(d.ndim(), 2);
            assert_eq!(d.is_scalar(), false);

            let d = file.new_attribute::<u8>().create("name2", ()).unwrap();
            assert_eq!(d.shape(), vec![]);
            assert_eq!(d.size(), 1);
            assert_eq!(d.ndim(), 0);
            assert_eq!(d.is_scalar(), true);
        })
    }

    #[test]
    pub fn test_get_file_attribute_names() {
        with_tmp_file(|file| {
            let _ = file.new_attribute::<f32>().create("name1", (2, 3)).unwrap();
            let _ = file.new_attribute::<u8>().create("name2", ()).unwrap();

            let attr_names = file.attribute_names().unwrap();
            assert_eq!(attr_names.len(), 2);
            assert!(attr_names.contains(&"name1".to_string()));
            assert!(attr_names.contains(&"name2".to_string()));
        })
    }

    #[test]
    pub fn test_get_dataset_attribute_names() {
        with_tmp_file(|file| {
            let ds = file.new_dataset::<u32>().create("d1", (10, 10)).unwrap();

            let _ = ds.new_attribute::<f32>().create("name1", (2, 3)).unwrap();
            let _ = ds.new_attribute::<u8>().create("name2", ()).unwrap();

            let attr_names = ds.attribute_names().unwrap();
            assert_eq!(attr_names.len(), 2);
            assert!(attr_names.contains(&"name1".to_string()));
            assert!(attr_names.contains(&"name2".to_string()));
        })
    }

    #[test]
    pub fn test_datatype() {
        with_tmp_file(|file| {
            assert_eq!(
                file.new_attribute::<f32>().create("name", 1).unwrap().dtype().unwrap(),
                Datatype::from_type::<f32>().unwrap()
            );
        })
    }

    #[test]
    pub fn test_read_write() {
        with_tmp_file(|file| {
            let arr = arr2(&[[1, 2, 3], [4, 5, 6]]);

            let attr = file.new_attribute::<f32>().create("foo", (2, 3)).unwrap();
            attr.as_writer().write(&arr).unwrap();

            let read_attr = file.attribute("foo").unwrap();
            assert_eq!(read_attr.shape(), vec![2, 3]);

            let arr_dyn: Array2<_> = read_attr.as_reader().read().unwrap();

            assert_eq!(arr, arr_dyn.into_dimensionality().unwrap());
        })
    }

    #[test]
    pub fn test_create() {
        with_tmp_file(|file| {
            let attr = file.new_attribute::<u32>().create("foo", (1, 2)).unwrap();
            assert!(attr.is_valid());
            assert_eq!(attr.shape(), vec![1, 2]);
            // FIXME - attr.name() returns "/" here, which is the name the attribute is connected to,
            // not the name of the attribute.
            //assert_eq!(attr.name(), "foo");
            assert_eq!(file.attribute("foo").unwrap().shape(), vec![1, 2]);
        })
    }

    #[test]
    pub fn test_missing() {
        with_tmp_file(|file| {
            let _ = file.new_attribute::<u32>().create("foo", (1, 2)).unwrap();

            let missing_result = file.attribute("bar");
            assert!(missing_result.is_err());
        })
    }

    #[test]
    pub fn test_write_read_str() {
        with_tmp_file(|file| {
            let s = VarLenUnicode::from_str("var len foo").unwrap();

            println!("file.new_attribute");
            let attr = file.new_attribute::<VarLenUnicode>().create("foo", ()).unwrap();

            println!("write attribute");
            attr.as_writer().write_scalar(&s).unwrap();

            println!("get attribute");
            let read_attr = file.attribute("foo").unwrap();

            println!("attribute shape");
            assert_eq!(read_attr.shape(), []);

            println!("read attribute");
            let r: VarLenUnicode = read_attr.as_reader().read_scalar().unwrap();

            assert_eq!(r, s);
        })
    }
}
