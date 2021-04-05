#![feature(
  asm,
  abi_x86_interrupt,
  const_slice_from_raw_parts,
  const_raw_ptr_deref,
  const_mut_refs,
  const_fn,
  const_fn_transmute,
  maybe_uninit_uninit_array,
  maybe_uninit_extra,
  default_alloc_error_handler,
  const_generics,
  const_evaluatable_checked,
  associated_type_defaults,
  pub_macro_rules,
  generic_associated_types,
  array_methods,
  with_options
)]
#![allow(unused, incomplete_features)]

pub mod bit_array;
pub mod block_interface;
pub mod fs;
pub mod linux_files;

use block_interface::*;

static mut GLOBAL_BLOCK_INTERFACE: GlobalBlockInterface<linux_files::Driver> =
  GlobalBlockInterface::new(linux_files::Driver::new());

// TODO maybe move this into main because it doesn't need to be defined in this file.
pub fn global_block_interface() -> &'static mut GlobalBlockInterface<linux_files::Driver> {
  unsafe { &mut GLOBAL_BLOCK_INTERFACE }
}

fn main() {
  global_block_interface()
    .try_init()
    .expect("Failed to init global block interface");
  let mut fs = fs::FileSystem::new(global_block_interface());
  let root_dir_fd = fs
    .root_dir(fs::FileMode::R)
    .expect("Failed to open root dir");
  println!("{:?}", root_dir_fd);
  let fd = fs
    .open(root_dir_fd, &["test.txt"], fs::FileMode::RW)
    .expect("Failed to open test file");
  let mut example = b"A bunch of text";
  fs.seek(fd, fs::SeekFrom::End(0));
  fs.write(fd, example.as_slice());
  println!("{:?}", fs.stat(fd));
  /*
  println!("{:?}", fd);
  println!("{:?}", fs.stat(fd));
  */
}

#[test]
fn test() {
  let mut b = BitArray::<4096>::new(false);
  for i in 0..1000 {
    b.set(i);
    assert!(b.get(i));
  }
  assert_eq!(b.num_free(), 4096 - 1000);
  for i in 0..1000 {
    b.unset(i);
    assert!(!b.get(i));
  }
  assert!(b.find_free().is_some());
  assert_eq!(b.num_free(), 4096);
}
