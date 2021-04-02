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
  let fs = fs::FileSystem::new(global_block_interface());
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
