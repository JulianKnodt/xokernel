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
  with_options
)]
#![allow(incomplete_features)]

pub mod bit_array;
pub mod block_interface;
pub mod linux_files;

fn main() {}

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
