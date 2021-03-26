use alloc::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use linked_list_allocator::LockedHeap;

/*
pub struct Dummy;

unsafe impl GlobalAlloc for Dummy {
  unsafe fn alloc(&self, _layout: Layout) -> *mut u8 { null_mut() }

  unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
    panic!("dealloc should be never called")
  }
}
*/

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();
//static ALLOCATOR: Dummy = Dummy;

pub fn init_heap() {
  // Assume this is only called once
  //unsafe { ALLOCATOR.lock().init(HEAP_START, HEAP_SIZE) }
}

pub const HEAP_START: usize = 0xdead_0000;
pub const HEAP_SIZE: usize = 16 * 1024;
