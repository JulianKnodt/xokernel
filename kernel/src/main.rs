#![no_std]
#![no_main]
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
  generic_associated_types,
  array_methods,
  int_bits_const,
  test,
  pub_macro_rules
)]
#![allow(unused, incomplete_features)]

extern crate alloc;

use alloc::prelude::v1::Box;
use bootloader::bootinfo::{BootInfo, MemoryRegionType};
use core::{fmt::Write, panic::PanicInfo};
use x86_64::{
  structures::{
    gdt::{Descriptor, GlobalDescriptorTable},
    idt::{InterruptDescriptorTable, InterruptStackFrame},
    tss::TaskStateSegment,
  },
  VirtAddr,
};

mod bit_array;
mod pci;
mod utils;
mod vga_buffer;
mod virtio;

mod fs;

mod block_interface;
use block_interface::{GlobalBlockInterface, Metadata};

mod allocator;

static mut GLOBAL_BLOCK_INTERFACE: GlobalBlockInterface<virtio::Driver> =
  GlobalBlockInterface::new(virtio::Driver::new());

// TODO maybe move this into main because it doesn't need to be defined in this file.
pub fn global_block_interface() -> &'static mut GlobalBlockInterface<virtio::Driver> {
  unsafe { &mut GLOBAL_BLOCK_INTERFACE }
}

const SETUP_DATA: *const SetupData = 0250u64 as *const SetupData;

#[repr(C)]
struct SetupData {
  next: u64,
  typ: u32,
  len: u32,
  data: *const u8,
}

#[no_mangle]
pub extern "C" fn _start(b_info: &'static BootInfo) -> ! {
  cli!();
  vga_buffer::print_at(b"Starting kernel...", 0, 0);

  init();
  vga_buffer::print_at(b"Finished Init", 1, 0);
  allocator::init_heap();
  vga_buffer::print_at(b"Finished Heap Init", 2, 0);

  pci::init_block_device_on_pci();

  /*
  let init = block_interface::global_block_interface().try_init(|bytes| {
    fs::Superblock::de(bytes).and_then(|(v, used)| {
      if used == bytes.len() {
        let v: Box<dyn Metadata> = Box::new(v);
        Ok(v)
      } else {
        Err(())
      }
    })
  });
  if init.is_err() {
    write!(
      vga_buffer::Writer::new(8, 0),
      "Failed to init block interface",
    );
  }
  */

  loop {}
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
  //  vga_buffer::print_at(b"Panicked with {}", 24, 0);
  write!(vga_buffer::Writer::new(21, 0), "{}", info,);
  loop {}
}

static mut IDT: InterruptDescriptorTable = InterruptDescriptorTable::new();
static mut TSS: TaskStateSegment = TaskStateSegment::new();
static mut GDT: GlobalDescriptorTable = GlobalDescriptorTable::new();

pub fn init() {
  // Init IST
  unsafe {
    TSS.interrupt_stack_table[0] = {
      const STACK_SIZE: usize = 4096 * 4;
      // Safe stack for when handling double faults
      static mut STACK: [u8; STACK_SIZE] = [0; STACK_SIZE];

      let stack_start = VirtAddr::from_ptr(&STACK);
      let stack_end = stack_start + STACK_SIZE;
      stack_end
    };
  }

  unsafe {
    let cs_seg = GDT.add_entry(Descriptor::kernel_code_segment());
    let tss_seg = GDT.add_entry(Descriptor::tss_segment(&TSS));
    GDT.load();
    x86_64::instructions::segmentation::set_cs(cs_seg);
    x86_64::instructions::tables::load_tss(tss_seg);
  }

  // Init IDT
  unsafe {
    IDT.breakpoint.set_handler_fn(breakpoint_handler);
    IDT
      .double_fault
      .set_handler_fn(double_fault_handler)
      .set_stack_index(0);
    IDT[80].set_handler_fn(syscall_handler);
    IDT.load();
  }
}

extern "x86-interrupt" fn breakpoint_handler(stack_frame: &mut InterruptStackFrame) {
  write!(
    vga_buffer::Writer::new(6, 0),
    "EXCEPTION: BREAKPOINT {:#?}",
    stack_frame
  );
}

extern "x86-interrupt" fn double_fault_handler(
  stack_frame: &mut InterruptStackFrame,
  error_code: u64,
) -> ! {
  write!(
    vga_buffer::Writer::new(6, 0),
    "EXCEPTION: DOUBLE FAULT({}) {:#?}",
    error_code,
    stack_frame
  );
  panic!();
}

extern "x86-interrupt" fn syscall_handler(stack_frame: &mut InterruptStackFrame) {
  // TODO get top 3(?) arguments of stack
  // match to find which syscall is being called
  // Then put return values on top of stack and return?
  panic!();
}

/*
const SYSCALL_COUNT: usize = 32;
static mut SYSCALLS: [*const usize; SYSCALL_COUNT] = [0 as *const _; SYSCALL_COUNT];

fn init_syscall(i: usize, defn: *const usize) {
  assert!(i < SYSCALL_COUNT);
  assert!(!defn.is_null());
  unsafe {
    SYSCALLS[i] = defn;
  }
}
*/
