#![no_std]
#![no_main]
#![feature(pub_macro_rules, asm)]
#![feature(abi_x86_interrupt)]
#![allow(unused)]

use core::{fmt::Write, panic::PanicInfo};
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame};

mod vga_buffer;
mod disk;
mod utils;

#[no_mangle]
pub extern "C" fn _start() -> ! {
  cli!();
  vga_buffer::print_at(b"Starting kernel...", 0, 0);
  init_syscall(0, req_pages as *const usize);

  init_idt();
  x86_64::instructions::interrupts::int3();
  loop {}
}

const SYSCALL_COUNT: usize = 32;
static mut SYSCALLS: [*const usize; SYSCALL_COUNT] = [0 as *const _; SYSCALL_COUNT];

/*
const PCBT_SIZE: usize = 64;
static mut PCBT: [PCB; PCBT_SIZE] = [PCB; PCBT_SIZE];
*/

fn init_syscall(i: usize, defn: *const usize) {
  assert!(i < SYSCALL_COUNT);
  assert!(!defn.is_null());
  unsafe {
    SYSCALLS[i] = defn;
  }
}

fn req_pages(n: usize) {
  todo!();
}

#[derive(Debug, PartialEq, Eq)]
struct CpbHandle(u32);

#[panic_handler]
fn _panic(_info: &PanicInfo) -> ! { loop {} }


pub fn sycall(i: usize) {
  assert!(i < SYSCALL_COUNT);
  todo!();
}

static mut IDT: InterruptDescriptorTable = InterruptDescriptorTable::new();

pub fn init_idt() {
  unsafe {
    IDT.breakpoint.set_handler_fn(breakpoint_handler);
    IDT.load();
  }
}

extern "x86-interrupt" fn breakpoint_handler(stack_frame: &mut InterruptStackFrame) {
  write!(
    vga_buffer::Writer::new(2, 0),
    "EXCEPTION: BREAKPOINT {:#?}",
    stack_frame
  );
}
