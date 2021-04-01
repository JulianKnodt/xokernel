use core::ptr::slice_from_raw_parts_mut;
use x86_64::structures::port::{PortRead, PortWrite};
use lazy_static::lazy_static;

static mut CONFIG_UART_BASE: *mut u32 = 0x09000000 as *mut u32;

/// Hopefully will just return the io-ports directly
#[inline]
fn io_ports() -> &'static mut [u8] {
  unsafe { core::mem::transmute(slice_from_raw_parts_mut(0x3f8 as *mut u8, 0x3ff - 0x3f8)) }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct UArtDriver {

}

lazy_static! {
    /// The initialized UArtDriver
    static ref UART_DRIVER: UArtDriver = UArtDriver::new();
}

pub fn init() {
}

pub fn print(v: &str) {
  // Write data to uart driver
  todo!()
}

impl UArtDriver {
  pub fn new() -> Self {
    UArtDriver {}
  }
}
