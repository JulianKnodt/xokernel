use crate::vga_buffer;
use core::fmt::Write;
use x86_64::structures::port::{PortRead, PortWrite};

// Location of where things are required to be accessed(?)
const CONFIG_ADDRESS: u16 = 0x0cf8;
// Actually generates configuration data to read
const CONFIG_DATA: u16 = 0x0cfc;
const BASE: u32 = 0x80000000;

// Offset here refers to the register of the thing we want
pub fn cfg(bus: u8, slot: u8, func: u8, offset: u8) -> u32 {
  BASE
    | ((bus as u32) << 16)
    | ((slot as u32) << 11)
    | ((func as u32) << 8)
    | ((offset as u32) & 0xfc)
}

pub fn enumerate() -> impl Iterator<Item = u32> {
  (0..256).flat_map(|bus| {
    (0..32).map(|device| {
      // TODO
      todo!()
    })
  })
}

// TODO setup
pub fn read() -> u32 {
  // The virtio block device is at cfg(0,4,0,0)!

  /*
  unsafe {
    PortWrite::write_to_port(CONFIG_ADDRESS, cfg(0,4,0,0));
    PortRead::read_from_port(CONFIG_DATA)
  }
  */
  unsafe {
    PortWrite::write_to_port(CONFIG_ADDRESS, cfg(0, 4, 0, 0x10));
    PortRead::read_from_port(CONFIG_DATA)
  }
}

pub fn init_block_device_on_pci() {
  let value: u32 = unsafe {
    PortWrite::write_to_port(CONFIG_ADDRESS, cfg(0, 4, 0, 0));
    PortRead::read_from_port(CONFIG_DATA)
  };
  assert_eq!(value & 0xFFFF, 0x1af4, "Not a PCI virtio device");
  let device_id = value >> 16;
  assert!(
    0x1000 < device_id && device_id < 0x107f,
    "Not a PCI virtio device"
  );
  assert_eq!(device_id, 0x1001, "Not a block device");

  // TODO actually read and write stuff to the device
}
