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

#[repr(C)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct CommonHeader {
  device_id: u16,
  vendor_id: u16,
  status: u16,
  command: u16,
  class_code: u8,
  subclass: u8,
  prog_if: u8,
  revision_id: u8,
  bist: u8,
  header_type: u8,
  latency_timer: u8,
  cache_line_size: u8,
}

#[repr(C)]
#[derive(Debug)]
pub struct PCIHeader0 {
  bars: [u32; 6],
  cardbus_cis_pointer: u32,
  subsys_id: u16,
  subsys_vendor_id: u16,
  exp_rom_base_addr: u32,
  reserved: (u16, u8),
  cap_ptr: u8,
  reserved_2: u32,
  latency: (u8, u8),
  interrupt_pin: u8,
  interrupt_line: u8,
}

pub fn init_block_device_on_pci() {
  let mut buf = [0u8; 16];
  assert_eq!(core::mem::size_of::<CommonHeader>(), 16);
  for i in 0..4 {
    let raw: u32 = unsafe {
      PortWrite::write_to_port(CONFIG_ADDRESS, cfg(0, 4, 0, i * 4));
      PortRead::read_from_port(CONFIG_DATA)
    };
    let i = (i * 4) as usize;
    buf[i..i + 4].copy_from_slice(&raw.to_ne_bytes());
  }
  let header: CommonHeader = unsafe { core::mem::transmute(buf) };
  assert_eq!(header.device_id, 0x1af4, "Not a PCI virtio device");
  assert_eq!(header.vendor_id, 0x1001, "Not a block device");
  assert!(
    0x1000 < header.vendor_id && header.vendor_id < 0x107f,
    "Not a PCI virtio device"
  );

  if header.header_type == 0 {
    let l = core::mem::size_of::<PCIHeader0>();
    let mut buf = [0; core::mem::size_of::<PCIHeader0>()];
    for i in 0..(l / 4) as u8 {
      let raw: u32 = unsafe {
        PortWrite::write_to_port(CONFIG_ADDRESS, cfg(0, 4, 0, 0x10 + i * 4));
        PortRead::read_from_port(CONFIG_DATA)
      };
      let i = (i * 4) as usize;
      buf[i..i + 4].copy_from_slice(&raw.to_ne_bytes());
    }
    let header0: PCIHeader0 = unsafe { core::mem::transmute(buf) };
    // TODO what do I with a PCI Header now?
    write!(vga_buffer::Writer::new(14, 0), "{:x}", header0.bars[0]);
  } else {
    // Did not encounter this case yet, so no need to fix it up
    todo!();
  }
}
