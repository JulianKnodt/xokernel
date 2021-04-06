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

#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Bar(u32);

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BarMemWidth {
  U32 = 0,
  U64 = 2,
  Reserved = 1,
}
impl From<u32> for BarMemWidth {
  fn from(v: u32) -> Self {
    match v {
      0 => Self::U32,
      2 => Self::U64,
      1 => Self::Reserved,
      _ => panic!("Unknown BarMemWidth"),
    }
  }
}

impl Bar {
  #[inline]
  pub fn addr(self) -> u32 { self.0 & (!0xf) }
  #[inline]
  pub fn prefetchable(self) -> u32 { (self.0 & 0b1000) >> 3 }
  #[inline]
  pub fn kind(self) -> BarMemWidth { ((self.0 & 0b110) >> 1).into() }
  pub fn is_mem_space(self) -> bool { self.0 & 0b1 == 0 }
  #[inline]
  /// Attempts to determine the BAR's address space size as specified on OSDev.
  pub unsafe fn addr_space_size(self) -> u32 {
    match self.kind() {
      BarMemWidth::Reserved => todo!(),
      BarMemWidth::U64 => todo!(),
      BarMemWidth::U32 => {
        let v = self.addr() as *mut u32;
        let init = v.read_volatile();
        v.write_volatile(!0);
        let out = (!v.read_volatile()) + 1;
        v.write_volatile(init);
        out
      },
    }
  }
}

#[repr(C)]
#[derive(Debug)]
pub struct PCIHeader0 {
  bars: [Bar; 6],
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

#[repr(align(16384))]
struct DeviceSpace([u8; 128 + 4096 + 16384 + 16]);

/// Space for block device to put its memory mapped base address registers
static mut DEVICE_SPACE: DeviceSpace = DeviceSpace([0; 128 + 4096 + 16384 + 16]);

pub fn init_block_device_on_pci() -> PCIHeader0 {
  let mut buf = [0u8; 16];
  assert_eq!(core::mem::size_of::<CommonHeader>(), 16);
  for i in 0..4 {
    let raw: u32 = unsafe {
      // Sets the pointer in configuration space
      PortWrite::write_to_port(CONFIG_ADDRESS, cfg(0, 4, 0, i * 4));
      // Read from the pointer in configuration space
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
    // TODO read initial value of bar, write to the PORT and then read from it.
    // From there, we can statically allocate some space for the values to go.
    let mut curr = 0;
    for i in 0..6 {
      let raw: u32 = unsafe {
        PortWrite::write_to_port(CONFIG_ADDRESS, cfg(0, 4, 0, 0x10 + 4 * i));
        let og: u32 = PortRead::read_from_port(CONFIG_DATA);
        PortWrite::write_to_port(CONFIG_DATA, !0u32);
        let raw = PortRead::read_from_port(CONFIG_DATA);
        PortWrite::write_to_port(CONFIG_DATA, og);
        raw
      };
      let mem = (!(raw & (!0xF))).wrapping_add(1);
      if mem == 0 {
        continue;
      }
      let dst = unsafe { (&DEVICE_SPACE.0 as *const u8).add(curr) };
      write!(vga_buffer::Writer::new(6 + i as usize, 0), "{:b}", dst as u32);
      assert!((dst as u64) < u32::MAX as u64);
      unsafe {
        PortWrite::write_to_port(CONFIG_ADDRESS, cfg(0, 4, 0, 0x10 + 4 * i));
        PortWrite::write_to_port(CONFIG_DATA, dst as u32);
      };
      curr += mem as usize;
    }
    unsafe {
      write!(
        vga_buffer::Writer::new(13 as usize, 0),
        "{}",
        DEVICE_SPACE.0.iter().all(|&b| b == 0)
      );
    }
    /*
    for (i, &b) in header0.bars.iter().enumerate() {
      write!(vga_buffer::Writer::new(12 + i, 0), "{} {}", b.0, b.is_mem_space());
    }
    */
    return header0;
  } else {
    // Did not encounter this case yet, so no need to fix it up
    todo!();
  }
}
