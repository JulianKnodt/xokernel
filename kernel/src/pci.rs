use crate::vga_buffer;
use core::fmt::Write;
use x86_64::structures::port::{PortRead, PortWrite};

macro_rules! bit_fn {
  ($bit: expr, $name: ident $(, $set_name: ident )?) => {
    #[inline]
    pub(crate) fn $name(self) -> bool { ((self.0 >> $bit) & 0b1) == 1 }
    $(
      #[inline]
      pub(crate) fn $set_name(self, v: bool)  -> Self {
        let mask = 1 << $bit;
        if v {
          Self(self.0 | mask)
        } else {
          Self(self.0 & (!mask))
        }
      }
    )?
  }
}

macro_rules! bit_register {
  ($name: ident, $size: ty, [$( ($bit: expr, $fn_name: ident $(,$set_name: ident)? ) $(,)?)*]) => {
    #[repr(transparent)]
    #[derive(Copy, Clone, PartialEq, Debug, Eq)]
    struct $name($size);
    impl $name {
      $( bit_fn!($bit, $fn_name $(, $set_name )?); )*
    }
  }
}

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

bit_register!(Status, u16, [
  (3, interrupt_status),
  (4, capabilities_list),
  (15, detected_parity_error),
  (8, master_data_parity_error),
]);

bit_register!(Command, u16, [
  (0, io_space, set_io_space),
  (1, mem_space, set_mem_space),
]);

#[repr(C)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct CommonHeader {
  device_id: u16,
  vendor_id: u16,
  status: Status,
  command: Command,
  class_code: u8,
  subclass: u8,
  prog_if: u8,
  revision_id: u8,
  bist: u8,
  header_type: u8,
  latency_timer: u8,
  cache_line_size: u8,
}

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
impl BarMemWidth {
  fn u32(self) -> bool { Self::U32 == self }
}

bit_register!(Bar, u32, [(0, io_space),]);

impl Bar {
  #[inline]
  pub fn addr(self) -> u32 { self.0 & (!0xf) }
  #[inline]
  pub fn prefetchable(self) -> bool { ((self.0 & 0b1000) >> 3) == 1 }
  #[inline]
  pub fn kind(self) -> BarMemWidth { ((self.0 & 0b110) >> 1).into() }
  pub fn mem_space(self) -> bool { (self.0 & 0b1) == 0 }
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

#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
struct DeviceSpace([u8; 16384]);

macro_rules! read_into_buf {
  ($size: expr, $offset:expr, [ $( $per: ty $(,)?)* ]) => {{
    let mut buf = [0u8; $size];
    let mut curr = 0;
    $(
      let raw: $per = unsafe {
        // Sets the pointer in configuration space
        PortWrite::write_to_port(CONFIG_ADDRESS, cfg(0, 4, 0, curr as u8 + $offset));
        // Read from the pointer in configuration space
        PortRead::read_from_port(CONFIG_DATA)
      };
      let len = <$per>::BITS/8;
      buf[curr as usize..curr as usize +len as usize].copy_from_slice(&raw.to_ne_bytes());
      curr += len as u32;
    )*
    buf
  }}
}

/// Initializes the block device on PCI
pub fn init_block_device_on_pci() -> PCIHeader0 {
  let mut buf = read_into_buf!(core::mem::size_of::<CommonHeader>(), 0, [
    u32, u32, u32, u32
  ]);
  let mut header: CommonHeader = unsafe { core::mem::transmute(buf) };
  assert_eq!(header.device_id, 0x1af4, "Not a PCI virtio device");
  assert_eq!(header.vendor_id, 0x1001, "Not a block device");
  assert!(
    0x1000 < header.vendor_id && header.vendor_id < 0x107f,
    "Not a PCI virtio device"
  );

  if header.header_type != 0 {
    todo!();
  }

  let mut buf = read_into_buf!(core::mem::size_of::<PCIHeader0>(), 0x10, [
    u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32,
  ]);
  let header0: PCIHeader0 = unsafe { core::mem::transmute(buf) };

  // ---- Finished reading in headers
  let raw: u32 = unsafe {
    PortWrite::write_to_port(CONFIG_ADDRESS, cfg(0, 4, 0, 0x10));
    let og: u32 = PortRead::read_from_port(CONFIG_DATA);
    PortWrite::write_to_port(CONFIG_DATA, !0u32);
    let raw = PortRead::read_from_port(CONFIG_DATA);
    PortWrite::write_to_port(CONFIG_DATA, og);
    raw
  };
  let mem = (!(raw & (!0xF))).wrapping_add(1);
  assert_eq!(mem, 128);
  let bar0 = header0.bars[0];
  assert!(bar0.io_space());
  let mut buf = [0u8; 128];
  let base_addr = bar0.addr();
  for i in 0..mem {
    let v: u8 = unsafe { PortRead::read_from_port(base_addr as u16 + i as u16) };
    buf[i as usize] = v;
  }
  let first_pci_cap: LegacyVirtioCommonCfg =
    unsafe { *(buf.as_ptr() as *const LegacyVirtioCommonCfg) };
  #[repr(align(16384))]
  struct Queue([u8;16384]);
  static mut Q: Queue = Queue([0;16384]);
  //write!(vga_buffer::Writer::new(16, 0), "{:x?}", first_pci_cap);
  unsafe {
    PortWrite::write_to_port(base_addr as u16 + 18, VirtioStatus::Reset as u8);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    PortWrite::write_to_port(base_addr as u16 + 18, VirtioStatus::Ack as u8);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    PortWrite::write_to_port(base_addr as u16 + 18, VirtioStatus::Driver as u8);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    let v: u8 = PortRead::read_from_port(base_addr as u16 + 18);
    write!(vga_buffer::Writer::new(6, 0), "{:x}", v);
  }
  for i in 0..mem {
    let v: u8 = unsafe { PortRead::read_from_port(base_addr as u16 + i as u16) };
    buf[i as usize] = v;
  }
  let first_pci_cap: LegacyVirtioCommonCfg =
    unsafe { *(buf.as_ptr() as *const LegacyVirtioCommonCfg) };
  write!(vga_buffer::Writer::new(7, 0), "{:x?}", first_pci_cap);
  unsafe {
    PortWrite::write_to_port(base_addr as u16 + 8, Q.0.as_ptr() as u32/4096);
    for i in 0..16384 {
      let v = (Q.0.as_ptr() as *const u8).add(i).read_volatile();
      if v != 0 {
        write!(vga_buffer::Writer::new(15, 0), "Got something");
      }
      let v: u32 = PortRead::read_from_port(Q.0.as_ptr() as u16);
      write!(vga_buffer::Writer::new(15, 0), "{:x}", v);
    }
  };
  return header0;
}

/// Found in bar0
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[repr(C)]
pub struct LegacyVirtioCommonCfg {
  // R
  device_features: u32,
  // R
  driver_feature_bits: u32,
  // R+W
  queue_addr: u32,
  // R
  queue_size: u16,
  // R+W
  queue_select: u16,
  // R+W
  queue_notify: u16,
  // R+W
  device_status: u8,
  // R
  isr_status: u8,
}

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum VirtioPCICapCfg {
  /* Common configuration */
  CommonCfg = 1,
  /* Notifications */
  NotifyCfg = 2,
  /* ISR Status */
  ISR = 3,
  /* Device specific configuration */
  Device = 4,
  /* PCI configuration access */
  PCI = 5,
}

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum VirtioStatus {
  Reset = 0,
  Ack = 1,
  Driver = 2,
  Failed = 128,
  FeaturesOk = 8,
  DriverOk = 4,
  DeviceNeedsReset = 64,
}

#[repr(C)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct VirtioPCICap {
  cap_vndr: u8,     /* Generic PCI field: PCI_CAP_ID_VNDR */
  cap_next: u8,     /* Generic PCI field: next ptr. */
  cap_len: u8,      /* Generic PCI field: capability length */
  cfg_type: u8,     //VirtioPCICapCfg, /* Identifies the structure. */
  bar: u8,          /* Where to find it. */
  padding: [u8; 3], /* Pad to full dword. */
  // The fields below should be encoded in memory as little endian
  offset: u32, /* Offset within bar. */
  length: u32, /* Length of the structure, in bytes. */
}
