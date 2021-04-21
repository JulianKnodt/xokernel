use crate::vga_buffer;
use core::fmt::Write;
use x86_64::structures::port::{PortRead, PortWrite};

macro_rules! bit_fn {
  ($bit: expr, $name: ident $(, $set_name: ident )?) => {
    #[inline]
    pub(crate) fn $name(self) -> bool { ((self.0 >> $bit) & 0b1) == 1 }
    $(
      #[inline]
      #[must_use]
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
    #[derive(Copy, Clone, PartialEq, Debug, Eq, Default)]
    struct $name($size);
    impl $name {
      #[inline]
      pub const fn empty() -> Self { Self(0) }
      $( bit_fn!($bit, $fn_name $(, $set_name )?); )*
    }
    impl From<$size> for $name {
      fn from(s: $size) -> Self {
        Self(s)
      }
    }
    impl core::ops::BitOr for $name {
      type Output = Self;
      fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
      }
    }
  }
}

// Location of where things are required to be accessed(?)
const CONFIG_ADDRESS: u16 = 0xcf8;
// Actually generates configuration data to read
const CONFIG_DATA: u16 = 0xcfc;
const BASE: u32 = 0x80000000;

// Offset here refers to the register of the thing we want
#[inline]
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
  (6, parity_error_response, set_parity_error_response),
  (2, bus_master, set_bus_master),
  (10, interrupt_disable),
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
  pub fn mem_space(self) -> bool { !self.io_space() }
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

bit_register!(DescTableFlags, u16, [
  (0, next, set_next),
  (1, write_only, set_write_only),
  (2, indirect, set_indirect),
]);

#[repr(align(16))]
#[derive(Debug, Default, Clone, Copy)]
pub struct DescTable {
  addr: u64,
  len: u32,
  flags: DescTableFlags,
  next: u16,
}

impl DescTable {
  const fn new() -> Self {
    DescTable {
      addr: 0,
      len: 0,
      flags: DescTableFlags::empty(),
      next: 0,
    }
  }
}

bit_register!(VirtAvailFlags, u16, [(0, no_interrupt)]);

#[repr(align(2))]
#[derive(Debug)]
pub struct AvailableRing<const QS: usize> {
  flags: VirtAvailFlags,
  idx: u16,
  ring: [u16; QS],
}

impl<const QS: usize> AvailableRing<QS> {
  const fn new() -> Self {
    Self {
      flags: VirtAvailFlags::empty(),
      idx: 0,
      ring: [0; QS],
    }
  }
}

bit_register!(VirtQUsedFlags, u16, [(0, no_interrupt),]);

#[repr(align(4))]
#[derive(Debug)]
pub struct UsedRing<const QS: usize> {
  flags: VirtQUsedFlags,
  idx: u16,
  used_elems: [UsedElem; QS],
}

impl<const QS: usize> UsedRing<QS> {
  const fn new() -> Self {
    Self {
      flags: VirtQUsedFlags::empty(),
      idx: 0,
      used_elems: [UsedElem { idx: 0, len: 0 }; QS],
    }
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct UsedElem {
  idx: u32,
  len: u32,
}

const fn q_align_padding<const QS: usize>() -> usize {
  (QS * core::mem::size_of::<DescTable>() + core::mem::size_of::<AvailableRing<QS>>()) % Q_ALIGN
}

#[repr(C)]
#[repr(align(4096))]
#[derive(Debug)]
struct VirtQueue<const QS: usize>
where
  [(); q_align_padding::<QS>()]: , {
  descriptors: [DescTable; QS],
  avail: AvailableRing<QS>,
  padding: [u8; q_align_padding::<QS>()],
  used: UsedRing<QS>,
}

impl<const QS: usize> VirtQueue<QS>
where
  [(); q_align_padding::<QS>()]: ,
{
  const fn new() -> Self {
    let mut descriptors = [DescTable::new(); QS];
    let mut i = 0;
    while i < QS {
      descriptors[i].next = i as u16 + 1;
      i += 1;
    }

    Self {
      descriptors,
      avail: AvailableRing::new(),
      padding: [0; q_align_padding::<QS>()],
      used: UsedRing::new(),
    }
  }
}

#[derive(Debug)]
struct VirtQueueMetadata<const QS: usize>
where
  [(); q_align_padding::<QS>()]: , {
  vq: VirtQueue<QS>,
  /// Pointer to beginning of free list in the virtqueue descriptor table.
  free_ptr: u16,
  /// Base addr of port at which this VirtQueue is.
  base_addr: u16,
  requests: [(VirtioBlkRequest, bool); 256],
}

impl<const QS: usize> VirtQueueMetadata<QS>
where
  [(); q_align_padding::<QS>()]: ,
{
  const fn new() -> Self {
    Self {
      vq: VirtQueue::new(),
      free_ptr: 0,
      base_addr: 0,
      requests: [(VirtioBlkRequest::new(VirtioBlkRequestType::Read, 0), false); 256],
    }
  }
  fn alloc_descriptor(&mut self, addr: u64) -> Result<u16, ()> {
    let desc = self.free_ptr;
    if desc as usize == QS {
      return Err(());
    }
    let entry = &mut self.vq.descriptors[desc as usize];
    let next = entry.next;
    self.free_ptr = next;
    entry.addr = addr;
    Ok(desc)
  }
  fn free_descriptor(&mut self, desc: u16) {
    let entry = &mut self.vq.descriptors[desc as usize];
    entry.next = self.free_ptr;
    self.free_ptr = desc;
  }
  fn blk_cmd(&mut self, ty: VirtioBlkRequestType, sector: u64, data: &[u8]) -> Result<(), ()> {
    let v: u16 = unsafe { PortRead::read_from_port(self.base_addr + 14) };
    assert_eq!(v, 0);
    let v: u16 = unsafe { PortRead::read_from_port(self.base_addr + 0x12) };
    let p = self.requests.iter().position(|r| !r.1).ok_or(())?;
    self.requests[p] = (VirtioBlkRequest::new(ty, sector), true);
    let req_desc = self.alloc_descriptor((&self.requests[p].0) as *const _ as u64)?;
    let req_entry = &mut self.vq.descriptors[req_desc as usize];
    req_entry.len = 16;
    req_entry.flags = DescTableFlags::empty().set_next(true);

    let data_desc = self.alloc_descriptor(data.as_ptr() as u64)?;
    let data_entry = &mut self.vq.descriptors[data_desc as usize];
    data_entry.len = 512;
    data_entry.flags = DescTableFlags::empty().set_next(true).set_write_only(true);

    let status_desc = self.alloc_descriptor((&self.requests[p].0.status as *const _) as u64)?;
    let status_entry = &mut self.vq.descriptors[status_desc as usize];
    status_entry.len = 1;
    status_entry.flags = DescTableFlags::empty().set_write_only(true);

    self.vq.descriptors[req_desc as usize].next = data_desc;
    self.vq.descriptors[data_desc as usize].next = status_desc;

    self.vq.avail.ring[self.vq.avail.idx as usize] = data_desc;
    self.vq.avail.idx += 1;
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);


    unsafe {
      PortWrite::write_to_port(self.base_addr + 16, 0u16);
    }

    Ok(())
  }
}

const Q_ALIGN: usize = 4096;
const fn virtqueue_size(queue_size: u16) -> u32 {
  const fn align(v: usize) -> usize {
    (v + Q_ALIGN) & (!Q_ALIGN)
  }
  let qs = queue_size as usize;
  let u16s = core::mem::size_of::<u16>();
  let out = align(core::mem::size_of::<DescTable>() * qs + u16s * (3 + qs))
    + align(u16s * 3 + core::mem::size_of::<UsedElem>() * qs);
  out as u32
}


macro_rules! read_into_buf {
  ($size: expr, $offset:expr, [ $( $per: ty $(,)?)* ] $(,)?) => {{
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

/*
macro_rules! read_buf_port_io {
  ($buf: expr, $base: expr, [ $( $per: ty $(,)?)* ] $(,)?) => {{
    let mut curr = 0;
    $(
      let raw: $per = unsafe {
        PortRead::read_from_port($base)
      };
      let len = (<$per>::BITS/8) as usize;
      $buf[curr..curr +len].copy_from_slice(&raw.to_ne_bytes());
      curr += len;
    )*
  }}
}
*/

static mut VIRT_QUEUE: VirtQueueMetadata<256> = VirtQueueMetadata::new();

fn read_pci_u16(cfg_val: u32) -> u16 {
  unsafe {
    // Sets the pointer in configuration space
    PortWrite::write_to_port(CONFIG_ADDRESS, cfg_val);
    // Read from the pointer in configuration space
    core::hint::black_box(PortRead::read_from_port(CONFIG_DATA))
  }
}
fn read_pci_u32(cfg_val: u32) -> u32 {
  unsafe {
    // Sets the pointer in configuration space
    PortWrite::write_to_port(CONFIG_ADDRESS, cfg_val);
    // Read from the pointer in configuration space
    PortRead::read_from_port(CONFIG_DATA)
  }
}

/// Initializes the block device on PCI
pub fn init_block_device_on_pci() {
  let buf = read_into_buf!(4, 0, [
    u32,
  ]);
  let (vendor_id, device_id): (u16, u16) = unsafe { core::mem::transmute(buf) };
  assert_eq!(vendor_id, 0x1af4, "Not a PCI virtio device: {:x}", vendor_id);
  assert!(
    0x1000 < device_id && device_id < 0x107f,
    "Not a PCI virtio device: {:x} {:x}", device_id, vendor_id
  );

  if device_id == 0x1001 {
    // assert_eq!(header.vendor_id, 0x1001, "Not a legacy block device");
    let mut buf = read_into_buf!(core::mem::size_of::<CommonHeader>(), 0, [
      u32, u32, u32, u32
    ]);
    let mut header: CommonHeader = unsafe { core::mem::transmute(buf) };
    let mut buf = read_into_buf!(core::mem::size_of::<PCIHeader0>(), 0x10, [
      u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32,
    ]);
    let header0: PCIHeader0 = unsafe { core::mem::transmute(buf) };
    legacy_init(header, header0);
  } else if device_id == 0x1042 {
    // assert_eq!(header.vendor_id, 0x1042, "Not a modern block device");
    modern_init();
  } else {
    todo!();
  }
}

fn modern_init() {
  let buf = read_into_buf!(4, 4, [
    u32,
  ]);
  let (command, status): (Command, Status) = unsafe { core::mem::transmute(buf) };
  assert!(!status.master_data_parity_error(), "{:b} {:b}", status.0, command.0);

  if !command.bus_master() {
    let v: u16 = unsafe {
      // Sets the pointer in configuration space
      PortWrite::write_to_port(CONFIG_ADDRESS, cfg(0, 4, 0, 4));
      // Read from the pointer in configuration space
      let new_header = command
        .set_bus_master(true)
        .set_io_space(true)
        .set_mem_space(true);
      PortWrite::write_to_port(CONFIG_DATA, new_header.0);
      PortRead::read_from_port(CONFIG_DATA)
    };
    let command = Command::from(v);
    let v: u16 = unsafe {
      // Sets the pointer in configuration space
      PortWrite::write_to_port(CONFIG_ADDRESS, cfg(0, 4, 0, 4));
      // Read from the pointer in configuration space
      PortRead::read_from_port(CONFIG_DATA)
    };
    write!(vga_buffer::Writer::new(8, 10), "{:b}", status.0);
  }
}

fn legacy_init(header: CommonHeader, header0: PCIHeader0) {
  // ---- Finished reading in headers, Start reading legacy configuration in
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
  let base_addr = bar0.addr() as u16;
  unsafe {
    VIRT_QUEUE.base_addr = base_addr;
  }

  unsafe {
    PortWrite::write_to_port(base_addr + 18, VirtioStatus::empty().0);
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);

    let v: u8 = PortRead::read_from_port(base_addr + 18);
    PortWrite::write_to_port(base_addr + 18, VirtioStatus::from(v).set_ack(true).0);
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);

    let v: u8 = PortRead::read_from_port(base_addr + 18);
    PortWrite::write_to_port(base_addr + 18, VirtioStatus::from(v).set_driver(true).0);
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);

    let device_features: u32 = PortRead::read_from_port(base_addr);
    // Feature negotiation
    let driver_features = VirtioBlkFeatures::from(device_features)
      .set_read_only(false)
      .set_blk_size(false)
      .set_topology(false);
    PortWrite::write_to_port(base_addr + 4, driver_features.0);

    let v: u8 = PortRead::read_from_port(base_addr + 18);
    PortWrite::write_to_port(
      base_addr + 18,
      VirtioStatus::from(v).set_features_ok(true).0,
    );
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    let v: u8 = PortRead::read_from_port(base_addr + 18);
    assert!(VirtioStatus::from(v).features_ok(), "Device did not accept features");
  }

  /*
  Perform device-specific setup, including discovery of virtqueues for the device, optional
  per-bus setup, reading and possibly writing the device’s virtio configuration space, and
  population of virtqueues.
  */
  assert_eq!(
    core::mem::size_of::<VirtQueue<256>>(),
    8192,
    "VirtQueue size changed",
  );

  assert_eq!(core::mem::align_of_val(unsafe { &VIRT_QUEUE }), 4096);

  unsafe {
    // Queue-select <-- 0, select 0th queue
    PortWrite::write_to_port(base_addr + 14, 0u16);
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);

    // Read from Queue-size --> Get requested size for 0th queue
    let v: u16 = PortRead::read_from_port(base_addr + 12);
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);

    assert_ne!(v, 0, "VirtQueue[0] does not exist");
    assert_eq!(v, 256, "unconfigured memory allocations: incorrect size");
    //assert_eq!(0, virtqueue_size(v));

    assert_eq!(core::mem::align_of_val(unsafe { &VIRT_QUEUE.vq.descriptors }), 16);
    assert_eq!(core::mem::align_of_val(unsafe { &VIRT_QUEUE.vq.avail }), 2);
    assert_eq!(core::mem::align_of_val(unsafe { &VIRT_QUEUE.vq.used }), 4);

    // Queue-address <-- VirtQueue Address divided by 4096 for legacy spec?
    assert!((unsafe { &VIRT_QUEUE.vq } as *const _ as usize) < (u32::MAX as usize));
    assert_eq!(unsafe { &VIRT_QUEUE.vq } as *const _ as usize % 4096, 0);
    let vq_addr = ((&VIRT_QUEUE.vq) as *const _ as u32) / 4096;
    PortWrite::write_to_port(base_addr + 8, vq_addr);
    assert_eq!(vq_addr, PortRead::read_from_port(base_addr + 8));
  }

  // Mark driver as ready to drive
  unsafe {
    let v: u8 = PortRead::read_from_port(base_addr + 18);
    PortWrite::write_to_port(base_addr + 18, VirtioStatus::from(v).set_driver_ok(true).0);
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    let v: u8 = PortRead::read_from_port(base_addr + 18);
    let v = VirtioStatus::from(v);
    assert!(v.driver_ok());
    assert!(v.features_ok());

    let v: u16 = PortRead::read_from_port(base_addr + 16);

    PortWrite::write_to_port(base_addr + 16, 0u16);
  }

  unsafe {
    crate::IDT[32 + header0.interrupt_line as usize].set_handler_fn(virtio_blk_handler);
    crate::IDT.load();
  }

  static mut EXAMPLE: [u8; 4096] = [123; 4096];
  unsafe {
    VIRT_QUEUE
      .blk_cmd(VirtioBlkRequestType::Write, 0, &EXAMPLE)
      .expect("Failed to write");
  }
}

use x86_64::structures::idt::InterruptStackFrame;
extern "x86-interrupt" fn virtio_blk_handler(stack_frame: &mut InterruptStackFrame) {
  panic!("{:?}", stack_frame);
}

bit_register!(LegacyDeviceStatus, u8, [
  (0, ack, set_ack),
  (1, driver, set_driver),
  (2, driver_ok, set_driver_ok),
  (3, features_ok, set_features_ok),
  (6, device_needs_reset),
  (7, failed, set_failed),
]);

bit_register!(VirtioBlkFeatures, u32, [
  (1, size_max, set_size_max),
  (2, seg_max, set_seg_max),
  (4, geometry, set_geometry),
  (5, read_only, set_read_only),
  (6, blk_size, set_blk_size),
  (9, flush, set_flush),
  (10, topology, set_topology),
]);

/// Found in bar0 on legacy virtio devices
#[repr(C)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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
  device_status: LegacyDeviceStatus,
  // R
  isr_status: u8,
}

/// Is after the LegacyVirtio header
#[derive(Debug)]
#[repr(C)]
struct VirtioBlkCfg {
  // capacity is number of 512 byte sectors
  // note when using a file as a disk I need to manually make the file big to give it sectors.
  capacity: u64,
  size_max: u32,
  seg_max: u32,
  cylinders: u16,
  heads: u8,
  sectors: u8,
  blk_size: u32,
  // # of logical blocks per physical block (log2)
  physical_block_exp: u8,
  // offset of first aligned logical block
  alignment_offset: u8,
  // suggested minimum I/O size in blocks
  min_io_size: u16,
  // optimal (suggested maximum) I/O size in blocks
  opt_io_size: u32,
  writeback: u8,
  _unused0: [u8; 3],
  max_discard_sectors: u32,
  max_discard_seg: u32,
  discard_sector_alignment: u32,
  max_write_zeroes_sectors: u32,
  max_write_zeroes_seg: u32,
  write_zeroes_may_unmap: u8,
  _unused1: [u8; 3],
}

bit_register!(VirtioStatus, u8, [
  (0, ack, set_ack),
  (1, driver, set_driver),
  (2, driver_ok, set_driver_ok),
  (3, features_ok, set_features_ok),
  (6, device_needs_reset),
  (7, failed),
]);

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VirtioBlkRequest {
  ty: VirtioBlkRequestType,
  _reserved: u32,
  sector: u64,
  status: VirtioBlkStatus,
}

impl VirtioBlkRequest {
  const fn new(ty: VirtioBlkRequestType, sector: u64) -> Self {
    Self {
      ty,
      _reserved: 0,
      sector,
      status: VirtioBlkStatus::Ok,
    }
  }
}

/// What kind of request is being sent to
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtioBlkRequestType {
  /// BLK_T_IN in documentation
  Read = 0,
  /// BLK_T_OUT in documentation
  Write = 1,
  /// Flushes either a write or a read
  Flush = 4,
  Discard = 11,
  /// Zeroes out some blocks
  WriteZero = 13,
}

/// Defines the status of some block request.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtioBlkStatus {
  Ok = 0,
  Err = 1,
  /// Unsupported request type
  Unsupported = 2,
}
