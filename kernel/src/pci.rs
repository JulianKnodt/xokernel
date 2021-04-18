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
  }
}

// Location of where things are required to be accessed(?)
const CONFIG_ADDRESS: u16 = 0x0cf8;
// Actually generates configuration data to read
const CONFIG_DATA: u16 = 0x0cfc;
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
  used_event: u16,
}

impl<const QS: usize> AvailableRing<QS> {
  const fn new() -> Self {
    Self {
      flags: VirtAvailFlags::empty(),
      idx: 0,
      ring: [0; QS],
      used_event: 0,
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
  avail_event: u16,
}

impl<const QS: usize> UsedRing<QS> {
  const fn new() -> Self {
    Self {
      flags: VirtQUsedFlags::empty(),
      idx: 0,
      used_elems: [UsedElem { idx: 0, len: 0 }; QS],
      avail_event: 0,
    }
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct UsedElem {
  idx: u32,
  len: u32,
}

const Q_ALIGN: usize = 4096;
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
  fn blk_cmd<const N: usize>(
    &mut self,
    ty: VirtioBlkRequestType,
    sector: u64,
    data: &[u8],
  ) -> Result<(), ()> {
    // TODO need to allocate this somewhere long-lived
    let req: VirtioBlkRequest<0> = VirtioBlkRequest::new(ty, sector);
    let req_desc = self.alloc_descriptor((&req) as *const _ as u64)?;
    let req_entry = &mut self.vq.descriptors[req_desc as usize];
    req_entry.len = 16;
    req_entry.flags = req_entry.flags.set_next(true);

    let data_desc = self.alloc_descriptor(data.as_ptr() as u64)?;
    let data_entry = &mut self.vq.descriptors[data_desc as usize];
    data_entry.len = 512;
    data_entry.flags = data_entry.flags.set_next(true).set_write_only(true);

    let status_desc = self.alloc_descriptor((&req.status as *const _) as u64)?;
    let status_entry = &mut self.vq.descriptors[status_desc as usize];
    status_entry.len = 1;
    status_entry.flags = status_entry.flags.set_write_only(true);

    self.vq.descriptors[req_desc as usize].next = data_desc;
    self.vq.descriptors[data_desc as usize].next = status_desc;

    self.vq.avail.ring[self.vq.avail.idx as usize] = data_desc;
    self.vq.avail.idx += 1;

    unsafe {
      PortWrite::write_to_port(self.base_addr + 16, 0u16);
    }

    Ok(())
  }
}

fn virtqueue_size(queue_size: u16) -> u32 {
  let align = |v| (v + Q_ALIGN) & (!Q_ALIGN);
  let qs = queue_size as usize;
  let u16s = core::mem::size_of::<u16>();
  let out = align(core::mem::size_of::<DescTable>() * qs + u16s * (3 + qs))
    + align(u16s * 3 + core::mem::size_of::<UsedElem>() * qs);
  out as u32
}

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

static mut VIRT_QUEUE: VirtQueueMetadata<256> = VirtQueueMetadata::new();

/// Initializes the block device on PCI
pub fn init_block_device_on_pci() -> PCIHeader0 {
  let mut buf = read_into_buf!(core::mem::size_of::<CommonHeader>(), 0, [
    u32, u32, u32, u32
  ]);
  let mut header: CommonHeader = unsafe { core::mem::transmute(buf) };
  assert_eq!(header.device_id, 0x1af4, "Not a PCI virtio device");
  assert_eq!(header.vendor_id, 0x1001, "Not a legacy block device");
  // assert_eq!(header.vendor_id, 0x1042, "Not a modern block device");
  assert!(
    0x1000 < header.vendor_id && header.vendor_id < 0x107f,
    "Not a PCI virtio device"
  );

  assert_eq!(header.header_type, 0);
  assert!(!header.command.interrupt_disable());

  let mut buf = read_into_buf!(core::mem::size_of::<PCIHeader0>(), 0x10, [
    u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32,
  ]);
  let header0: PCIHeader0 = unsafe { core::mem::transmute(buf) };
  assert_eq!(header0.interrupt_pin, 0);

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
  let mut buf = [0u8; 128];
  let base_addr = bar0.addr();

  for i in 0..mem {
    let v: u8 = unsafe { PortRead::read_from_port(base_addr as u16 + i as u16) };
    buf[i as usize] = v;
  }

  let legacy_cfg: LegacyVirtioCommonCfg =
    unsafe { *(buf.as_ptr() as *const LegacyVirtioCommonCfg) };
  let rest = &buf[core::mem::size_of::<LegacyVirtioCommonCfg>()..];
  let virtio_blk_cfg: VirtioBlkCfg = unsafe { (rest.as_ptr() as *const VirtioBlkCfg).read() };
  assert_ne!(virtio_blk_cfg.capacity, 0, "Found empty virtio blk cfg");

  unsafe {
    PortWrite::write_to_port(base_addr as u16 + 18, VirtioStatus::Reset as u8);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    PortWrite::write_to_port(base_addr as u16 + 18, VirtioStatus::Ack as u8);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    PortWrite::write_to_port(base_addr as u16 + 18, VirtioStatus::Driver as u8);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    // TODO write some features we want? Not sure which are wanted
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

  assert_eq!(core::mem::align_of_val(unsafe { &VIRT_QUEUE }), 4096,);
  assert_eq!(unsafe { &VIRT_QUEUE } as *const _ as usize % 4096, 0);
  // write!(vga_buffer::Writer::new(7, 0), "{:x?}", legacy_cfg);

  unsafe {
    // Queue-select <-- 0, := select 0th queue
    PortWrite::write_to_port(base_addr as u16 + 14, 0u16);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    // Queue-size --> Get requested size for 0th queue
    let v: u16 = PortRead::read_from_port(base_addr as u16 + 12);

    assert_ne!(v, 0, "VirtQueue[0] does not exist");
    assert_eq!(
      v, 256,
      "Did not want to configure memory allocations: incorrect size"
    );
    // Queue-address <-- VirtQueue Address divided by 4096 for legacy spec?
    let vq_addr = (&VIRT_QUEUE.vq as *const _ as u32) / 4096;
    PortWrite::write_to_port(base_addr as u16 + 8, vq_addr);
    assert_eq!(vq_addr, PortRead::read_from_port(base_addr as u16 + 8));

    // Queue-notify <-- 0th index = available buffers FIXME not sure if this is needed in init
    PortWrite::write_to_port(base_addr as u16 + 16, 0u16);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }

  let vq = unsafe { core::ptr::read_volatile(&VIRT_QUEUE) };
  write!(vga_buffer::Writer::new(7, 0), "{:x?}", vq);

  // Mark driver as ready to drive
  unsafe {
    PortWrite::write_to_port(base_addr as u16 + 18, VirtioStatus::DriverOk as u8);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
  return header0;
}

fn virtio_blk_isr(interrupt_id: u32) {
  // TODO get block device
  // Acknowledge interrupt
}

bit_register!(LegacyDeviceStatus, u8, [
  (0, ack, set_ack),
  (1, driver, set_driver),
  (2, driver_ok, set_driver_ok),
  (3, features_ok, set_features_ok),
  (6, device_needs_reset),
  (7, failed, set_failed),
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
#[derive(Debug, Clone, PartialEq, Eq)]
struct VirtioBlkRequest<const N: usize> {
  ty: VirtioBlkRequestType,
  _reserved: u32,
  sector: u64,
  data: [[u8; 512]; N],
  status: VirtioBlkStatus,
}

impl<const N: usize> VirtioBlkRequest<N> {
  fn new(ty: VirtioBlkRequestType, sector: u64) -> Self {
    Self {
      ty,
      _reserved: 0x3d,
      sector,
      data: [[0; 512]; N],
      status: VirtioBlkStatus::Ok,
    }
  }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtioBlkRequestType {
  In = 0,
  Out = 1,
  Flush = 4,
  Discard = 11,
  Zero = 13,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtioBlkStatus {
  Ok = 0,
  Err = 1,
  Unsupported = 2,
}
