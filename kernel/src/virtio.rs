use crate::{block_interface::BlockDevice, pci};
#[derive(Debug, PartialEq, Eq)]
pub struct Driver {}

impl BlockDevice for Driver {
  const NUM_BLOCKS: usize = 4096;
  const BLOCK_SIZE: usize = 512;
  fn read(&self, block_num: u32, dst: &mut [u8]) -> Result<usize, ()> { todo!() }
  fn write(&self, block_num: u32, src: &[u8]) -> Result<usize, ()> { todo!() }
  fn init(&mut self) {
    // TODO get headers from here?
    let header = pci::init_block_device_on_pci();
    // Reset device
    // Set Ack bit
    // read device feature bits, write subset of features bits supported by us
    // Set FeatOk status bit
    // Reread device status to ensure that feature bits all still set
    // discovery of virtqueues
    // write virtio config space
    // populate virtqueues
  }
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

#[repr(C)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct VirtioPCICap {
  cap_vndr: u8,              /* Generic PCI field: PCI_CAP_ID_VNDR */
  cap_next: u8,              /* Generic PCI field: next ptr. */
  cap_len: u8,               /* Generic PCI field: capability length */
  cfg_type: VirtioPCICapCfg, /* Identifies the structure. */
  bar: u8,                   /* Where to find it. */
  padding: [u8; 3],          /* Pad to full dword. */
  // The fields below should be encoded in memory as little endian
  offset: u32, /* Offset within bar. */
  length: u32, /* Length of the structure, in bytes. */
}

#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum VirtioDeviceStatus {
  Ack = 1,
  Driver = 2,
  Failed = 128,
  FeaturesOk = 8,
  DriverOk = 4,
  DeviceNeedsReset = 64,
}

impl Driver {
  pub const fn new() -> Self { Driver {} }
}

const VIRTIO_MAGIC: u32 = 0x74726976;

#[repr(packed)]
struct VirtioRegs {
  magic: u32,
  version: u32,
  device_id: u32,
  vendor_id: u32,
  device_features: u32,
  device_features_sel: u32,
  _reserved0: [u32; 2],
  driver_features: u32,
  driver_features_sel: u32,
  _reserved1: [u32; 2],
  queue_sel: u32,
  queue_num_max: u32,
  queue_num: u32,
  _reserved2: [u32; 2],
  queue_ready: u32,
  _reserved3: [u32; 2],
  queue_notify: u32,
  _reserved4: [u32; 3],
  interrupt_status: u32,
  interrupt_ack: u32,
  _reserved5: [u32; 5],
  status: u32,
  _reserved6: [u32; 3],
  queue_desc_low: u32,
  queue_desc_high: u32,
  _reserved7: [u32; 2],
  queue_avail_low: u32,
  queue_avail_high: u32,
  _reserved8: [u32; 2],
  queue_used_low: u32,
  queue_used_high: u32,
  _reserved9: [u32; 21],
  config_generation: u32,
  // Config: [u32; 0],
}

/*
struct VirtualQueue {
  struct Buffers[QueueSize] {
    uint64_t Address; // 64-bit address of the buffer on the guest machine.
    uint32_t Length;  // 32-bit length of the buffer.
    uint16_t Flags;   // 1: Next field contains linked buffer index;  2: Buffer is write-only (clear for read-only).
                      // 4: Buffer contains additional buffer addresses.
    uint16_t Next;    // If flag is set, contains index of next buffer in chain.
  }
  struct Available {
    uint16_t Flags;             // 1: Do not trigger interrupts.
    uint16_t Index;             // Index of the next ring index to be used.  (Last available ring buffer index+1)
    uint16_t [QueueSize] Ring;  // List of available buffer indexes from the Buffers array above.
    uint16_t EventIndex;        // Only used if VIRTIO_F_EVENT_IDX was negotiated
  }
  uint8_t[] Padding;  // Reserved
  // 4096 byte alignment
  struct Used {
    uint16_t Flags;            // 1: Do not notify device when buffers are added to available ring.
    uint16_t Index;            // Index of the next ring index to be used.  (Last used ring buffer index+1)
    struct Ring[QueueSize] {
      uint32_t Index;  // Index of the used buffer in the Buffers array above.
      uint32_t Length; // Total bytes written to buffer.
    }
    uint16_t AvailEvent;       // Only used if VIRTIO_F_EVENT_IDX was negotiated
  }
}
*/
