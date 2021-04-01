use crate::{block_interface::BlockDevice, pci};
#[derive(Debug)]
pub struct Driver {}

impl BlockDevice for Driver {
  const NUM_BLOCKS: usize = 4096;
  const BLOCK_SIZE: usize = 512;
  fn read(&self, block_num: u32, dst: &mut [u8]) -> Result<usize, ()> { todo!() }
  fn write(&self, block_num: u32, src: &[u8]) -> Result<usize, ()> { todo!() }
  fn init(&mut self) {}
}

impl Driver {
  pub const fn new() -> Self { Driver {} }
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
