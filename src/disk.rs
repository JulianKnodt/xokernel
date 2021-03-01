#[derive(Debug, PartialEq, Default, Clone, Copy)]
struct DiskCap {
  start_block: u32,
  len: u32,
}

#[derive(Debug, PartialEq, Copy, Clone)]
struct CapabilityHandle(usize);
/*
static mut DISK_CAPS: [DiskCap; 256] = [DiskCap {
  start_block: 0,
  len: 0,
}; 256];
static mut DISK_CAPS_INUSE: u8 = 0;
*/
