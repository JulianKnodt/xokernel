use crate::block_interface::BlockDevice;
#[derive(Debug)]
pub struct Driver {}

impl BlockDevice for Driver {
  const NUM_BLOCKS: usize = 4096;
  const BLOCK_SIZE: usize = 512;
  fn read(&self, block_num: u32, dst: &mut [u8]) -> Result<usize, ()> { todo!() }
  fn write(&self, block_num: u32, src: &[u8]) -> Result<usize, ()> { todo!() }
}

impl Driver {
  const fn new() -> Self { Driver {} }
}

pub static mut DRIVER: Driver = Driver::new();

pub fn init() {}
