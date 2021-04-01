use crate::block_interface::BlockDevice;
use std::fs::File;
use std::sync::RwLock;
use std::io::{Seek, SeekFrom, Read, Write};

#[derive(Debug)]
pub struct Driver {
  backing: Option<RwLock<File>>,
}

impl Driver {
  pub const fn new() -> Self {
    Self { backing: None }
  }
}

impl BlockDevice for Driver {
  const NUM_BLOCKS: usize = 4096;
  const BLOCK_SIZE: usize = 512;
  fn read(&self, block_num: u32, dst: &mut [u8]) -> Result<usize, ()> {
    let f = self.backing.as_ref().unwrap();
    let mut f = f.write().expect("Failed to lock file for writing");
    f.seek(SeekFrom::Start(block_num as u64 * Self::BLOCK_SIZE as u64)).expect("Failed to seek");
    f.read(&mut dst[..Self::BLOCK_SIZE]).map_err(|_| ())
  }
  fn write(&self, block_num: u32, dst: &[u8]) -> Result<usize, ()> {
    let f = self.backing.as_ref().unwrap();
    let mut f = f.write().expect("Failed to lock file for writing");
    f.seek(SeekFrom::Start(block_num as u64 * Self::BLOCK_SIZE as u64)).expect("Failed to seek");
    f.write(dst).map_err(|_| ())
  }
  fn init(&mut self) {
    let backing = File::with_options()
      .read(true)
      .write(true)
      .create(true)
      .open("diskblocks")
      .expect("Failed to open diskblocks file");
    self.backing = Some(RwLock::new(backing));
  }
}
