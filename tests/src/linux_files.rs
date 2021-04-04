use crate::block_interface::BlockDevice;
use std::{
  fs::File,
  io::{Read, Seek, SeekFrom, Write},
  sync::Mutex,
};

#[derive(Debug)]
pub struct Driver {
  backing: Option<Mutex<File>>,
}

impl Driver {
  pub const fn new() -> Self { Self { backing: None } }
}

impl BlockDevice for Driver {
  const NUM_BLOCKS: usize = 4096;
  const BLOCK_SIZE: usize = 512;
  fn read(&self, block_num: u32, dst: &mut [u8]) -> Result<usize, ()> {
    let f = self.backing.as_ref().unwrap();
    let mut f = f.lock().expect("Failed to lock file for reading");
    f.seek(SeekFrom::Start(block_num as u64 * Self::BLOCK_SIZE as u64))
      .expect("Failed to seek");
    let len = dst.len().min(Self::BLOCK_SIZE);
    f.read(&mut dst[..len]).map_err(|_| ())
  }
  fn write(&self, block_num: u32, dst: &[u8]) -> Result<usize, ()> {
    let f = self.backing.as_ref().unwrap();
    let mut f = f.lock().expect("Failed to lock file for writing");
    f.seek(SeekFrom::Start(block_num as u64 * Self::BLOCK_SIZE as u64))
      .expect("Failed to seek");
    f.write(dst).map_err(|_| ())
  }
  fn init(&mut self) {
    let backing = File::with_options()
      .read(true)
      .write(true)
      .create(true)
      .open("diskblocks")
      .expect("Failed to open diskblocks file");
    backing
      .set_len(Self::NUM_BLOCKS as u64 * Self::BLOCK_SIZE as u64)
      .expect("Failed to set size");
    self.backing = Some(Mutex::new(backing));
  }
}
