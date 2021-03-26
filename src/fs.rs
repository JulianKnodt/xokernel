const BLOCK_SIZE: usize = 4096;
pub trait BlockDevice {
  fn read(&self, block_num: u32, into: &mut [u8; BLOCK_SIZE]) -> Result<(), ()>;
  fn write(&self, block_num: u32, data: &[u8; BLOCK_SIZE]) -> Result<(), ()>;
  fn num_blocks(&self) -> usize;
}

pub trait BlockDeviceInterface {
  fn read(&self, block_num: u32, into: &mut [u8; BLOCK_SIZE]) -> Result<(), ()>;
  fn write(&self, block_num: u32, data: &[u8; BLOCK_SIZE]) -> Result<(), ()>;
}

pub trait Metadata {
  fn new() -> Self;
  fn owned(&self) -> &[u32];
  fn insert(&mut self, b: u32) -> Result<(), ()>;
  fn remove(&mut self, b: u32) -> Result<(), ()>;
}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FileDescriptor(u8);
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct INode {
  blocks: [u32; 8],
  in_use: u8,
}

#[derive(Debug)]
pub struct BitMap {
  data: [u8; 512],
}

impl BitMap {
  pub fn get(&self, i: usize) -> bool {
    let idx = i / 8;
    ((self.data[idx] >> (i % 8)) & 0b1) == 1
  }
  pub fn set(&mut self, i: usize) -> bool {
    let idx = i / 8;
    let old = ((self.data[idx] >> (i % 8)) & 0b1) == 1;
    self.data[idx] |= 1 << (i % 8);
    old
  }
  pub fn unset(&mut self, i: usize) -> bool {
    let idx = i / 8;
    let old = ((self.data[idx] >> (i % 8)) & 0b1) == 1;
    self.data[idx] &= !(1 << (i % 8));
    old
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Superblock {
  magic_number: u32,
  inode_start_block: u32,
  num_inodes: u32,
  data_start_block: u32,
  num_data: u32,
}

pub struct FileSystem<B: BlockDevice> {
  block_device: B,
  // Fixed size of 4096 blocks
  free_map: BitMap,
  file_descriptors: [FileDescriptor; 256],
  superblock: Superblock,
}

impl Metadata for INode {
  fn new() -> Self {
    INode {
      blocks: [0; 8],
      in_use: 0,
    }
  }
  fn owned(&self) -> &[u32] { &self.blocks[..self.in_use as usize] }
  fn insert(&mut self, b: u32) -> Result<(), ()> {
    if self.in_use >= 8 {
      return Err(());
    }
    self.in_use += 1;
    self.blocks[self.in_use as usize - 1] = b;
    Ok(())
  }
  fn remove(&mut self, b: u32) -> Result<(), ()> {
    if self.in_use == 0 {
      return Err(());
    }
    self.in_use -= 1;
    self.blocks[self.in_use as usize] = 0;
    Ok(())
  }
}

/*
// need one block for superblock at beginning
impl<B: BlockDevice> Metadata for FileSystem<B> {
  fn owned(&self) -> &[u32] { &[0] }
  fn insert(&mut self, b: u32) -> Result<(), ()> { return Err(()) }
  fn remove(&mut self, b: u32) -> Result<(), ()> { return Err(()) }
}
*/

impl<B: BlockDevice> FileSystem<B> {
  pub fn new() -> Self { todo!() }
  pub fn open(&self) -> Result<FileDescriptor, ()> { todo!() }
  pub fn close(&self, fd: FileDescriptor) -> Result<(), ()> { todo!() }
}
