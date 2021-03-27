use crate::block_interface::{Metadata, GLOBAL_BLOCK_INTERFACE};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FileDescriptor(u8);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct INode {
  blocks: [u32; 8],
  in_use: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Superblock {
  magic_number: u32,
  inode_start_block: u32,
  num_inodes: u32,
  data_start_block: u32,
  num_data: u32,
}

pub struct FileSystem {
  superblock: Superblock,
  file_descriptors: [FileDescriptor; 256],
}

impl Metadata for INode {
  fn new() -> Self {
    INode {
      blocks: [0; 8],
      in_use: 0,
    }
  }
  fn owned(&self) -> &[u32] { &self.blocks[..self.in_use as usize] }
  fn insert(&self, b: u32) -> Result<Self, ()> {
    if (self.in_use as usize) == self.blocks.len() {
      return Err(());
    }
    let mut out = self.clone();
    out.blocks[out.in_use as usize] = b;
    out.in_use += 1;
    Ok(out)
  }
  fn remove(&self, b: u32) -> Result<Self, ()> {
    if self.in_use == 0 {
      return Err(());
    }
    let mut out = self.clone();
    out.in_use -= 1;
    out.blocks[out.in_use as usize] = 0;
    Ok(out)
  }
}

impl FileSystem {
  pub fn new() -> Self { todo!() }
  pub fn open(&self) -> Result<FileDescriptor, ()> { todo!() }
  pub fn close(&self, fd: FileDescriptor) -> Result<(), ()> { todo!() }
}
