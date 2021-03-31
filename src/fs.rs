use crate::{
  block_interface::{global_block_interface, Metadata, MetadataHandle, Owner},
  default_ser_impl,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FileDescriptor(u8);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct INode {
  blocks: [u32; 8],
  in_use: u8,
}

const MAGIC_NUMBER: u32 = 0x101_0_F_0ff;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Superblock {
  magic_number: u32,
  num_inodes: u32,
  num_data: u32,
  owned: Option<[u32; 1]>,
}

impl Metadata for Superblock {
  fn new() -> Self {
    Superblock {
      magic_number: MAGIC_NUMBER,
      num_inodes: 128,
      num_data: 128,
      owned: None,
    }
  }
  fn owned(&self) -> &[u32] {
    match self.owned {
      None => &[],
      Some(ref v) => v,
    }
  }
  fn insert(&self, b: u32) -> Result<Self, ()> {
    if self.owned.is_some() {
      return Err(());
    }
    Ok(Self {
      owned: Some([b]),
      ..*self
    })
  }
  fn remove(&self, b: u32) -> Result<Self, ()> {
    if self.owned.is_none() {
      return Err(());
    }
    Ok(Self {
      owned: None,
      ..*self
    })
  }
  fn len(&self) -> usize { core::mem::size_of::<Self>() }
  default_ser_impl!();
}

pub struct FileSystem {
  superblock: MetadataHandle,
  file_descriptors: [FileDescriptor; 256],
  metadatas: [Option<MetadataHandle>; 256],
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
  fn len(&self) -> usize { core::mem::size_of::<Self>() }
  default_ser_impl!();
}

impl FileSystem {
  pub fn new(sb_md: Option<MetadataHandle>) -> Self {
    let sb_md = sb_md.unwrap_or_else(|| {
      global_block_interface()
        .new_metadata::<Superblock>(Owner::LibFS)
        .expect("Failed to allocate superblock metadata")
    });
    let sb = global_block_interface()
      .md_ref::<Superblock>(sb_md)
      .unwrap();
    if sb.owned().is_empty() {
      global_block_interface()
        .req_block::<Superblock>(sb_md, 0)
        .expect("Failed to allocate block for superblock");
    }

    // TODO read from superblock
    todo!()
  }
  pub fn open(&self) -> Result<FileDescriptor, ()> { todo!() }
  pub fn close(&self, fd: FileDescriptor) -> Result<(), ()> { todo!() }
}
