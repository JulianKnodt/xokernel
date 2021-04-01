use crate::{
  block_interface::{global_block_interface, Metadata, MetadataHandle, Owner, FIRST_FREE_BLOCK},
  default_ser_impl,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FileDescriptor(u32);

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FileMode {
  R,
  W,
  RW,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FileDescEntry {
  inode: u32,
  offset: u32,
  open_refs: u16,
  mode: FileMode,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Directory {
  parent_inode: u32,
  num_entries: u32,
  // don't need to read through all of the entries here
  inode_to_names: [([u8; 16], u16); 128],
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct INode {
  refs: u16,
  kind: INodeKind,
  file_size: u32,
  data_blocks: [u16; 8],
}

#[repr(u16)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum INodeKind {
  File = 0,
  Directory = 1,
}

const MAGIC_NUMBER: u32 = 0x101_0_F_0ff;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Superblock {
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
  open_files: [FileDescEntry; 256],
  metadatas: [Option<MetadataHandle>; 256],
  root: u32,
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
        .req_block::<Superblock>(sb_md, FIRST_FREE_BLOCK)
        .expect("Failed to allocate block for superblock");
    }

    // TODO read from superblock
    todo!()
  }
  /// Opens a given path (always absolute)
  pub fn open(&self, path: &[&str], mode: FileMode) -> Result<FileDescriptor, ()> {
    let mut curr_inode: u32 = self.root;
    for segment in path {}
    todo!()
  }
  pub fn close(&mut self, FileDescriptor(i): FileDescriptor) -> Result<(), ()> {
    let i = i as usize;
    let open_file = &mut self.open_files[i];
    if open_file.open_refs == 0 {
      return Err(());
    }
    open_file.open_refs -= 1;
    // If inode is no longer referenced here, need to delete it
    todo!();
    return Ok(());
  }
}
