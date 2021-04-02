use crate::{
  bit_array::nearest_div_8,
  block_interface::{
    AllMetadata, BlockDevice, GlobalBlockInterface, Metadata, MetadataHandle, Owner,
    FIRST_FREE_BLOCK,
  },
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

impl Default for FileMode {
  fn default() -> Self { Self::R }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
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
pub struct Superblock {
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
  type Iter<'a> = core::iter::Copied<core::slice::Iter<'a, u32>>;
  fn owned(&self) -> Self::Iter<'_> {
    match self.owned {
      None => [].iter(),
      Some(ref v) => v.iter(),
    }
    .copied()
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
  const LEN: usize = core::mem::size_of::<Self>();
  default_ser_impl!();
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeMetadata {
  start: u32,
  count: u32,
}

impl Metadata for RangeMetadata {
  fn new() -> Self { RangeMetadata { start: 0, count: 0 } }
  type Iter<'a> = core::ops::Range<u32>;
  fn owned(&self) -> Self::Iter<'_> { self.start..self.start + self.count }
  const LEN: usize = core::mem::size_of::<Self>();
  fn insert(&self, b: u32) -> Result<Self, ()> {
    if self.count == 0 {
      return Ok(Self { start: b, count: 1 });
    }
    // Handle an extra block after the end of this one
    if b == self.start + self.count + 1 {
      return Ok(Self {
        start: self.start,
        count: self.count + 1,
      });
    } else if b == self.start.checked_sub(1).ok_or(())? {
      return Ok(Self {
        start: self.start - 1,
        count: self.count + 1,
      });
    }
    Err(())
  }

  fn remove(&self, b: u32) -> Result<Self, ()> {
    todo!();
  }

  default_ser_impl!();
}

pub struct FileSystem<B: 'static + BlockDevice>
where
  B: BlockDevice,
  [(); nearest_div_8(B::NUM_BLOCKS)]: ,
  [(); B::BLOCK_SIZE]: , {
  superblock: MetadataHandle,
  open_files: [FileDescEntry; 256],
  inode_md: MetadataHandle,
  data_md: MetadataHandle,
  root: u32,
  gbi: &'static mut GlobalBlockInterface<B>,
}

impl<B> FileSystem<B>
where
  B: BlockDevice,
  [(); nearest_div_8(B::NUM_BLOCKS)]: ,
  [(); B::BLOCK_SIZE]: ,
{
  pub fn new(gbi: &'static mut GlobalBlockInterface<B>) -> Self {
    let sb_mh = gbi.metadatas_for(Owner::LibFS).find_map(|(mh, amd)| {
      if matches!(amd, AllMetadata::Superblock(_)) {
        Some(mh)
      } else {
        None
      }
    });
    if sb_mh.is_some() {
      // File system previously existed, reinitialize
      todo!();
    } else {
      // Need to build new FS
      return Self::mk(gbi);
    }
  }
  /// Makes a new instance of this file system on this block interface
  pub fn mk(gbi: &'static mut GlobalBlockInterface<B>) -> Self {
    let sb_mh = gbi
      .new_metadata::<Superblock>(Owner::LibFS)
      .expect("Failed to mk sb md");
    gbi
      .req_block(sb_mh, FIRST_FREE_BLOCK)
      .expect("Failed to initialize Super block");
    let inode_mh = gbi
      .new_metadata::<RangeMetadata>(Owner::LibFS)
      .expect("Failed to mk inode md");
    for i in FIRST_FREE_BLOCK + 1..FIRST_FREE_BLOCK + 1 + 256 {
      gbi
        .req_block(inode_mh, i)
        .expect("Failed to initialize Super block");
    }
    let data_mh = gbi
      .new_metadata::<Superblock>(Owner::LibFS)
      .expect("Failed to mk data md");
    for i in FIRST_FREE_BLOCK + 1 + 256..FIRST_FREE_BLOCK + 1 + 256 + 1024 {
      gbi
        .req_block(data_mh, i)
        .expect("Failed to initialize Super block");
    }
    Self {
      superblock: sb_mh,
      inode_md: inode_mh,
      data_md: data_mh,
      open_files: [FileDescEntry::default(); 256],
      // which block contains the first free node
      root: FIRST_FREE_BLOCK + 1,
      gbi,
    }
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
